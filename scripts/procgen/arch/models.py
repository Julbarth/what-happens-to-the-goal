"""Actor-critic networks for the Attribute Maze architecture ablation: paper IMPALA CNN, IMPALA+LSTM, multi-token ViT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def orthogonal_init(module, gain=nn.init.calculate_gain("relu")):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x):
        out = self.conv1(F.relu(x))
        out = self.conv2(F.relu(out))
        return out + x


class ImpalaBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.res1 = ResidualBlock(c_out)
        self.res2 = ResidualBlock(c_out)

    def forward(self, x):
        x = F.max_pool2d(self.conv(x), kernel_size=3, stride=2, padding=1)
        return self.res2(self.res1(x))


class ImpalaModel(nn.Module):
    """Identical to train-procgen common/model.py ImpalaModel (same parameter names, so paper checkpoints load)."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.block1 = ImpalaBlock(in_channels, 16)
        self.block2 = ImpalaBlock(16, 32)
        self.block3 = ImpalaBlock(32, 32)
        self.fc = nn.Linear(32 * 8 * 8, 256)
        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))
        return F.relu(self.fc(F.relu(x).flatten(1)))


class ViTEncoder(nn.Module):
    """Pre-LN ViT over 8x8-pixel patch tokens (64 tokens for 64x64 input) with a flatten readout.

    The flatten readout (per-token LN -> Linear(dim->tok_dim) -> concat) keeps token identity. A global CLS/mean
    readout dilutes the few object patches ~1/64 against background tokens, and pooled-readout ViTs trained with PPO
    on the ManiSkill tasks ignored the image. No ReLU on the final projection.
    """

    def __init__(self, in_channels=3, img=64, patch=8, dim=128, depth=4, heads=4, mlp_ratio=4, tok_dim=16, out_dim=256):
        super().__init__()
        n_tok = (img // patch) ** 2
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, n_tok, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        # Separate layers rather than nn.TransformerEncoder, whose deepcopy gives every layer the same init.
        # dropout=0: the PPO ratio assumes the same function at rollout and update time.
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(dim, heads, dim * mlp_ratio, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.tok_proj = nn.Linear(dim, tok_dim)
        self.fc = nn.Linear(tok_dim * n_tok, out_dim)
        self.output_dim = out_dim

    def forward(self, x):
        t = self.patch_embed(x).flatten(2).transpose(1, 2)
        t = self.blocks(t + self.pos)
        return self.fc(self.tok_proj(self.norm(t)).flatten(1))


class ActorCritic(nn.Module):
    """Encoder -> optional LSTM core -> linear policy (gain 0.01) and value (gain 1) heads, as in CategoricalPolicy.

    The recurrent state is the concatenation [h, c] so it fits the flat hidden-state buffer of the PPO storage.
    `starts` marks observations that begin a new episode; (h, c) is zeroed before such an observation.
    """

    def __init__(self, arch, n_actions=15, lstm_hidden=256):
        super().__init__()
        self.arch = arch
        if arch in ("impala", "lstm"):
            self.embedder = ImpalaModel(3)
        elif arch == "vit":
            self.embedder = ViTEncoder()
        else:
            raise ValueError(arch)
        feat = self.embedder.output_dim
        self.recurrent = arch == "lstm"
        if self.recurrent:
            self.lstm = nn.LSTM(feat, lstm_hidden)
            for name, p in self.lstm.named_parameters():
                if "bias" in name:
                    nn.init.constant_(p, 0)
                else:
                    nn.init.orthogonal_(p, 1.0)
            feat = lstm_hidden
            self.state_dim = 2 * lstm_hidden
        else:
            self.state_dim = 1
        self.fc_policy = orthogonal_init(nn.Linear(feat, n_actions), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(feat, 1), gain=1.0)

    def initial_state(self, n, device):
        return torch.zeros(n, self.state_dim, device=device)

    def core(self, feats, state, starts):
        """feats (T*N, F) time-major, state (N, 2H), starts (T*N,). Returns outputs (T*N, H) and final state."""
        if not self.recurrent:
            return feats, state
        n = state.shape[0]
        hdim = self.lstm.hidden_size
        feats = feats.view(-1, n, feats.shape[-1])
        keep = (1.0 - starts.view(-1, n)).unsqueeze(-1)
        h, c = state[:, :hdim].unsqueeze(0), state[:, hdim:].unsqueeze(0)
        outs = []
        for t in range(feats.shape[0]):
            h, c = h * keep[t], c * keep[t]
            o, (h, c) = self.lstm(feats[t:t + 1], (h, c))
            outs.append(o[0])
        return torch.cat(outs, 0), torch.cat([h[0], c[0]], -1)

    def forward(self, obs_u8, state, starts):
        """obs_u8: uint8 (B, 3, 64, 64); /255 here matches ScaledFloatFrame exactly."""
        x = obs_u8.float() / 255.0
        hidden, state = self.core(self.embedder(x), state, starts)
        logits = self.fc_policy(hidden)
        dist = Categorical(logits=F.log_softmax(logits, dim=1))
        return dist, self.fc_value(hidden).reshape(-1), state
