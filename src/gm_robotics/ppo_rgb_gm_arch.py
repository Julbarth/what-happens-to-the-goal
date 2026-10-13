"""ppo_rgb_gm.py with an --arch flag: cnn (NatureCNN), lstm (recurrent PPO), vit / vit_relu (ViT image encoder)."""
# Modified from ManiSkill examples/baselines/ppo/ppo_rgb.py (Apache License 2.0), itself based on CleanRL (MIT). See NOTICE.
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal

# ManiSkill specific imports
import mani_skill.envs
import gm_robotics.envs  # noqa: F401  registers AttributeTable-v1 / LocationReach-v1
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    arch: str = "cnn"
    """policy architecture: cnn | lstm | vit | vit_relu (ViT with a ReLU after the CLS projection)"""
    env_variant: str = "gm"
    """GM-testbed variant passed to the env constructor (train and in-training eval)"""
    out_root: str = "runs"
    """directory that holds runs/{exp_name}"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"
    """the group of the run for wandb"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "all"
    """the environment rendering mode"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 64
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.2
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False
    lstm_reset_every_step: bool = False
    """diagnostic: zero the LSTM state before every step (removes recurrence, keeps the env-wise update path)"""
    mb_by_env: bool = False
    """diagnostic for non-recurrent archs: minibatch over whole env sequences exactly as the recurrent path does"""
    lr_warmup_iters: int = 0
    """linear lr warmup from lr/warmup to lr over this many iterations (0 = off, baseline)"""
    adam_eps: float = 1e-5
    """Adam epsilon (baseline 1e-5)"""
    log_diag: bool = True
    """log image-feature dead fraction and pre-clip gradient norms each iteration (no effect on training math)"""
    vit_readout: str = "cls"
    """vit image readout: cls (baseline) | mean (mean-pool patch tokens, no CLS token, then LN) | flatten (per-token proj, concat)"""
    vit_stem: str = "patch"
    """vit tokenizer: patch (baseline linear patch embedding) | conv (strided 3x3 conv stem, Xiao et al. 2021, then 1x1 to dim)"""
    vit_patch: int = 16
    """vit patch size / total stem stride (token grid = image / vit_patch)"""
    vit_pos: str = "learned"
    """vit position embedding: learned (baseline) | sincos (fixed 2D sin-cos, MAE-style)"""
    aug_shift: int = 0
    """DrQ random shift: replicate-pad by this many px and random-crop back, applied to rollout rgb (stored augmented); 0 = off"""
    enc_lr_scale: float = 1.0
    """image-encoder lr = learning_rate * this (separate Adam param group when != 1)"""
    enc_critic_only: bool = False
    """DrQ-style: the actor sees detached features, so the encoder is trained by the value loss only"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def random_shift(rgb, pad):
    """DrQ random shift on uint8 NHWC images: replicate-pad by `pad` and crop back at a per-image random integer offset."""
    n, h, w, _ = rgb.shape
    x = nn.functional.pad(rgb.permute(0, 3, 1, 2).float(), (pad, pad, pad, pad), mode="replicate")
    dy = torch.randint(0, 2 * pad + 1, (n,), device=rgb.device)
    dx = torch.randint(0, 2 * pad + 1, (n,), device=rgb.device)
    rows = (dy[:, None] + torch.arange(h, device=rgb.device)[None])[:, None, :, None].expand(n, x.shape[1], h, x.shape[3])
    x = x.gather(2, rows)
    cols = (dx[:, None] + torch.arange(w, device=rgb.device)[None])[:, None, None, :].expand(n, x.shape[1], h, w)
    return x.gather(3, cols).to(rgb.dtype).permute(0, 2, 3, 1)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels=sample_obs["rgb"].shape[-1]
        image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

def sincos_2d(gh, gw, dim, extra=0):
    """MAE-style fixed 2D sin-cos position embedding, shape (1, extra + gh*gw, dim); `extra` leading zero rows (CLS)."""
    q = dim // 4
    omega = 1.0 / 10000 ** (torch.arange(q, dtype=torch.float32) / q)
    yy, xx = torch.meshgrid(torch.arange(gh, dtype=torch.float32), torch.arange(gw, dtype=torch.float32), indexing="ij")
    oy, ox = yy.flatten()[:, None] * omega[None], xx.flatten()[:, None] * omega[None]
    pe = torch.cat([oy.sin(), oy.cos(), ox.sin(), ox.cos()], 1)
    return torch.cat([torch.zeros(extra, dim), pe], 0)[None]

class ViTNet(nn.Module):
    """Pre-LN ViT image encoder + the NatureCNN state branch.

    Baseline (readout=cls, stem=patch, patch=16): 16x16 linear patch embedding, CLS readout -> LN -> Linear -> 256.
    feature_relu=True mirrors NatureCNN's fc+ReLU and can leave all 256 feature units dead (an image-blind policy).
    readout=mean drops the CLS token and reads LN(mean of patch tokens); readout=flatten projects each LN'd token to
    tok_dim and concatenates them (keeps the spatial layout linearly readable, as NatureCNN's flatten does).
    stem=conv replaces the patch embedding by log2(patch) stride-2 3x3 conv+GELU layers and a 1x1 conv to dim.
    """
    def __init__(self, sample_obs, patch=16, dim=128, depth=4, heads=4, mlp_ratio=4, feature_size=256, feature_relu=False,
                 readout="cls", stem="patch", tok_dim=16, pos="learned"):
        super().__init__()
        in_channels = sample_obs["rgb"].shape[-1]
        h, w = sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2]
        self.readout = readout
        n_tok = (h // patch) * (w // patch)
        if stem == "patch":
            self.stem = None
            self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)
        elif stem == "conv":
            n = int(np.log2(patch))
            assert 2 ** n == patch
            chans = [in_channels] + [min(dim, 24 * 2 ** i) for i in range(n)]
            layers = []
            for c_in, c_out in zip(chans[:-1], chans[1:]):
                layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.GELU()]
            self.stem = nn.Sequential(*layers)
            self.patch_embed = nn.Conv2d(chans[-1], dim, kernel_size=1)
        else:
            raise ValueError(stem)
        if readout == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.cls, std=0.02)
            n_tok += 1
        if pos == "learned":
            self.pos = nn.Parameter(torch.zeros(1, n_tok, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        elif pos == "sincos":
            # Fixed (not learned, not saved): a learned pos/CLS is an input-independent offset Adam can grow freely.
            self.register_buffer("pos", sincos_2d(h // patch, w // patch, dim, extra=n_tok - (h // patch) * (w // patch)),
                                 persistent=False)
        else:
            raise ValueError(pos)
        # Separate constructions, not nn.TransformerEncoder: its deepcopy would give every layer identical init.
        # dropout=0 because PPO's ratio assumes the policy is the same function at rollout and update time.
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(dim, heads, dim * mlp_ratio, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        fc_in = dim
        if readout == "flatten":
            self.tok_proj = nn.Linear(dim, tok_dim)
            fc_in = tok_dim * n_tok
        elif readout not in ("cls", "mean"):
            raise ValueError(readout)
        self.fc = nn.Sequential(nn.Linear(fc_in, feature_size), nn.ReLU()) if feature_relu else nn.Linear(fc_in, feature_size)
        self.out_features = feature_size
        self.state = None
        if "state" in sample_obs:
            self.state = nn.Linear(sample_obs["state"].shape[-1], 256)
            self.out_features += 256

    def image_features(self, rgb):
        x = rgb.float().permute(0, 3, 1, 2) / 255
        if self.stem is not None:
            x = self.stem(x)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        if self.readout == "cls":
            x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1)
        x = self.blocks(x + self.pos)
        if self.readout == "cls":
            x = self.norm(x)[:, 0]
        elif self.readout == "mean":
            x = self.norm(x.mean(1))
        else:
            x = self.tok_proj(self.norm(x)).flatten(1)
        return self.fc(x)

    def forward(self, observations) -> torch.Tensor:
        out = [self.image_features(observations["rgb"])]
        if self.state is not None:
            out.append(self.state(observations["state"]))
        return torch.cat(out, dim=1)

class Agent(nn.Module):
    def __init__(self, envs, sample_obs, feature_net=None):
        super().__init__()
        self.feature_net = feature_net if feature_net is not None else NatureCNN(sample_obs=sample_obs)
        latent_size = self.feature_net.out_features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5)
        self.actor_detach = False
    def get_features(self, x):
        return self.feature_net(x)
    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x.detach() if self.actor_detach else x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class LSTMAgent(nn.Module):
    """NatureCNN features (512) -> 1-layer LSTM(256) -> the baseline actor/critic heads; recurrent PPO as in CleanRL ppo_atari_lstm.

    Batched inputs are time-major flattened (T*B). `done` flags obs that are the FIRST obs of a new episode
    (ManiSkill auto-resets in the same step), so (h, c) is zeroed before that obs is processed.
    """
    hidden = 256
    reset_every_step = False

    def __init__(self, envs, sample_obs):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        self.lstm = nn.LSTM(self.feature_net.out_features, self.hidden)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        n_act = np.prod(envs.unwrapped.single_action_space.shape)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.hidden, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.hidden, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, n_act), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, n_act) * -0.5)

    def initial_state(self, n, device):
        return (torch.zeros(1, n, self.hidden, device=device), torch.zeros(1, n, self.hidden, device=device))

    def get_states(self, x, lstm_state, done):
        feats = self.feature_net(x)
        b = lstm_state[0].shape[1]
        feats = feats.reshape((-1, b, self.lstm.input_size))
        done = done.reshape((-1, b))
        if self.reset_every_step:
            done = torch.ones_like(done)
        out = []
        for f, d in zip(feats, done):
            keep = (1.0 - d).view(1, -1, 1)
            h, lstm_state = self.lstm(f.unsqueeze(0), (keep * lstm_state[0], keep * lstm_state[1]))
            out.append(h)
        return torch.flatten(torch.cat(out), 0, 1), lstm_state

    def get_value(self, x, lstm_state, done):
        return self.critic(self.get_states(x, lstm_state, done)[0])

    def get_action(self, x, lstm_state, done, deterministic=False):
        h, lstm_state = self.get_states(x, lstm_state, done)
        action_mean = self.actor_mean(h)
        if deterministic:
            return action_mean, lstm_state
        return Normal(action_mean, torch.exp(self.actor_logstd.expand_as(action_mean))).sample(), lstm_state

    def get_action_and_value(self, x, lstm_state, done, action=None):
        h, lstm_state = self.get_states(x, lstm_state, done)
        action_mean = self.actor_mean(h)
        probs = Normal(action_mean, torch.exp(self.actor_logstd.expand_as(action_mean)))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(h), lstm_state

def make_agent(arch, envs, sample_obs, **vit_kwargs):
    if arch == "cnn":
        return Agent(envs, sample_obs=sample_obs)
    if arch in ("vit", "vit_relu"):
        return Agent(envs, sample_obs=sample_obs, feature_net=ViTNet(sample_obs, feature_relu=arch == "vit_relu", **vit_kwargs))
    if arch == "lstm":
        return LSTMAgent(envs, sample_obs=sample_obs)
    raise ValueError(arch)

class Logger:
    """Appends (step, tag, value) rows to metrics.csv; avoids a tensorboard dependency."""
    def __init__(self, log_wandb=False, path=None) -> None:
        self.log_wandb = log_wandb
        new = not os.path.exists(path)
        self.f = open(path, "a")
        if new:
            self.f.write("step,tag,value\n")
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.f.write(f"{step},{tag},{float(scalar_value)}\n")
        self.f.flush()
    def close(self):
        self.f.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    run_dir = f"{args.out_root}/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(obs_mode="rgb", render_mode=args.render_mode, sim_backend="physx_cuda", variant=args.env_variant)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args.include_state)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"{run_dir}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"{run_dir}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["ppo", "walltime_efficient"]
            )
        with open(f"{run_dir}/args.txt", "w") as fh:
            fh.write("\n".join(f"{key}={value}" for key, value in vars(args).items()) + "\n")
        logger = Logger(log_wandb=args.track, path=f"{run_dir}/metrics.csv")
    else:
        print("Running evaluation")

    # ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    recurrent = args.arch == "lstm"
    vit_kwargs = dict(readout=args.vit_readout, stem=args.vit_stem, patch=args.vit_patch, pos=args.vit_pos) if args.arch.startswith("vit") else {}
    agent = make_agent(args.arch, envs, next_obs, **vit_kwargs).to(device)
    if recurrent:
        agent.reset_every_step = args.lstm_reset_every_step
    if args.enc_critic_only:
        assert not recurrent
        agent.actor_detach = True
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"arch={args.arch} params={n_params}")
    if not args.evaluate:
        open(f"{run_dir}/params.txt", "w").write(f"{args.arch} {n_params}\n")
    if args.enc_lr_scale != 1.0:
        enc = [p for n_, p in agent.named_parameters() if n_.startswith("feature_net.") and not n_.startswith("feature_net.state")
               and "extractors.state" not in n_]
        enc_ids = {id(p) for p in enc}
        param_groups = [dict(params=[p for p in agent.parameters() if id(p) not in enc_ids]),
                        dict(params=enc, lr=args.learning_rate * args.enc_lr_scale)]
    else:
        param_groups = agent.parameters()
    optimizer = optim.Adam(param_groups, lr=args.learning_rate, eps=args.adam_eps)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    cumulative_times = defaultdict(float)

    # Resume after preemption/crash: restores weights, optimizer, counters and RNGs, but not the
    # in-flight env states (envs start from a fresh reset), so a resumed run is not bit-identical.
    resume_path = f"{run_dir}/resume.pt"
    start_iteration = 1
    if os.path.exists(resume_path) and not args.evaluate:
        rs = torch.load(resume_path, weights_only=False)
        agent.load_state_dict(rs["agent"])
        optimizer.load_state_dict(rs["optimizer"])
        start_iteration, global_step = rs["iteration"] + 1, rs["global_step"]
        torch.set_rng_state(rs["torch_rng"]); torch.cuda.set_rng_state(rs["cuda_rng"])
        np.random.set_state(rs["np_rng"]); random.setstate(rs["py_rng"])
        start_time = time.time() - rs["elapsed"]
        print(f"resumed from {resume_path} at iteration {rs['iteration']} global_step={global_step}")

    # Envs start from a fresh reset (also after resume), so a zero hidden state is the correct episode start.
    if recurrent:
        lstm_state = agent.initial_state(args.num_envs, device)

    for iteration in range(start_iteration, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            if recurrent:
                eval_state = agent.initial_state(args.num_eval_envs, device)
                eval_done = torch.zeros(args.num_eval_envs, device=device)
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    if recurrent:
                        eval_action, eval_state = agent.get_action(eval_obs, eval_state, eval_done, deterministic=True)
                    else:
                        eval_action = agent.get_action(eval_obs, deterministic=True)
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_action)
                    if recurrent:
                        eval_done = torch.logical_or(eval_terminations, eval_truncations).to(torch.float32)
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"{run_dir}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
            if not args.evaluate and iteration > 1:
                torch.save(dict(agent=agent.state_dict(), optimizer=optimizer.state_dict(), iteration=iteration - 1,
                                global_step=global_step, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                                np_rng=np.random.get_state(), py_rng=random.getstate(), elapsed=time.time() - start_time),
                           resume_path + ".tmp")
                os.replace(resume_path + ".tmp", resume_path)
        # Annealing the rate if instructed to do so.
        if args.anneal_lr or args.lr_warmup_iters > 0:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations if args.anneal_lr else 1.0
            if args.lr_warmup_iters > 0:
                frac *= min(1.0, iteration / args.lr_warmup_iters)
            for g, b in zip(optimizer.param_groups, base_lrs):
                g["lr"] = b * frac
        rollout_time = time.perf_counter()
        if recurrent:
            initial_lstm_state = (lstm_state[0].clone(), lstm_state[1].clone())
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            if args.aug_shift > 0:
                next_obs = dict(next_obs, rgb=random_shift(next_obs["rgb"], args.aug_shift))
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if recurrent:
                    action, logprob, _, value, lstm_state = agent.get_action_and_value(next_obs, lstm_state, next_done)
                else:
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                if args.aug_shift > 0:
                    infos["final_observation"]["rgb"] = random_shift(infos["final_observation"]["rgb"], args.aug_shift)
                with torch.no_grad():
                    if recurrent:
                        # The terminal obs continues the finished episode: run it from the post-step state, no reset.
                        fv = agent.get_value(infos["final_observation"], (lstm_state[0][:, done_mask], lstm_state[1][:, done_mask]),
                                             torch.zeros(int(done_mask.sum()), device=device))
                    else:
                        fv = agent.get_value(infos["final_observation"])
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = fv.view(-1)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            boot_obs = dict(next_obs, rgb=random_shift(next_obs["rgb"], args.aug_shift)) if args.aug_shift > 0 else next_obs
            next_value = (agent.get_value(boot_obs, lstm_state, next_done) if recurrent else agent.get_value(boot_obs)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        # Recurrent: minibatch over envs, each keeping its whole num_steps sequence, replayed from the rollout-start state.
        by_env = recurrent or args.mb_by_env
        b_inds = np.arange(args.num_envs if by_env else args.batch_size)
        mb_size = args.num_envs // args.num_minibatches if by_env else args.minibatch_size
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        update_time = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), mb_size):
                end = start + mb_size
                if recurrent:
                    mb_envs = b_inds[start:end]
                    mb_inds = flatinds[:, mb_envs].ravel()
                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                        b_obs[mb_inds], (initial_lstm_state[0][:, mb_envs], initial_lstm_state[1][:, mb_envs]),
                        b_dones[mb_inds], b_actions[mb_inds])
                else:
                    mb_inds = flatinds[:, b_inds[start:end]].ravel() if by_env else b_inds[start:end]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                if args.log_diag:
                    enc = [p for n_, p in agent.named_parameters() if n_.startswith("feature_net.") and "state" not in n_
                           and p.grad is not None]
                    enc_gn = torch.norm(torch.stack([p.grad.norm() for p in enc])).item()
                total_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.log_diag:
            with torch.no_grad():
                sub = b_obs[np.arange(0, args.batch_size, max(1, args.batch_size // 1024))]
                fn = agent.feature_net
                img = (fn.extractors["rgb"](sub["rgb"].float().permute(0, 3, 1, 2) / 255) if hasattr(fn, "extractors")
                       else fn(sub)[:, :256])
                logger.add_scalar("diag/img_dead_frac", (img <= 0).all(0).float().mean().item(), global_step)
                logger.add_scalar("diag/img_feat_std", img.std(0).mean().item(), global_step)
                if not recurrent:
                    a0 = agent.get_action(sub, deterministic=True)
                    a1 = agent.get_action(dict(sub, rgb=torch.roll(sub["rgb"], 1, 0)), deterministic=True)
                    logger.add_scalar("diag/rgb_swap_da", (a1 - a0).abs().mean().item(), global_step)
                if isinstance(fn, ViTNet):
                    if fn.readout == "cls":
                        logger.add_scalar("diag/vit_cls_rms", fn.cls.pow(2).mean().sqrt().item(), global_step)
                    if isinstance(fn.pos, nn.Parameter):
                        logger.add_scalar("diag/vit_pos_rms", fn.pos.pow(2).mean().sqrt().item(), global_step)
            logger.add_scalar("diag/enc_grad_norm", enc_gn, global_step)
            logger.add_scalar("diag/grad_norm", float(total_gn), global_step)
        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
    if args.save_model and not args.evaluate:
        model_path = f"{run_dir}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        open(f"{run_dir}/DONE", "w").write(f"{global_step}\n")

    envs.close()
    if logger is not None: logger.close()
