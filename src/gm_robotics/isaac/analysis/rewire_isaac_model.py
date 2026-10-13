"""Rewired policy head on a frozen Isaac GM-nav trunk (importable without the Isaac app)."""
import numpy as np
import torch
import torch.nn as nn

from gm_robotics.isaac.agent import layer_init


class Rewired(nn.Module):
    """Frozen trunk of `base` up to h_pre plus its proprio branch; fresh downstream according to the arm."""

    def __init__(self, base, arm, init_seed):
        super().__init__()
        self.base, self.arm = base, arm
        torch.manual_seed(init_seed)
        if arm in ("pre", "remix"):
            self.fc = nn.Linear(4096, 256)
            if arm == "remix":
                self.fc.requires_grad_(False)
        self.head = nn.Sequential(layer_init(nn.Linear(512, 512)), nn.ReLU(),
                                  layer_init(nn.Linear(512, 3), std=0.01 * np.sqrt(2)))

    def forward(self, h_pre, h_fc_native, h_state):
        h = h_fc_native if self.arm in ("fc_head", "untrained") else torch.relu(self.fc(h_pre))
        return self.head(torch.cat([h, h_state], 1))

    def trainable(self):
        return [q for n_, q in self.named_parameters() if q.requires_grad and not n_.startswith("base.")]
