"""NatureCNN actor-critic, identical in architecture to the ManiSkill GM runs (gm_robotics/ppo_rgb_gm.py)."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    def __init__(self, rgb_shape, state_dim, feature_size=256):
        super().__init__()
        h, w, c = rgb_shape
        cnn = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(), nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            n_flat = cnn(torch.zeros(1, c, h, w)).shape[1]
        self.rgb = nn.Sequential(cnn, nn.Linear(n_flat, feature_size), nn.ReLU())
        self.state = nn.Linear(state_dim, 256)
        self.out_features = feature_size + 256

    def forward(self, obs):
        x = obs["rgb"].float().permute(0, 3, 1, 2) / 255.0
        return torch.cat([self.rgb(x), self.state(obs["state"])], 1)


class Agent(nn.Module):
    def __init__(self, rgb_shape, state_dim, act_dim):
        super().__init__()
        self.feature_net = NatureCNN(rgb_shape, state_dim)
        n = self.feature_net.out_features
        self.critic = nn.Sequential(layer_init(nn.Linear(n, 512)), nn.ReLU(inplace=True), layer_init(nn.Linear(512, 1)))
        self.actor_mean = nn.Sequential(layer_init(nn.Linear(n, 512)), nn.ReLU(inplace=True),
                                        layer_init(nn.Linear(512, act_dim), std=0.01 * np.sqrt(2)))
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def get_value(self, obs):
        return self.critic(self.feature_net(obs))

    def get_action(self, obs, deterministic=False):
        mean = self.actor_mean(self.feature_net(obs))
        if deterministic:
            return mean
        return Normal(mean, self.actor_logstd.expand_as(mean).exp()).sample()

    def get_action_and_value(self, obs, action=None):
        x = self.feature_net(obs)
        mean = self.actor_mean(x)
        probs = Normal(mean, self.actor_logstd.expand_as(mean).exp())
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
