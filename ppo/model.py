import numpy as np
import torch
import os
import math
import torch.nn as nn
import torch.nn.functional as F
import gym

from ppo.distributions import Bernoulli, Categorical, DiagGaussian
from ppo.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MetaPolicy(nn.Module):
    def __init__(self, input_space, action_space):
        super(MetaPolicy, self).__init__()

        obs_shape = input_space.shape
        act_shape = action_space.shape

        #######################################################################
        #                              POLICY                                 #
        #######################################################################

        self.base = PolicyMLP(obs_shape[0])

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.policy = nn.Sequential(self.base, self.dist)

        #######################################################################
        #                             META NET                                #
        #######################################################################

        self.meta_net = MetaMLP(obs_shape[0], act_shape[0])

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return 10

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):

        value, actor_features, rnn_hxs = self.base(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs, dist_entropy

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):

        value, actor_features, rnn_hxs = self.base(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, dist

    def actions_prob(self, inputs, rnn_hxs, masks, action):

        value, actor_features, rnn_hxs = self.base(inputs)
        dist = self.dist(actor_features)
        action_probs = dist.probs
        return action_probs

    def predict_intrinsic(self, inputs, actions):

        int_rews, int_vals = self.meta_net(inputs, actions)

        return int_rews, int_vals


class PolicyMLP(nn.Module):
    def __init__(self, num_inputs, hidden_size=64):
        super(PolicyMLP, self).__init__()

        self.hidden_size = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    @property
    def output_size(self):
        return self.hidden_size

    def forward(self, inputs, *args):

        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        return self.critic_linear(hidden_critic), hidden_actor


class MetaMLP(nn.Module):
    def __init__(self, num_obs_inputs, num_act_inputs, hidden_size=64):
        super(MetaMLP, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.meta_reward = nn.Sequential(
            init_(nn.Linear(num_obs_inputs + num_act_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)))

        self.meta_critic = nn.Sequential(
            init_(nn.Linear(num_obs_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)))

        self.train()

    def forward(self, inputs, actions, *args):

        rews = self.meta_reward(torch.cat(inputs, actions))
        vals = self.meta_critic(inputs)

        return rews, vals
