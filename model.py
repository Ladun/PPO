
import omegaconf

import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        # -------- Initialize variables --------

        self.is_cont    = config.env.is_continuous
        self.device     = device
        self.shared_layer = config.network.shared_layer

        if self.is_cont:
            # if action space is defined as continuous, make variance
            self.action_dim = config.env.action_dim
            self.action_var = torch.full((self.action_dim, ), config.network.action_std_init ** 2).to(self.device)

        if config.network.shared_layer:
            self.shared_net = nn.Sequential(
                nn.Linear(config.env.state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh()
            )
            self.actor = nn.Sequential(
                nn.Linear(64, config.env.action_dim),
                nn.Tanh()
            )
            self.critic = nn.Linear(64, 1)

        else:
            self.actor = nn.Sequential(
                nn.Linear(config.env.state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, config.env.action_dim),
                nn.Tanh()
            )
            self.critic = nn.Sequential(
                nn.Linear(config.env.state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )


    def set_action_std(self, std):
        # Change the action variance

        if self.is_cont:
            self.action_var = torch.full((self.action_dim, ), std ** 2).to(self.device)
        else:
            print("[Warning] Calling Actor::set_action_std() on discrete action space policy")


    def forward(self, state, action=None):
        if self.shared_layer:
            state = self.shared_net(state)

        if self.is_cont:
            # continuous space action 
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            # discrete space action
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        # Get (action, action's log probs, estimated Value)
        if action is None:
            action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action, action_logprob, dist.entropy(), self.critic(state)


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.is_cont    = config.env.is_continuous
        self.action_dim = config.env.action_dim

        hidden_dim = config.hidden_dim
        self.m = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        if not self.is_cont:
            action = torch.nn.functional.one_hot(action, self.action_dim).float()
        state_action = torch.cat([state, action], dim=1)

        r = self.m(state_action)

        return r
    
    def get_irl_reward(self, state, action):
        logit = self.forward(state, action)
        prob = torch.sigmoid(logit)

        reward = -torch.log(prob)

        return  reward