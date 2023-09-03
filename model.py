
import omegaconf

import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


def get_model(model_info, **kwargs):
    # Make the model based on the value written in config.

    def get_value(tmp):
        if isinstance(tmp, omegaconf.listconfig.ListConfig):
            return sum([get_value(_tmp) for _tmp in tmp])
        if isinstance(tmp, str):
            return kwargs[tmp]
        
        return tmp

    # Function body
    model = []  
    for _, info in enumerate(model_info):
        # Module name
        m = info[0] 
        # Convert str value to value defined in kwargs
        v = [get_value(t) for t in info[1]]
        # index for where to use the value
        # idx = info[1]

        model.append(eval(m)(*v))

    return nn.Sequential(*model)



class Actor(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        # -------- Initialize variables --------

        self.is_cont    = config.env.is_continuous
        self.device     = device

        # -------- Define models --------

        if self.is_cont:
            # if action space is defined as continuous, make variance
            self.action_dim = config.env.action_dim
            self.action_var = torch.full((self.action_dim, ), config.actor.action_std_init ** 2).to(self.device)

        self.m = get_model(
            config.actor.arch,
            state_dim=config.env.state_dim,
            action_dim=config.env.action_dim
        )

        # -------- warning --------

        if self.is_cont:
            if isinstance(self.m[-1], nn.Softmax):
                print("[Warning] action is continuous space but model output layer is softmax")
    

    def set_action_std(self, std):
        # Change the action variance

        if self.is_cont:
            self.action_var = torch.full((self.action_dim, ), std ** 2).to(self.device)
        else:
            print("[Warning] Calling Actor::set_action_std() on discrete action space policy")


    def forward(self, state, action=None):
        if self.is_cont:
            # continuous space action 
            action_mean = self.m(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            # discrete space action
            action_probs = self.m(state)
            dist = Categorical(action_probs)

        # Get (action, action's log probs, estimated Value)
        if action is None:
            action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action, action_logprob, dist.entropy()
    

class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()

        # -------- Define models --------

        self.m = get_model(
            config.critic.arch,
            state_dim=config.env.state_dim
        )

    def forward(self, inp):
        return self.m(inp)


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.is_cont    = config.env.is_continuous
        self.action_dim = config.env.action_dim

        self.m = get_model(
            config.gail.arch,
            state_dim=config.env.state_dim,
            action_dim=config.env.action_dim
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