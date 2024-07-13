
import numpy as np
import torch

class PPOMemory:
    def __init__(self, gamma, tau, device):
        self.storage = {
            "state": [],
            "action": [],
            "reward": [],
            "done": [],
            "value": [],
            "logprob": []
        }

        self.gamma = gamma
        self.tau = tau
        self.device = device

    def store(self, **kwargs):

        for k, v in kwargs.items():
            if k not in self.storage:
                print("[Warning] wrong data insertion")
            else:
                self.storage[k].append(v)

    def compute_gae_and_get(self, v, d):
        """
        parameters:
            ===========  ==========================  ==================
            Symbol       Shape                       Type
            ===========  ==========================  ==================
            v            (num_envs,)                 torch.Tensor       // value in the next state
            d            (num_envs,)                 numpy.ndarray      // done in the next state

        reference from:
            https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py line 64

        desc:
            Information about the value in storage
            ===========  ==================================  ==================
            Symbol       Shape                               Type
            ===========  ==================================  ==================
            state        list of (num_envs, (obs_space))     numpy.ndarray
            reward       list of (num_envs,)                 numpy.ndarray
            done         list of (num_envs,)                 numpy.ndarray
            action       list of (num_envs,)                 torch.Tensor
            logprob      list of (num_envs,)                 torch.Tensor
            value        list of (num_envs,)                 torch.Tensor
            ===========  ==================================  ==================
        """
        storage = {k: torch.stack(v)
                      if isinstance(v[0], torch.Tensor)
                      else torch.from_numpy(np.stack(v)).to(self.device)
                   for k, v in self.storage.items()}
        steps, num_envs = storage['reward'].size()
        storage['value'] = torch.cat([storage['value'], v.unsqueeze(0)], dim=0)
        storage['done'] = torch.cat([storage['done'], torch.from_numpy(d).to(self.device).unsqueeze(0)], dim=0).float()

        gae_t = torch.zeros(num_envs).to(self.device)
        storage['advant'] = torch.zeros((steps, num_envs)).to(self.device)

        # Each episode is calculated separately by done.
        for t in reversed(range(steps)):
            # delta(t) = reward(t) + γ * value(t+1) - value(t)
            delta_t = storage['reward'][t] \
                      + self.gamma * storage['value'][t+1] * (1 - storage['done'][t + 1]) \
                      - storage['value'][t]
            # gae(t) = delta(t) + γ * τ * gae(t + 1)
            gae_t = delta_t + self.gamma * self.tau * gae_t * (1 - storage['done'][t + 1])
            storage['advant'][t] = gae_t

        # Remove value in the next state
        storage['value'] = storage['value'][:steps]
        storage['v_target'] = storage['advant'] + storage['value']
        # storage['v_target'] = rewards[:steps]
        
        # The first two values ​​refer to the trajectory length and number of envs.
        storage = {k: v.reshape(-1, *v.size()[2:]) for k, v in storage.items()}

        self.reset_storage()
        return storage


    def reset_storage(self):
        self.storage = {k: [] for k, v in self.storage.items()}

    def __len__(self):
        return len(self.storage['state']) * self.storage['state'][0].shape[0]
