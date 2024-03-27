
import random
from collections import defaultdict, deque

import numpy as np
import torch


class PPOMemory:
    def __init__(self, gamma, tau, device):
        self.temp_memory = {
            "state": [],
            "action": [],
            "reward": [],
            "done": [],
            "value": [],
            "logprob": []
        }

        self.max_memory_size = 300
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.trajectories = deque(maxlen=self.max_memory_size)

    def store(self, **kwargs):

        for k, v in kwargs.items():
            if k not in self.temp_memory:
                print("[Warning] wrong data insertion")
            else:
                self.temp_memory[k].append(v)

    def finish(self, n_s, v, d):
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
                   for k, v in self.temp_memory.items()}
        steps, num_envs = storage['reward'].size()
        storage['value'] = torch.cat([storage['value'], v.unsqueeze(0)], dim=0)
        storage['done'] = torch.cat([storage['done'], torch.from_numpy(d).to(self.device).unsqueeze(0)], dim=0).float()

        # TEMP
        storage['state'] = torch.cat([storage['state'], n_s.unsqueeze(0)], dim=0)
        for i in range(storage['done'].size()[1]):
            self.trajectories.append({
                k: v[:, i] for k, v in storage.items()
            })
        self.reset_temp_memory()
        return

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
        storage['value'] = storage['value']
        storage['v_target'] = storage['advant'] + storage['value'][:steps]

        for i in range(storage['done'].size()[1]):
            self.trajectories.append({
                k: v[:, i] for k, v in storage.items()
            })
        
        # # The first two values ​​refer to the trajectory length and number of envs.
        # storage = {k: v.reshape(-1, *v.size()[2:]) for k, v in storage.items()}

        self.reset_temp_memory()
        # return storage

    def calculate_advantage(self, storage, network):

        steps, num_envs = storage['reward'].size()

        _, _, _, values = network(storage['state'].reshape((steps + 1) * num_envs, -1).float())
        storage['values'] = values.reshape(steps + 1, num_envs)

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
        storage['state'] = storage['state'][:steps]
        storage['done'] = storage['done'][:steps]
        storage['v_target'] = storage['advant'] + storage['value']

        # # The first two values ​​refer to the trajectory length and number of envs.
        storage = {k: v.reshape(-1, *v.size()[2:]) for k, v in storage.items()}

        return storage

    def get_data(self, num_of_trajectories, network):
        idx = list(range(len(self.trajectories)))
        random.shuffle(idx)

        batch = defaultdict(list)
        for i in idx[:num_of_trajectories]:
            for k, v in self.trajectories[i].items():
                batch[k].append(v.unsqueeze(1))

        return self.calculate_advantage({k: torch.cat(v, dim=1) for k, v in batch.items()}, network)

    def reset_temp_memory(self):
        self.temp_memory = {k: [] for k, v in self.temp_memory.items()}

    def __len__(self):
        return len(self.temp_memory['state']) * self.temp_memory['state'][0].shape[0]
