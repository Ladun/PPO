
import logging

import torch
import gymnasium as gym

logger = logging.getLogger(__name__)

# currently we support only gym env
class TrainEnvironment:
    def __init__(self, env_name, is_continuous, seed):

        self.is_init = False
        self.env_name = env_name
        self.is_continuous = is_continuous
        self.seed = seed

        self.env = None
        self.state_dim = None
        self.action_dim = None

    def init(self, render_mode=None):
        logger.info("(Init) training environment name : " + self.env_name)

        self.env = gym.make(self.env_name, render_mode=render_mode)
        # state space dimension
        self.state_dim = self.env.observation_space.shape[0]

        # action space dimension
        if self.is_continuous:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n

        self.is_init = True

    def reset(self):
        # return: (state, info)
        #   state shape: (observation_dim, )
        return self.env.reset(seed=self.seed)

    def _preprocess_action(self, action):
        if isinstance(action, torch.Tensor):
            if self.is_continuous:
                return action.cpu().detach().numpy().flatten()
            else:
                return action.item()
        return action

    def step(self, action):
        action = self._preprocess_action(action)
        return self.env.step(action)

    def close(self):
        self.env.close()
        self.is_init = False
