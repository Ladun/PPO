
import os
import gymnasium
from gymnasium.envs.registration import registry, make, spec

def register(id, *args, **kvargs):
    if id in registry:
        return
    else:
        return gymnasium.envs.registration.register(id, *args, **kvargs)

register(id='AntBulletEnv-v0',
         entry_point='custom_env.env.ant_env:AntEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)
register(id='HumanoidBulletEnv-v0',
         entry_point='custom_env.env.humanoid_env:HumanoidEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

env_list = ["AntBulletEnv-v0", "HumanoidBulletEnv-v0"]