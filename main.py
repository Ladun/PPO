
import argparse
import logging

import envpool
import custom_env
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from agent import PPOAgent
from utils.general import get_config
from utils.envs import (
    create_mujoco_env
)


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--eval_n_episode", type=int, default=10)
    parser.add_argument("--load_postfix", type=str, default=None,
                        help="pretrained model prefix(ex/ number of episode, 'best' or 'last') from same experiments")
    parser.add_argument("--experiment_path", type=str, default=None,
                        help="path to pretrained model ")
    parser.add_argument("--video_path", type=str, default='videos',
                        help="path to saving playing video ")
    parser.add_argument("--not_resume", action='store_true')
    parser.add_argument("--desc", type=str, default="",
                        help="Additional description of the executing code")
    args = parser.parse_args()

    return args

def make_env(env_name):
    def _init():
        env = gym.make(env_name, env_config={"render_mode":"rgb_array"})
        return env
    return _init


def main():
    args = parse_args()

    # Setting logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.StreamHandler()])
    logging.info(f"Description: {args.desc}")

    if args.load_postfix and args.experiment_path:
        trainer = PPOAgent.load(experiment_path=args.experiment_path, 
                                postfix=args.load_postfix,
                                resume=not args.not_resume)
    else:
        # Get config
        config = get_config(args.config)
        trainer = PPOAgent(config)        

    if args.train:
        if trainer.config.env.env_name in envpool.list_all_envs():
            envs = envpool.make(trainer.config.env.env_name, 
                                env_type="gymnasium", 
                                num_envs=trainer.config.env.num_envs)
        else:
            envs = AsyncVectorEnv([make_env(config.env.env_name) for _ in range(config.env.num_envs)])
            
        trainer.step(envs, args.exp_name)

    if args.eval:
        
        if trainer.config.env.env_name in custom_env.env_list:
        
            if args.video_path:
                env = gym.make(trainer.config.env.env_name, env_config={"render_mode": 'rgb_array'})
                env = gym.wrappers.RecordVideo(env, args.video_path)
            else:
                env = gym.make(trainer.config.env.env_name, env_config={"render_mode": 'human'})
        else:
            env = create_mujoco_env(trainer.config.env.env_name, video_path=args.video_path)
            
        trainer.play(
            env=env,
            num_episodes=args.eval_n_episode,
            max_ep_len=2048
        )
        

if __name__ == "__main__":
    main()