
import argparse
import os
import logging

import envpool

from agent import PPOAgent
from utils.general import get_device, get_config
from utils.evaluation_env import create_mujoco_env


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
    parser.add_argument("--not_resume", action='store_true')
    parser.add_argument("--desc", type=str, default="",
                        help="Additional description of the executing code")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # Get config
    config = get_config(args.config)

    # Setting logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.StreamHandler()])
    logging.info(f"Description: {args.desc}")

    trainer = PPOAgent(config,
                       get_device())
    if args.load_postfix and args.experiment_path:
        trainer = PPOAgent.load(experiment_path=args.experiment_path, 
                                postfix=args.load_postfix,
                                resume=not args.not_resume)

    if args.train:
        envs = envpool.make(config.env.env_name, 
                            env_type="gymnasium", 
                            num_envs=config.env.num_envs)
        trainer.step(envs, args.exp_name)

    if args.eval:
        env = create_mujoco_env(config.env.env_name, video_path='videos')
        trainer.play(
            num_episodes=args.eval_n_episode,
            max_ep_len=2048
        )
        

if __name__ == "__main__":
    main()