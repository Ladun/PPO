import argparse
import os
import numpy as np
import torch

from agent import PPOAgent
from utils.envs import (
    create_mujoco_env
)
from utils.general import get_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", type=str, default=None,
                        help="path to pretrained model ")
    parser.add_argument("--load_postfix", type=str, required=True, 
                        help="pretrained model prefix(ex/ number of episode, 'best' or 'last') to play")
    parser.add_argument("--n_episode", type=int, default=30)
    parser.add_argument("--expert_dir", type=str, default='experts')
    parser.add_argument("--minimum_score", type=float, required=True)
    args = parser.parse_args()

    # make expert_dir
    os.makedirs(args.expert_dir, exist_ok=True)
    # TODO: if we need, copy config file to experts path

    return args

def get_trainer_and_env(args):

    # Get actor ====================    

    trainer = PPOAgent.load(experiment_path=args.experiment_path, 
                            postfix=args.load_postfix,
                            resume=False)

    # Get environment ==============
    env = create_mujoco_env(trainer.config.env.env_name, video_path=None)
    return trainer, env


def main():
    args = parse_args()

    trainer, env = get_trainer_and_env(args)

    # Loop for 'n' episode
    episode = 0
    while episode < args.n_episode:
        
        # state, action list for record
        collected_state  = []
        collected_action = []

        # env reset
        state, _ = env.reset()
        done = False
        step = 0
        score = 0

        # Loop for end of epsiode
        while not done or step < args.config.max_episode_len:
            # get action from actor
            with torch.no_grad():                
                if trainer.config.train.observation_normalizer:
                    state = trainer.obs_normalizer(state, update=False)
                state = torch.from_numpy(state).unsqueeze(0).float()
                action, _, _, _ = trainer.network(state)

            # collect the state and action
            collected_state.append(state)
            collected_action.append(action)

            # env step
            next_state, reward, terminated, truncated, _ = env.step(np.clip(action.cpu().numpy().squeeze(0), 
                                                                            env.action_space.low, 
                                                                            env.action_space.high))

            score += reward
            step += 1

            done = terminated + truncated

            state = next_state

            if done:
                break
        
        print(f"{episode} episode score: {score}")
        # save state and action    
        if score > args.minimum_score:
            print(f"{episode} epsiode is saved at {args.expert_dir}")
            collected_state = torch.vstack(collected_state)
            collected_action = torch.vstack(collected_action)
            
            torch.save({"state": collected_state, "action": collected_action},
                       os.path.join(args.expert_dir, f"sa_{episode}.pth"))
            episode += 1
        else:
            print(f"{episode} resampling")


if __name__ == "__main__":
    main()