import argparse
import os

import torch

from agent import PPOAgent
from env import TrainEnvironment
from utils.general import get_device, get_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--load_postfix", type=str, required=True, 
                        help="pretrained model prefix(ex/ number of episode, 'best' or 'last') to play")
    
    parser.add_argument("--n_episode", type=int, default=10)
    parser.add_argument("--expert_dir", type=str, default='experts')
    parser.add_argument("--minimum_score", type=float, required=True)
    args = parser.parse_args()

    args.config = get_config(args.config)

    # make expert_dir
    args.expert_dir = os.path.join(args.expert_dir, args.config.env.env_name)
    os.makedirs(args.expert_dir, exist_ok=True)
    # TODO: if we need, copy config file to experts path

    return args

def get_actor_and_env(args):

    # Get actor ====================    

    trainer = PPOAgent(args.config,
                       get_device())
    trainer.load(args.load_postfix)

    # Get environment ==============

    env = TrainEnvironment(
        env_name=args.config.env.env_name,
        is_continuous=args.config.env.is_continuous,
        seed=args.config.seed
    )
    env.init()

    return trainer.actor, env


def main():
    args = parse_args()

    actor, env = get_actor_and_env(args)

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
                state = torch.tensor(state).unsqueeze(0).float()
                action, _, _ = actor(state)

            # collect the state and action
            collected_state.append(state)
            collected_action.append(action)

            # env step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            score += reward

            state = next_state
            step += 1

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