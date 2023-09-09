
import argparse
import os
import logging

from agent import PPOAgent
from utils.general import get_device, get_config, get_experiments_base_path


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--eval_n_episode", type=int, default=10)
    parser.add_argument("--load_postfix", type=str, default=None,
                        help="pretrained model prefix(ex/ number of episode, 'best' or 'last') from same experiments")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="path to pretrained model ")
    parser.add_argument("--desc", type=str, default="",
                        help="Additional description of the executing code")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # Get config
    config = get_config(args.config, exp_name=args.exp_name)
    os.makedirs(get_experiments_base_path(config), exist_ok=True)

    # Setting logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler(os.path.join(get_experiments_base_path(config), f"running_{'eval' if args.eval else 'train'}_log.log")),
                            logging.StreamHandler()
                        ])
    logging.info(f"Description: {args.desc}")

    trainer = PPOAgent(config,
                       get_device())
    if args.load_postfix or args.checkpoint_path:
        trainer.load(postfix=args.load_postfix,
                     checkpoint_path=args.checkpoint_path)

    if args.train:
        trainer.step()

    if args.eval:
        trainer.play(
            num_episodes=args.eval_n_episode,
            max_ep_len=2048,
            use_rendering=True
        )
        

if __name__ == "__main__":
    main()