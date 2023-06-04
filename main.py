
import argparse

from agent import PPOAgent
from utils import get_device, get_config

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--load_path", type=str, default=None,
                        help="pretrained model prefix(ex/ number of episode, 'best' or 'last') to play")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    trainer = PPOAgent(get_config(args.config), get_device())

    if args.train:
        trainer.step()

    if args.load_path:
        trainer.play(
            args.load_path,
            num_episodes=10,
            max_ep_len=300,
            use_rendering=True
        )
        

if __name__ == "__main__":
    main()