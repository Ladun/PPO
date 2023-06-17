
import argparse

from agent import PPOAgent
from utils.general import get_device, get_config

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--load_postfix", type=str, default=None,
                        help="pretrained model prefix(ex/ number of episode, 'best' or 'last') to play")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    trainer = PPOAgent(get_config(args.config,
                                  exp_name=args.exp_name),
                       get_device())
    if args.load_postfix:
        trainer.load(args.load_postfix)

    if args.train:
        trainer.step()

    if args.eval:
        trainer.play(
            num_episodes=10,
            max_ep_len=2048,
            use_rendering=True
        )
        

if __name__ == "__main__":
    main()