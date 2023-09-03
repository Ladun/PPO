import os

import torch
import torch.utils.data as data

def get_expert_data(config):
    files = os.listdir(config.gail.expert_dir)

    state_data = []
    action_data = []
    for file in files:
        temp = torch.load(os.path.join(config.gail.expert_dir, file))
        
        state_data.append(temp['state'])
        action_data.append(temp['action'])

    state_data = torch.cat(state_data, dim=0)
    action_data = torch.cat(action_data, dim=0)

    assert state_data.size(1) == config.env.state_dim and \
            action_data.size(1) == config.env.action_dim, "Wrong demonstration data"

    return state_data, action_data


if __name__ == "__main__":
    from general import get_config

    config = get_config("../configs/Ant-v4-gail.yaml")
    dataset = get_expert_data(config)

    print(dataset[0].shape)
    print(dataset[1].shape)
    
