
import os
import random
from datetime import datetime
import numpy as np

import omegaconf
from omegaconf import OmegaConf

import torch


def get_cur_time_code():
    return datetime.now().replace(microsecond=0).strftime('%Y%m%d%H%M%S')


def make_directory_and_get_path(dir_path, file_name=None, uniqueness=False):
    if not os.path.exists(dir_path):
        print("Make directory:", dir_path)
    os.makedirs(dir_path, exist_ok=True)

    if file_name:
        if uniqueness:
            n, e = os.path.splitext(file_name)
            cur_time = get_cur_time_code()
            file_name = f'{n}_{cur_time}{e}'
        
        path = os.path.join(dir_path, file_name)
    else: 
        path = dir_path

    print(f"Path is '{path}'")

    return path


def get_config(yaml_file=None, yaml_string=None, exp_name=None, overwrite_exp=False, **kwargs):
    assert yaml_file is not None or yaml_string is not None or len(kwargs.keys()) > 0
    
    if yaml_string is not None:
        conf = OmegaConf.create(yaml_string)
    elif yaml_file is None:
        conf = OmegaConf.create(kwargs)
    else:
        conf = OmegaConf.load(yaml_file)
    
    # --- Create experiment name ---
    if exp_name is None:    
        exp_name = f"exp{get_cur_time_code()}"
    # save old exp_name
    if "exp_name" in conf:       
        if overwrite_exp:
            conf.old_exp_name = conf.exp_name
            conf.exp_name = exp_name
    else:
        conf.exp_name = exp_name

    return conf


def get_device(device=None):
    # --- Define torch device from config 'device'
    device = torch.device(
        'cpu' 
        if not torch.cuda.is_available() or device is None
        else device
    )
    torch.cuda.empty_cache()
    return device


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def pretty_config(config, indent_level=0):
    tab = '\t' * indent_level

    for k, v in config.items():
        if isinstance(v, omegaconf.dictconfig.DictConfig):
            print(f"{tab}{k}:")
            pretty_config(v, indent_level + 1)
        else:
            print(f"{tab}{k}: {v}")