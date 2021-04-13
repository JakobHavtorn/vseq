import argparse
import os
import subprocess

import wandb
import yaml

from vseq.settings import ROOT_PATH


sweep_config_file = os.path.join(ROOT_PATH, 'experiments', 'sweeps', 'sweep_bowman.yaml')

parser = argparse.ArgumentParser()
parser.add_argument("--n_agents", default=1, type=int, help="number of agents")
parser.add_argument("--cuda_devices", default='', type=str, help="available cuda devices")
parser.add_argument("--sweep_config", default=sweep_config_file, type=str, help="sweep config .yaml file")

args = parser.parse_args()

args.cuda_devices = args.cuda_devices.split()
assert len(args.cuda_devices) == args.n_agents, "Must specify zero or as many CUDA devices as agents"
assert all(device.isnumeric() for device in args.cuda_devices), "All devices must be integer IDs"


with open(args.sweep_config) as file_buffer:
    sweep_config = yaml.safe_load(file_buffer)

del sweep_config['program']


import IPython; IPython.embed()

exit(0)

sweep_id = wandb.sweep(sweep_config)
