import os
import sys
from shutil import copyfile

import torch

def save_exp_file(run_dir):
    src_file = os.path.realpath(sys.argv[0])
    tgt_file = os.path.join(run_dir, 'experiment.py')
    copyfile(src_file, tgt_file)

def save_model(run_dir, state_dict, name):
    assert len(name) > 0
    basename = f'{name}.pt'
    save_path = os.path.join(run_dir, basename)
    torch.save(state_dict, save_path)