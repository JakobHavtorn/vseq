#!/bin/bash

# exit if error
set -e

# get command
CMD_STRING=$@

# use HOME directory as base
VENV_PATH=$HOME/venvs
REPO_PATH=$HOME/Documents

# run setup script
bash dtuhpc/hpc_python_setup.sh

# unload modules
module unload python3/3.8.2
module unload cuda/11.1
# load modules
module load python3/3.8.2
module load cuda/11.1

# activate venv
source $VENV_PATH/vseq/bin/activate

# debugging
pwd
which python3
which python
which pip3
which pip

# execute script
export WANDB_NOTES=""
# export WANDB_MODE="disabled"
export VSEQ_DATA_ROOT_DIRECTORY=/zhome/c2/b/86488/Documents/vseq/data
eval $CMD_STRING
