#!/bin/bash

# exit if error
set -e

# get command
CMD_STRING=$@

# run setup script
bash hpc_python_setup.sh

# activate envrionment
source $VENV_PATH/vseq/bin/activate

# Set $HOME if running as a bsub script
if [ -z "$BSUB_O_WORKDIR" ]; then
    export HOME=$BSUB_O_WORKDIR
fi
# Set $HOME if running as a qsub script
if [ -z "$PBS_O_WORKDIR" ]; then
    export HOME=$PBS_O_WORKDIR
fi
cd $HOME

# logging
pwd
ll -ah

# execute script
WANDB_NOTES=""
WANDB_MODE="disabled"
eval $CMD_STRING
