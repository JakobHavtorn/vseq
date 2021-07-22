#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpujdha
### -- set the job Name --
#BSUB -J CW-VAE
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 4 gpu in exclusive process mode --
#BSUB -gpu "num=4:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 672:00
# request 10GB of system-memory
#BSUB -R "span[hosts=1] rusage[mem=10GB]"
### -- set the email address --
# if you want to receive e-mail notifications on a non-default address
#BSUB -u jdh@corti.ai
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo dtuhpc/logs/%J.out
#BSUB -eo dtuhpc/logs/%J.out
# -- end of LSF options --

# exit if error
set -e

echo "==============================================================="
echo "Submitting job with command:"
echo "$@"
echo "==============================================================="

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

echo "==============================================================="
echo "Executing command ..."
# execute script
export WANDB_NOTES=""
# export WANDB_MODE="disabled"
export VSEQ_DATA_ROOT_DIRECTORY=/zhome/c2/b/86488/Documents/vseq/data
eval $CMD_STRING
