# !/bin/bash

# stop on error
set -e

# unload modules
module unload python3/3.9.5
module unload cuda/11.1
# load modules
module load python3/3.9.5
module load cuda/11.1

# use HOME directory as base
VENV_PATH=$HOME/venvs
REPO_PATH=$HOME/Documents

# setup virtual env
if [ ! -d $VENV_PATH/vseq ]
then
    echo "Found no virtual environment at " $VENV_PATH/vseq
    echo "Installing vseq..."
    mkdir -p $VENV_PATH/vseq
    cd $VENV_PATH
    python3 -m venv vseq --copies
    source $VENV_PATH/vseq/bin/activate

    cd $REPO_PATH/vseq
    python3 -m pip install --upgrade -f https://download.pytorch.org/whl/torch_stable.html --upgrade --editable .
    python3 -m pip install --upgrade torch==1.8.1+cu111 torchvision torchaudio torchtext -f https://download.pytorch.org/whl/torch_stable.html
    echo "Finished installing vseq"
else
    echo "Using existing virtual environment vseq"
    source $VENV_PATH/vseq/bin/activate
fi

# # create data root directory
# touch VSEQ.env
# echo ""

# # download data
# python3 scripts/data/prepare_timit.py

echo "Setup script completed successfully"
