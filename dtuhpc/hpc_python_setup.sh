# !/bin/bash

# Stop on error
set -e

# unload modules
module unload python3/3.9.5
module unload cuda/11.1
# load modules
module load python3/3.9.5
module load cuda/11.1

# Use HOME directory as base
VENV_PATH=$HOME/venvs
REPO_PATH=$HOME/Documents

# Setup virtual env
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

echo "Setup script completed successfully"
