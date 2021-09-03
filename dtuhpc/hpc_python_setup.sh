# !/bin/bash

# Script that installs a python virtual environment

# stop on error
set -e

# unload modules
module unload python3/3.8.2
module unload cuda/11.1
# load modules
module load python3/3.8.2
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
    # python3 -m venv vseq
    virtualenv vseq
    source $VENV_PATH/vseq/bin/activate

    which python3
    which python
    which pip3
    which pip

    cd $REPO_PATH/vseq
    pip install -f https://download.pytorch.org/whl/torch_stable.html -r requirements.txt
    echo "Finished installing vseq"
else
    echo "Using existing virtual environment vseq"
    source $VENV_PATH/vseq/bin/activate
fi


if [ ! -f VSEQ.env ]
then
    # create data root directory
    echo "Creating VSEQ.env file and setting VSEQ_DATA_ROOT_DIRECTORY"
    touch VSEQ.env
    echo "VSEQ_DATA_ROOT_DIRECTORY=/data/research" > VSEQ.env
fi


echo "Setup script completed successfully"