# !/bin/bash

# stop on error
set -e

# unload modules
module unload python3/3.9.5
module unload cuda/11.1

# setup virtual env
if [ ! -d $VENV_PATH/vseq ]
then
    echo "Found no virtual environment at " $VENV_PATH/vseq
    echo "Installing vseq..."

    # which python3
    # which pip3

    mkdir -p $VENV_PATH/vseq
    cd $VENV_PATH
    python3 -m venv vseq  # --copies

    # which python3
    # which pip3

    # load modules
    module load python3/3.9.5
    module load cuda/11.1

    # which python3
    # which pip3

    source $VENV_PATH/vseq/bin/activate

    # which python3
    # which pip3

    cd $REPO_PATH/vseq
    python3 -m pip install -f https://download.pytorch.org/whl/torch_stable.html --upgrade --editable .
    #python3 -m pip install -f https://download.pytorch.org/whl/torch_stable.html --upgrade torch==1.8.1+cu111 torchvision torchaudio torchtext
    echo "Finished installing vseq"
else
    echo "Using existing virtual environment vseq"
    source $VENV_PATH/vseq/bin/activate

    # load modules
    module load python3/3.9.5
    module load cuda/11.1
fi

# # create data root directory
# touch VSEQ.env
# echo ""

# # download data
# python3 scripts/data/prepare_timit.py

echo "Setup script completed successfully"
