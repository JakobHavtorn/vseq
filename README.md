# vseq


## Install 

```bash
# Reinstall
conda deactivate
conda env remove -n vseq -y
conda create -y -n vseq python==3.9
conda activate vseq
pip install -f https://download.pytorch.org/whl/torch_stable.html --upgrade --editable . 
nbstripout --install
```

## wandb sweeps

See `experiments/sweep_bowman.yaml`

> `wandb sweep experiments/sweeps/sweep_bowman.yaml`

> `wandb agent <sweep-id>`
