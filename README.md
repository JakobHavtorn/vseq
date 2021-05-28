# vseq

## Install 


Install primary environment relying on pre-innstalled system CUDA.

```bash
conda deactivate
conda env remove -n vseq -y
conda create -y -n vseq python==3.9
conda activate vseq
pip install -f https://download.pytorch.org/whl/torch_stable.html --upgrade --editable . 
```

Install primary environment using Conda to get CUDA.

```
conda deactivate
conda env remove -n vseq -y
conda create -y -n vseq python==3.8
conda activate vseq
conda install -y pytorch torchvision torchaudio torchtext cudatoolkit=11.1 -c pytorch -c nvidia
pip install -f https://download.pytorch.org/whl/torch_stable.html --upgrade --editable . 
```

Install extra requirements

```bash
pip install -r requirements-extra.txt
nbstripout --install
```


## Test

To run testsm, execute

```bash
pytest -sv --cov --cov-report=term tests
```


## Minimal experiment example
```python
```


## Data structure

```
root_dir
    data/
        librispeech/
        penn_treebank/
    source/
        librispeech/
        penn_treebank/
```


## Type-hints

This repo uses [Python type-hints](https://docs.python.org/3/library/typing.html).

It also extends this with the [`torchtyping`](https://github.com/patrick-kidger/torchtyping) package which allows defining the shape, data type, layout and names of axis of `torch.Tensors` by using a new `TensorType[...]` type.


## wandb

We track experiments using wandb.

### Enable/Disable

- `wandb online`, `WANDB_MODE=online` or `wandb.init(mode="online")` - runs in online mode, the default
- `wandb offline`, `WANDB_MODE=offline` or `wandb.init(mode="offline")` - runs in offline mode, writes all data to disk for later syncing to a server
- `wandb disabled`, `WANDB_MODE=disabled` or `wandb.init(mode="disabled")` - makes all calls to wandb api's noop's, while maintaining core functionality such as wandb.config and wandb.summary in case you have logic that reads from these dicts.
- `wandb enabled`, `WANDB_MODE=` or `wandb.init(mode="enabled")`- sets the mode to back online

### Clean

In the `wandb` directory `/some/path/wandb` run:
- `wandb sync --clean` to clean out all previously synced runs.
- `wandb sync --clean-old-hours INT` to clean out all runs older than `INT` hours.

### wandb sweeps

See `experiments/sweep_test.yaml`

> `wandb sweep experiments/sweeps/sweep_test.yaml`

> `wandb agent <sweep-id>`

## Implementation suggestions
- [ ] Method or class to define and build `Dataloader`s with proper defaults e.g. `pin_memory == True`.


## Resources

Examples of how to write fairly fast custom recurrent cells with `TorchScript` can be found here: 
https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
