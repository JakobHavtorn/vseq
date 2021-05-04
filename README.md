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

It also extends this with the [`torchtyping`](https://github.com/patrick-kidger/torchtyping) package which allows defining the shape, data type, layout and names of axis of `torch.Tensors`

Turn this:

```python
def batch_outer_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x has shape (batch, x_channels)
    # y has shape (batch, y_channels)
    # return has shape (batch, x_channels, y_channels)
    return x.unsqueeze(-1) * y.unsqueeze(-2)
```

into this:

```python
from torchtyping import TensorType


def batch_outer_product(
    x:   TensorType["batch", "x_channels"],
    y:   TensorType["batch", "y_channels"]
) -> TensorType["batch", "x_channels", "y_channels"]:

    return x.unsqueeze(-1) * y.unsqueeze(-2)
```



## WANDB

### wandb sweeps

See `experiments/sweep_test.yaml`

> `wandb sweep experiments/sweeps/sweep_test.yaml`

> `wandb agent <sweep-id>`

## Enable/Disable

- `wandb online`, `WANDB_MODE=online` or `wandb.init(mode="online")` - runs in online mode, the default
- `wandb offline`, `WANDB_MODE=offline` or `wandb.init(mode="offline")` - runs in offline mode, writes all data to disk for later syncing to a server
- `wandb disabled`, `WANDB_MODE=disabled` or `wandb.init(mode="disabled")` - makes all calls to wandb api's noop's, while maintaining core functionality such as wandb.config and wandb.summary in case you have logic that reads from these dicts.
- `wandb enabled`, `WANDB_MODE=` or `wandb.init(mode="enabled")`- sets the mode to back online


## Implementation suggestions
- [ ] Method or class to define and build `Dataloader`s with proper defaults e.g. `pin_memory == True`.
