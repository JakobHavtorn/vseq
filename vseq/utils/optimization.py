from typing import Union, List, Dict

import torch


def get_learning_rates_list(optimizer: torch.optim.Optimizer) -> Union[List[float], float]:
    """Return learning rates of an optimizer as a list of floats per parameter group or a single float if one group"""
    if len(optimizer.param_groups) > 1:
        return [float(param_group["lr"]) for param_group in optimizer.param_groups]
    return next(optimizer.param_groups)["lr"]


def get_learning_rates_dict(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """Return learning rates of an optimizer as a dict of floats with keys given by `lr_{i}` for parameter group `i`"""
    if len(optimizer.param_groups) > 1:
        return {f"lr_{i}": float(param_group["lr"]) for i, param_group in enumerate(optimizer.param_groups)}
    return {"lr": optimizer.param_groups[0]["lr"]}
