import inspect
import logging
import os

from typing import Any, Optional, List, Iterable, Union

import torch
import torch.nn as nn
import torchinfo

import vseq.models


LOGGER = logging.getLogger(name=__file__)

MODEL_CLASS_NAME_STR = "model_class_name.pt"
MODEL_INIT_KWRGS_STR = "model_kwargs.pt"
MODEL_STATE_DICT_STR = "model_state_dict.pt"


def load_model(path, model_class_name: str = None, device: torch.device = torch.device("cpu")):
    if not os.path.exists(path):
        raise RuntimeError(f"Tried to load model checkpoint but the path does not exist: {path}")

    if model_class_name is None:
        if os.path.exists(os.path.join(path, MODEL_CLASS_NAME_STR)):
            model_class_name = torch.load(os.path.join(path, MODEL_CLASS_NAME_STR))
            LOGGER.debug(f"Loading '{model_class_name}' from 'vseq.models'")
        else:
            raise RuntimeError(f"Name of class of model to load was not given and not saved in checkpoint: {path}")

    model_class = getattr(vseq.models, model_class_name)
    model = model_class.load(path, device=device)
    return model


class BaseModel(nn.Module):
    """Base class for end-use type Modules (e.g. models)"""

    def __init__(self):
        super().__init__()
        self._init_arguments = None
        self._kwarg_names = self._capture_argument_names()

    def _capture_argument_names(self):
        signature = inspect.signature(self.__class__.__init__)
        return [p for p in signature.parameters if p != "self"]

    def init_arguments(self):
        """Return a dictionary of the kwargs used to instantiate this module"""
        if self._init_arguments is None:
            self._init_arguments = self._get_init_arguments()
        return self._init_arguments

    def _get_init_arguments(self):
        """Retrieve the values of keyword arguments used to instantiate this Module (assumes they are all properties)"""
        missing_names = [name for name in self._kwarg_names if name not in vars(self)]

        if len(missing_names) > 0:
            msg = (
                f"Models need to define the `kwargs` to `__init__` as attributes but {str(self.__class__)} is "
                f"missing the following attributes: {missing_names}."
            )
            raise RuntimeError(msg)

        init_arguments = {attr: getattr(self, attr) for attr in self._kwarg_names}
        return init_arguments

    @property
    def device(self):
        """Heuristically return the device which this model is on"""
        return next(self.parameters()).device

    def get_checkpoint(self):
        """Return a checkpoint dict of the module class name, init_arguments and state_dict"""
        checkpoint = dict(
            model_class_name=self.__class__.__name__,
            model_init_kwargs=self.init_arguments(),
            model_state_dict=self.state_dict()
        )
        return checkpoint

    def save(self, path):
        """Save the module class name, init_arguments and state_dict to different files in the directory given by path"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.__class__.__name__, os.path.join(path, MODEL_CLASS_NAME_STR))
        torch.save(self.init_arguments(), os.path.join(path, MODEL_INIT_KWRGS_STR))
        torch.save(self.state_dict(), os.path.join(path, MODEL_STATE_DICT_STR))

    @classmethod
    def load(cls, path, device: str = "cpu"):
        """Return an instance of the concrete module instantiated using saved init_arguments and with state_dict loaded"""
        model_kwargs = torch.load(os.path.join(path, MODEL_INIT_KWRGS_STR))
        kwargs = model_kwargs.pop("kwargs", {})  # TODO If we handle *args and **kwargs then these could be removed
        args = model_kwargs.pop("args", [])

        model = cls(*args, **kwargs, **model_kwargs)
        model.to(device)

        state_dict = torch.load(os.path.join(path, MODEL_STATE_DICT_STR), map_location=device)
        model.load_state_dict(state_dict)
        return model

    def get_parameter_vector(self):
        parameters = [p.flatten() for p in self.parameters()]
        parameters = torch.cat(parameters)
        return parameters

    def get_gradient_vector(self):
        gradient = [p.grad.flatten() for p in self.parameters()]
        gradient = torch.cat(gradient)
        return gradient

    def get_gradient_norm(self):
        total_norm = 0
        for p in self.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm

    def extra_repr(self):
        """All init_arguments as extra representation in a string formatted as a dictionary"""
        if not self.init_arguments():
            return ""
        s = ",\n  ".join(f"{k}={v}" for k, v in self.init_arguments().items() if not isinstance(v, nn.Module))
        s = "kwargs={\n" + "  " + s + "\n}"
        return s

    def summary(
        self: nn.Module,
        input_size: Optional[Union[Iterable[int]]] = None,
        input_data: Optional[Iterable[torch.Tensor]] = None,
        batch_dim: Optional[int] = None,
        col_names: Optional[Iterable[str]] = ["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        col_width: int = 25,
        depth: int = 6,
        device: Optional[torch.device] = None,
        dtypes: Optional[List[torch.dtype]] = None,
        row_settings: Optional[Iterable[str]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ):
        """Return a summary of the model"""
        return torchinfo.summary(
            model=self,
            input_size=input_size,
            input_data=input_data,
            batch_dim=batch_dim,
            col_names=col_names,
            col_width=col_width,
            depth=depth,
            device=device,
            dtypes=dtypes,
            row_settings=row_settings,
            verbose=verbose,
            **kwargs,
        )
