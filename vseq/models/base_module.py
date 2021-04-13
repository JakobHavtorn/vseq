import inspect
import logging
import os

import torch
import torch.nn as nn

import vseq.models

from vseq.utils.summary import summary


LOGGER = logging.getLogger(name=__file__)

MODEL_CLASS_NAME_STR = 'model_class_name.pt'
MODEL_INIT_KWRGS_STR = 'model_kwargs.pt'
MODEL_STATE_DICT_STR = 'model_state_dict.pt'


def load_model(path, model_class_name: str = None, device: str = 'cpu'):
    if model_class_name is None:
        if os.path.exists(os.path.join(path, MODEL_CLASS_NAME_STR)):
            model_class_name = torch.load(os.path.join(path, MODEL_CLASS_NAME_STR))
            LOGGER.debug(f"Loading '{model_class_name}' from 'vseq.models'")
        else:
            raise RuntimeError(f'Name of class of model to load not specified and not saved in checkpoint: {path}')

    model_class = getattr(vseq.models, model_class_name)
    model = model_class.load(path, device=device)
    return model


class BaseModule(nn.Module):
    """Base class for end-use type Modules (e.g. models)"""

    def __init__(self):
        super().__init__()
        # TODO We may need to handle *args and **kwargs better
        #      Any "*" and "**" type arguments can be identified by checking Parameter.kind.
        #      For "*" arguments, 'Parameter.kind == Parameter.VAR_KEYWORD'
        #      For "**" arguments: 'Parameter.kind == ParameterVAR_POSITIONAL)'
        #      These could then be assigned their own hidden attribute here similar to `_init_arguments`
        #      We might also be able to unpack "**" type arguments into `_kwarg_names` and `_init_arguments`
        #      For "*" type arguments we can get the variable's name via the __init__ signature and link them to values in the order they are passed in.
        signature = inspect.signature(self.__class__.__init__)
        self._kwarg_names = [p for p in signature.parameters if p != 'self']
        self._init_arguments = None

    def init_arguments(self):
        """Return a dictionary of the kwargs used to instantiate this module"""
        if self._init_arguments is None:
            self._init_arguments = self._get_init_arguments()
        return self._init_arguments

    def _get_init_arguments(self):
        """Retrieve the values of keyword arguments used to instantiate this Module (assumes they are all properties)"""
        missing_names = [name for name in self._kwarg_names if name not in vars(self)]

        if len(missing_names) > 0:
            msg = f'Models need to define the `kwargs` to `__init__` as attributes but {str(self.__class__)} is ' \
                    f'missing the following attributes: {missing_names}.'
            raise RuntimeError(msg)

        init_arguments = {attr: getattr(self, attr) for attr in self._kwarg_names}
        return init_arguments

    @property
    def device(self):
        """Heuristically return the device which this model is on"""
        return next(self.parameters()).device

    def save(self, path):
        """Save the module class name, init_arguments and state_dict to different files in the directory given by path"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.__class__.__name__, os.path.join(path, MODEL_CLASS_NAME_STR))
        torch.save(self.init_arguments(), os.path.join(path, MODEL_INIT_KWRGS_STR))
        torch.save(self.state_dict(), os.path.join(path, MODEL_STATE_DICT_STR))

    @classmethod
    def load(cls, path, device: str = 'cpu'):
        """Return an instance of the concrete module instantiated using saved init_arguments and with state_dict loaded"""
        model_kwargs = torch.load(os.path.join(path, MODEL_INIT_KWRGS_STR))
        kwargs = model_kwargs.pop('kwargs', {})  # TODO If we handle *args and **kwargs then these could be removed
        args = model_kwargs.pop('args', [])

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
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def extra_repr(self):
        """All init_arguments as extra representation in a string formatted as a dictionary"""
        if not self.init_arguments():
            return ''
        s = ',\n  '.join(f'{k}={v}' for k, v in self.init_arguments().items() if not isinstance(v, nn.Module))
        s = 'kwargs={\n' + '  ' + s + '\n}'
        return s

    def summary(self, input_example=None, input_size=None, batch_size=1, input_dtype=torch.FloatTensor, device=None, tb_summary_writer=None, **forward_kwargs):
        """Return a summary of the model"""
        if tb_summary_writer:
            # NOTE May have to make forward pass deterministic
            x = torch.randn(input_size)
            tb_summary_writer.add_graph(self, (x,))  # check_trace of jit.trace should be false since model is stochastic

        return summary(self, input_example=input_example, input_size=input_size, batch_size=batch_size, input_dtype=input_dtype, device=device, **forward_kwargs)
