import torch
import torch.nn as nn

import vseq.modules

from .base_module import BaseModule


class VAE(BaseModule):
    def __init__(self):
        super().__init__()
