from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn.modules.sparse import Embedding
from torch.tensor import Tensor

from torchtyping import TensorType

from vseq.models.base_model import BaseModel


class CWVAEEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: TensorType["B", "T", "C"]) -> TensorType["B", "D"]:
        raise NotImplementedError()


class CWVAEEmbeddingLM(CWVAEEmbedding):
    def __init__(self, embedding_dim: int, num_embeddings: int, **kwargs):
        super().__init__(embedding_dim)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, **kwargs)

    def forward(self, x: TensorType["B", "T", "C", int]) -> TensorType["B", "D"]:
        e = self.embedding(x).sum(1)
        return e


class CWVAEEmbeddingAudio(CWVAEEmbedding):
    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__(embedding_dim)


class CWVAEStage(nn.Module):
    def __init__(self, latent_dim: int, clock_rate: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.clock_rate = clock_rate

    def forward(self, x: TensorType["B", "T", "C"], x_sl: TensorType["B", torch.int64]):
        pass

    def infer(self, x: TensorType["B", "T", "C"], x_sl: TensorType["B", torch.int64]):
        pass

    def reconstruct(self, x: TensorType["B", "T", "C"], x_sl: TensorType["B", torch.int64]):
        pass

    def generate(self):
        pass


class CWVAELM(BaseModel):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        delimiter_token_idx: int,
        latent_dims: List[int],
        clock_rates: List[int],
        hidden_size: int,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.delimiter_token_idx = delimiter_token_idx
        self.latent_dims = latent_dims
        self.clock_rates = clock_rates
        self.hidden_size = hidden_size

        self.embedding = CWVAEEmbeddingLM(num_embeddings + 1, embedding_dim)  # One embedding share among all stages
        self.mask_token_idx = num_embeddings

        stages = nn.ModuleList()
        for latent_dim, k in zip(latent_dims, clock_rates):
            stages.append(
                CWVAEStage(
                    latent_dim=latent_dim,
                    clock_rate=k,
                )
            )
        self.stages = stages

    def compute_loss(self, x):
        pass

    def infer(self, x: TensorType["B", "T", "C"], x_sl: TensorType["B", torch.int64]):
        pass

    def reconstruct(self, x: TensorType["B", "T", "C"], x_sl: TensorType["B", torch.int64]):
        pass

    def generate(self):
        pass

    def forward(self, x: TensorType["B", "T", "C"], x_sl: TensorType["B", torch.int64]):
        
        metrics = []
        output = []
        loss = self.compute_loss()
        return loss, metrics, output
