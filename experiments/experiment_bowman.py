import argparse
import logging
import math

from typing import Iterable, Union
from torch.types import Number

import tqdm
import torch

from torch.utils.data import DataLoader

import vseq
from vseq import data
import vseq.data
import vseq.models
import vseq.training
import vseq.utils
import vseq.utils.device

from vseq.data import transforms
from vseq.data import DataModule, BaseDataset
from vseq.data.batcher import TextBatcher
from vseq.data.transforms import EncodeInteger, Compose
from vseq.data.datapaths import PENN_TREEBANK_VALID, PENN_TREEBANK_TRAIN, PENN_TREEBANK_TRAIN, PENN_TREEBANK_VALID
from vseq.data.text_cleaners import clean_librispeech
from vseq.data.tokens import ENGLISH_STANDARD, DELIMITER_TOKEN
from vseq.data.tokenizers import char_tokenizer, word_tokenizer
from vseq.data.token_map import TokenMap
from vseq.data.samplers import EvalSampler, FrameSampler
from vseq.utils.rand import set_seed
from vseq.data.vocabulary import load_vocabulary


torch.autograd.set_detect_anomaly(True)
LOGGER = logging.getLogger(name=__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="VAE", help="model type (vae | lvae | biva)")
parser.add_argument("--epochs", default=500, type=int, help="number of epochs")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument("--num_workers", default=16, type=int, help="number of dataloader workers")
parser.add_argument("--test_every", default=1, type=int, help="test every x epochs")
parser.add_argument("--seed", default=42, type=int, help="random seed")
parser.add_argument("--task_name", default="", type=str, help="run task_name suffix")
parser.add_argument("--task_tags", default=[], type=str, nargs="+", help="tags for the task")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

args, _ = parser.parse_known_args()

set_seed(args.seed)

device = vseq.utils.device.get_device() if args.device == "auto" else torch.device(args.device)

vocab = load_vocabulary(PENN_TREEBANK_TRAIN)
token_map = TokenMap(tokens=vocab, add_start=False, add_end=False, add_delimit=True)
penn_treebank_transform = EncodeInteger(
    token_map=token_map,
    tokenizer=word_tokenizer,
)
batcher = TextBatcher()

modalities = [
    ("txt", penn_treebank_transform, batcher)
]

train_dataset = BaseDataset(
    source=PENN_TREEBANK_TRAIN,
    modalities=modalities,
)
val_dataset = BaseDataset(
    source=PENN_TREEBANK_VALID,
    modalities=modalities,
)

train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=args.num_workers,
    shuffle=True,
    batch_size=args.batch_size
)
val_loader = DataLoader(
    dataset=val_dataset,
    collate_fn=val_dataset.collate,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=args.batch_size
)

delimiter_token_idx = token_map.get_index(DELIMITER_TOKEN)
model = vseq.models.Bowman(
    num_embeddings=len(token_map),
    embedding_dim=256,
    hidden_size=256,
    delimiter_token_idx=delimiter_token_idx
)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# x, x_sl = next(iter(train_loader))[0]
# x = x.to(device)
# x_sl = x_sl.to(device)
# model.summary(x.shape, batch_size=1, x_sl=x_sl)


class Metric:
    base_tags = set()

    def __init__(self, name: str, tags: set, accumulate: bool, keep_on_device: bool):
        self.name = name
        self.tags = self.base_tags if tags is None else (tags | self.base_tags)
        self.accumulate = accumulate
        self.keep_on_device = keep_on_device
        self.accumulated_values = []

    @property
    def log_value(self):
        raise NotImplementedError()

    @property
    def str_value(self):
        raise NotImplementedError()

    def _accumulate(self, value):
        if not self.keep_on_device:
            value = value.cpu()
        self.accumulated_values.append(value)

    def update(self, value: torch.Tensor):
        raise NotImplementedError()

    def reset(self):
        self.accumulated_values = []


class RunningMean(Metric):
    _str_value_fmt = '10.3'

    def __init__(self, name: str, tags: set, accumulate: bool = False, keep_on_device: bool = False):
        super().__init__(name=name, tags=tags, accumulate=accumulate, keep_on_device=keep_on_device)
        self._running = 0
        self._running_weight = 0

    @property
    def log_value(self):
        return self._running

    @property
    def str_value(self):
        return f"{self._running:{self._str_value_fmt}f}"

    def update(self, value: Union[torch.Tensor, Number], weight=1.0):

        if self.accumulate:
            self.accumulate(value)

        weight = weight.cpu().item() if isinstance(weight, torch.Tensor) else weight    
        value = value.sum().cpu().item() if isinstance(value, torch.Tensor) else value

        self._running_weight += weight

        weight_value = weight / self._running_weight
        weight_running = (self._running_weight - weight) / self._running_weight
        self._running = value * weight_value + self._running * weight_running
    
    def reset(self):
        super().reset()
        self.running = 0
        self.running_weight = 0


class LLMetric(RunningMean):
    base_tags = {'likelihood'}

    def __init__(self, name: str = "ll", tags: set = None, accumulate: bool = False, keep_on_device: bool = False):
        super().__init__(name=name, tags=tags, accumulate=accumulate, keep_on_device=keep_on_device)


class ELBOMetric(RunningMean):
    base_tags = {'likelihood'}

    def __init__(self, name: str = "elbo", tags: set = None, accumulate: bool = False, keep_on_device: bool = False):
        super().__init__(name=name, tags=tags, accumulate=accumulate, keep_on_device=keep_on_device)


class KLMetric(RunningMean):
    base_tags = {'kl_divergences'}

    def __init__(self, name: str = "kl", tags: set = None, accumulate: bool = False, keep_on_device: bool = False):
        super().__init__(name=name, tags=tags, accumulate=accumulate, keep_on_device=keep_on_device)


class PerplexityMetric(RunningMean):
    base_tags = set()

    def __init__(self, name: str = "pp", tags: set = None, accumulate: bool = False, keep_on_device: bool = False):
        super().__init__(name=name, tags=tags, accumulate=accumulate, keep_on_device=keep_on_device)

    def update(self, log_likelihood):
        value = log_likelihood / math.log(2)
        value = 2 ** value


class BitsPerDim(RunningMean):
    base_tags = set()

    def __init__(self, name: str = "bpd", tags: set = None, accumulate: bool = False, keep_on_device: bool = False):
        super().__init__(name, tags, accumulate=accumulate, keep_on_device=keep_on_device)

from types import SimpleNamespace


class Aggregator:
    def __init__(self, *metrics: Iterable[Metric], source_name_len: int = 15) -> None:
        self.metrics = metrics
        self.source_name_len = source_name_len
        self.current_source = None
        self.steps = 0

    def set(self, source_name):
        self.current_source = source_name

    def reset(self):
        self.print(end='\n')
        log_values = SimpleNamespace(**{metric.name: metric.log_value for metric in self.metrics})
        setattr(self, self.current_source, log_values)
        for metric in self.metrics:
            metric.reset()

        self.steps = 0
        self.current_source = None

    def print(self, end='\r'):
        s = [f"{metric.name} = {metric.str_value}" for metric in self.metrics]
        s = f"{self.current_source:<{self.source_name_len}s} | " + " | ".join(s)
        print(s, end=end)

    def log(self):
        pass

    # def __call__(self, loader):
    #     self.current_source = loader.dataset.source
        
    #     n, N = 0, len(loader)
    #     for batch in loader:
    #         yield batch
    #         n += 1
    #         self.log(end=("\n" if n == N else "\r"))
            
    #     log_values = SimpleNamespace(**{metric.name: metric.log_value for metric in self.metrics})
    #     setattr(self, self.current_source, log_values)

    #     for metric in self.metrics:
    #         metric.reset()


metric_elbo = ELBOMetric(tags={'loss'})
metric_ll = LLMetric()
metric_kl = KLMetric()
metric_perplexity = PerplexityMetric()

aggregator = Aggregator(metric_ll, metric_kl, metric_elbo, metric_perplexity)


def epochs(N):
    BOLD = '\033[1m'
    END = '\033[0m'
    for epoch in range(1, N + 1):
        print(BOLD + f"\nEpoch {epoch}:" + END)
        yield epoch


for epoch in epochs(args.epochs):

    aggregator.set(train_dataset.source)
    for (x, x_sl), metadata in train_loader:
        x = x.to(device)

        loss, output = model(x, x_sl, word_dropout_rate=0.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_elbo.update(output.elbo, weight=output.elbo.numel())
        metric_ll.update(output.ll.sum() / (x_sl - 1).sum(), weight=(x_sl - 1).sum())
        metric_kl.update(output.kl, weight=output.kl.numel())

        aggregator.print()

    # aggregator.log()
    aggregator.reset()

    aggregator.set(val_dataset.source)
    for (x, x_sl), metadata in train_loader:
        x = x.to(device)

        loss, output = model(x, x_sl, word_dropout_rate=0.0)

        metric_elbo.update(output.elbo, weight=output.elbo.numel())
        metric_ll.update(output.ll.sum() / (x_sl - 1).sum(), weight=(x_sl - 1).sum())
        metric_kl.update(output.kl, weight=output.kl.numel())

        aggregator.print()

    # aggregator.log()
    aggregator.reset()


    # wandb.log("train_loss", aggregator.train.loss)
    # wandn.log("test_loss", aggregator.test.loss)

    # step, log = 0, ""
    # # for (x, x_sl), metadata in tqdm.tqdm(train_loader, total=len(train_loader)):
    # mean_log_prob, mean_kl, mean_loss = 0, 0, 0
    # for (x, x_sl), metadata in train_loader:
    #     # import IPython; IPython.embed()

    #     step += 1
    #     x = x.to(device)

    #     log_prob, kl_divergence, p = model(x, x_sl, word_dropout_rate=0.0)

    #     log_prob_reduced = log_prob.sum() / (x_sl - 1).sum()
    #     kl_divergence_reduced = kl_divergence.sum()
    #     loss = kl_divergence_reduced - log_prob_reduced

    #     mean_log_prob = mean_log_prob * ((step - 1) / step) + log_prob_reduced.item() / step
    #     mean_kl = mean_kl * ((step - 1) / step) + kl_divergence_reduced.item() / step
    #     mean_loss = mean_loss * ((step - 1) / step) + loss.item() / step

    #     print(len(log) * " ", end="\r")
    #     log = f"Loss={mean_loss:.3f}, " f"KL={mean_kl:.3f}, " f"log-prob={mean_log_prob:.3f}, " f"({step}/{100})"
    #     print(log, end="\r")

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     if step == 100:
    #         sl = x_sl.numpy() - 1
    #         refs_encoded = x[:, 1:].cpu().numpy()
    #         hyps_encoded = p.logits.argmax(dim=-1).T.cpu().numpy()
    #         refs = token_map.decode_batch(refs_encoded[:2], sl[:2])
    #         hyps = token_map.decode_batch(hyps_encoded[:2], sl[:2])

    #         print("\n\nREF #1:", refs[0])
    #         print("HYP #1:", hyps[0])
    #         print("\nREF #2:", refs[1])
    #         print("HYP #2:", hyps[1])
    #         break
    #     # model.generate(n_samples=4)
    #     # break

    #     # loss = criterion(x, x_hat, x_sl)

    #     # loss.backward()
