import argparse
import logging
import tqdm

import torch

from torch.utils.data import DataLoader

import vseq
import vseq.data
import vseq.models
import vseq.training
import vseq.utils
import vseq.utils.device

from vseq.data import transforms
from vseq.data import DataModule, BaseDataset
from vseq.data.collate import collate_text
from vseq.data.transforms import EncodeInteger, Compose
from vseq.data.datapaths import LIBRISPEECH_DEV_CLEAN, LIBRISPEECH_TRAIN
from vseq.data.text_cleaners import clean_librispeech
from vseq.data.tokens import ENGLISH_STANDARD, DELIMITER_TOKEN
from vseq.data.tokenizers import char_tokenizer
from vseq.data.token_map import TokenMap
from vseq.data.samplers import EvalSampler, FrameSampler
from vseq.utils.rand import set_seed

torch.autograd.set_detect_anomaly(True)
LOGGER = logging.getLogger(name=__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="VAE", help="model type (vae | lvae | biva)")
parser.add_argument("--epochs", default=500, type=int, help="number of epochs")
parser.add_argument("--lr", default=2e-3, type=float, help="base learning rate")
parser.add_argument("--test_every", default=1, type=int, help="test every x epochs")
parser.add_argument("--seed", default=42, type=int, help="random seed")
parser.add_argument("--task_name", default="", type=str, help="run task_name suffix")
parser.add_argument("--task_tags", default=[], type=str, nargs="+", help="tags for the task")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

args, _ = parser.parse_known_args()

set_seed(args.seed)

device = vseq.utils.device.get_device() if args.device == "auto" else torch.device(args.device)

token_map = TokenMap(tokens=ENGLISH_STANDARD, add_start=False, add_end=False, add_delimit=True)
PennTreebankTransform = transforms.Compose(
    transforms.TextCleaner(clean_librispeech),
    EncodeInteger(
        token_map=token_map,
        tokenizer=char_tokenizer,
    ),
)

modalities = [
    # ('flac', MelSpectrogram(), collate_spectrogram),
    ("txt", PennTreebankTransform, collate_text)
]


train_dataset = BaseDataset(
    source=LIBRISPEECH_TRAIN,
    modalities=modalities,
)
val_dataset = BaseDataset(
    source=LIBRISPEECH_DEV_CLEAN,
    modalities=modalities,
)

train_sampler = FrameSampler(source=LIBRISPEECH_TRAIN, sample_rate=16000, max_seconds=960)
val_sampler = EvalSampler(source=LIBRISPEECH_DEV_CLEAN, sample_rate=16000, max_seconds=960)

train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=16,
    # shuffle=True,
    # batch_size=16
    batch_sampler=train_sampler,
)
val_loader = DataLoader(
    dataset=val_dataset,
    collate_fn=val_dataset.collate,
    num_workers=16,
    # shuffle=False,
    # batch_size=16
    batch_sampler=val_sampler,
)

delimiter_token_idx = token_map.get_index(DELIMITER_TOKEN)
model = vseq.models.Bowman(
    num_embeddings=len(token_map), embedding_dim=256, hidden_size=256, delimiter_token_idx=delimiter_token_idx
)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


# model.summary(train_dataset[0][0].shape)


class Metric:
    def __init__(self,  name=None) -> None:
        self.name = name


class LLMetric(Metric):
    def __init__(self) -> None:
        pass


class Aggregator:
    def __init__(self, *metrics: Iterable[Metric]) -> None:
        self.metrics = metrics
        self.names = 2

    def log(self):
        pass


metric_ll = LLMetric(name='p(x|z)')
metric_kl = KLMetric(name='')
metric_elbo = ELBOMetric()
metric_perplexity = PerplexityMetric()

aggregator = Aggregator(metric_ll, metric_kl, metric_elbo, metric_perplexity)


for epoch in range(args.epochs):

    aggregator.set(train_dataset.source)
    for (x, x_sl), metadata in train_loader:
        x = x.to(device)

        loss, output = model(x, x_sl, word_dropout_rate=0.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_ll.update(output.ll)
        metric_kl.update(output.kl)
        metric_elbo.update(output.elbo)

        aggregator.log()

    aggregator.set(val_dataset.source)
    for (x, x_sl), metadata in train_loader:
        x = x.to(device)

        loss, output = model(x, x_sl, word_dropout_rate=0.0)

        metric_ll.update(output.ll)
        metric_kl.update(output.kl)
        metric_elbo.update(output.elbo)

        aggregator.log()













    step, log = 0, ""
    # for (x, x_sl), metadata in tqdm.tqdm(train_loader, total=len(train_loader)):
    mean_log_prob, mean_kl, mean_loss = 0, 0, 0
    for (x, x_sl), metadata in train_loader:
        # import IPython; IPython.embed()

        step += 1
        x = x.to(device)

        log_prob, kl_divergence, p = model(x, x_sl, word_dropout_rate=0.0)

        log_prob_reduced = log_prob.sum() / (x_sl - 1).sum()
        kl_divergence_reduced = kl_divergence.sum()
        loss = kl_divergence_reduced - log_prob_reduced

        mean_log_prob = mean_log_prob * ((step - 1) / step) + log_prob_reduced.item() / step
        mean_kl = mean_kl * ((step - 1) / step) + kl_divergence_reduced.item() / step
        mean_loss = mean_loss * ((step - 1) / step) + loss.item() / step

        print(len(log) * " ", end="\r")
        log = f"Loss={mean_loss:.3f}, " f"KL={mean_kl:.3f}, " f"log-prob={mean_log_prob:.3f}, " f"({step}/{100})"
        print(log, end="\r")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 100:
            sl = x_sl.numpy() - 1
            refs_encoded = x[:, 1:].cpu().numpy()
            hyps_encoded = p.logits.argmax(dim=-1).T.cpu().numpy()
            refs = token_map.decode_batch(refs_encoded[:2], sl[:2])
            hyps = token_map.decode_batch(hyps_encoded[:2], sl[:2])

            print("\n\nREF #1:", refs[0])
            print("HYP #1:", hyps[0])
            print("\nREF #2:", refs[1])
            print("HYP #2:", hyps[1])
            break
        # model.generate(n_samples=4)
        # break

        # loss = criterion(x, x_hat, x_sl)

        # loss.backward()
