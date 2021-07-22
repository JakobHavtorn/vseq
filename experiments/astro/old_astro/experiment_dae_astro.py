import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["WANDB_MODE"] = "disabled" # equivalent to "wandb disabled"

import argparse
import logging
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import rich
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import vseq
import vseq.data
import vseq.models
import vseq.utils
import vseq.utils.device

from vseq.data import BaseDataset
from vseq.data.batchers import TextBatcher, AudioBatcher
from vseq.data.datapaths import LIBRISPEECH_TRAIN, LIBRISPEECH_DEV_CLEAN
from vseq.data.tokens import ENGLISH_STANDARD
from vseq.data.tokenizers import char_tokenizer
from vseq.data.loaders import TextLoader
from vseq.data.token_map import TokenMap
from vseq.data.transforms import Compose, EncodeInteger, TextCleaner, AstroSpeech
from vseq.data.samplers import LengthTrainSampler, LengthEvalSampler
from vseq.evaluation import Tracker
from vseq.training import set_dropout
from vseq.utils.rand import set_seed, get_random_seed
from vseq.models import AstroDAE

# for tests
from types import SimpleNamespace
from vseq.modules import LSTMBlock
from vseq.data.transforms import Compose, MuLawEncode, Quantize
from vseq.evaluation import LossMetric, WindowMeanMetric, SeqAccuracyMetric
from vseq.data.token_map import TokenMap
from vseq.utils.operations import sequence_mask

parser = argparse.ArgumentParser()
parser.add_argument("--sample_rate", default=800, type=int, help="sample rate")

parser.add_argument("--max_len", default=2400, type=int, help="max sum of batch lengths")
parser.add_argument("--max_pool_difference", default=10, type=int, help="max char variation in batch")

parser.add_argument("--duration", default=50, type=int, help="number of frames for each astro speech token")
parser.add_argument("--fade", default=15, type=int, help="cos-annealing duration for fading in/out astro tokens")
parser.add_argument("--min_mel", default=200, type=int, help="the mel-frequency shift between tokens")
parser.add_argument("--mel_delta", default=10, type=int, help="the mel-frequency shift between tokens")
parser.add_argument("--token_shift", default=0, type=int, help="frequency span for each astro token")

parser.add_argument("--lr_max", default=3e-4, type=float, help="start learning rate")
parser.add_argument("--lr_min", default=5e-5, type=float, help="end learning rate")
parser.add_argument("--optimizer", default='Adam', type=str, help="optimizer")
parser.add_argument("--optimizer_kwargs", default='{}', type=json.loads, help="extra kwargs for optimizer")

parser.add_argument("--conv_kernels", default=[10, 10, 5], nargs='+', type=int, help="Conv1D kernels")
parser.add_argument("--lstm_layers", default=1, type=int, help="number of LSTM layers pr block")
parser.add_argument("--hidden_size", default=256, type=int, help="size of the LSTM layers")
parser.add_argument("--dropout_prob", default=0.0, type=float, help="size of the LSTM layers")

parser.add_argument("--epochs", default=30, type=int, help="number of epochs")
parser.add_argument("--warm_up", default=5, type=int, help="epochs before lr annealing starts")
parser.add_argument("--num_workers", default=4, type=int, help="number of dataloader workers")
parser.add_argument("--seed", default=None, type=int, help="random seed")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

args, _ = parser.parse_known_args()


if args.seed is None:
    args.seed = get_random_seed()

set_seed(args.seed)

device = vseq.utils.device.get_device() if args.device == "auto" else torch.device(args.device)

wandb.init(
    entity="vseq",
    project="dae-astro-libri",
    group=None,
)
wandb.config.update(args)

rich.print(vars(args))

token_map = TokenMap(tokens=ENGLISH_STANDARD)
output_size = len(token_map)

text_loader = TextLoader("txt", cache=True)
text_cleaner = TextCleaner(lambda s: s.lower().strip())
encode_int_out = EncodeInteger(token_map=token_map, tokenizer=char_tokenizer)
text_transform = Compose(
    text_cleaner,
    encode_int_out
)
text_batcher = TextBatcher()

encode_int_in = EncodeInteger(
    token_map=token_map,
    tokenizer=char_tokenizer
)
astro_speech = AstroSpeech(
    num_tokens=len(token_map),
    whitespace_idx=token_map.token2index[" "],
    sample_rate=args.sample_rate,
    duration=args.duration,
    fade=args.fade,
    mel_delta=args.mel_delta,
    token_shift=args.token_shift
)
audio_transform = Compose(
    text_cleaner,
    encode_int_in,
    astro_speech
)
audio_batcher = AudioBatcher()

modalities = [
    (text_loader, audio_transform, audio_batcher),
    (text_loader, text_transform, text_batcher)
]

train_dataset = BaseDataset(
    source=LIBRISPEECH_TRAIN,
    modalities=modalities,
)
val_dataset = BaseDataset(
    source=LIBRISPEECH_DEV_CLEAN,
    modalities=modalities,
)

train_sampler = LengthTrainSampler(
    source=LIBRISPEECH_TRAIN,
    field="length.txt.chars",
    max_len=float(args.max_len),
    max_pool_difference=float(args.max_pool_difference),
    num_batches=2000
)

val_sampler = LengthEvalSampler(
    source=LIBRISPEECH_DEV_CLEAN,
    field="length.txt.chars",
    max_len=float(args.max_len),
)

train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=args.num_workers,
    batch_sampler=train_sampler
)

val_loader = DataLoader(
    dataset=val_dataset,
    collate_fn=val_dataset.collate,
    num_workers=args.num_workers,
    batch_sampler=val_sampler
)

# model = AstroDAE(
#     token_map=token_map,
#     in_channels=1,
#     kernels=args.conv_kernels,
#     hidden_size=args.hidden_size,
#     lstm_layers=args.lstm_layers,
#     bits=8
# )

class Model(nn.Module):

    def __init__(self, prior=None):
        
        super().__init__()

        # encoder
        self.enc_conv = nn.Conv1d(
            in_channels=1,
            out_channels=256,
            kernel_size=10,
            stride=10
        )
        self.enc_act = nn.ReLU6()
        self.enc_conv_2 = nn.Conv1d(
            in_channels=256,
            out_channels=len(token_map),
            kernel_size=5,
            stride=5
        )

        # decoder
        self.dec_conv = nn.ConvTranspose1d(
            in_channels=len(token_map),
            out_channels=256,
            kernel_size=5,
            stride=5
        )
        self.dec_act = nn.ReLU6()
        self.dec_conv_2 = nn.ConvTranspose1d(
            in_channels=256,
            out_channels=2 ** 8,
            kernel_size=10,
            stride=10
        )

        # define uniform prior
        self.prior = prior if prior is not None else torch.ones(len(token_map)) / len(token_map)
        self.k = len(token_map)

        # target encode
        mu_law_encoder = MuLawEncode()
        quantizer = Quantize(bits=8)
        self.target_encode = Compose(mu_law_encoder, quantizer)

    def set_transform_device(self, device):

        b = self.target_encode.transforms[1].boundaries
        self.target_encode.transforms[1].boundaries = b.to(device)
        self.prior = self.prior.to(device)

    def forward(self, x, x_sl, y, y_sl, tau=1.0, hard=True):
        
        t = self.target_encode(x)

        if x.ndim == 2:
            x = x.unsqueeze(dim=1)

        logits = self.enc_conv(x)
        logits = self.enc_act(logits)
        logits = self.enc_conv_2(logits)
        logits = logits.permute(0, 2, 1)

        if isinstance(hard, bool):
            z_btd = F.gumbel_softmax(logits=logits, tau=tau, hard=hard)
        else:
            if self.training:
                gumbel_noise = -torch.log(-torch.log(torch.rand(logits.shape))).to(x.device)
                gumbel_logits = logits + gumbel_noise
            else:
                gumbel_logits = logits
            shift_logits = gumbel_logits - gumbel_logits.max(dim=2, keepdim=True)[0] # softmax max trick
            exps = shift_logits.exp() / tau
            sm = exps / exps.sum(dim=2, keepdim=True)
            oh = F.one_hot(sm.argmax(dim=2), num_classes=self.k)
            z_btd = sm + hard * (oh - sm.detach())

        z_sl = x_sl // 50
        tm_z = sequence_mask(z_sl, dtype=torch.float32, device=x.device) # (B, T)
        z_btd = z_btd * tm_z.unsqueeze(2)
        z = z_btd.permute(0, 2, 1)

        r = self.dec_conv(z)
        r = self.dec_act(r)
        r = self.dec_conv_2(r)
        r = r.permute(0, 2, 1) # (B, D, T) --> (B, T, D)

        p_y = r - r.logsumexp(dim=-1, keepdim=True)
        log_prob = p_y.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        tm = sequence_mask(x_sl, dtype=torch.float32, device=x.device)
        log_prob = log_prob * tm
        
        rec_loss = - log_prob.sum() / x_sl.sum()

        # compute sample entropy (if hard --> 0)
        H_pd =  - (z_btd * torch.log(z_btd + 1e-10)).sum(-1) * tm_z
        H = H_pd.sum() / z_sl.sum()

        # compute div loss # 0 (not great)
        # softmax = (logits * tm_z.unsqueeze(2)).mean(dim=(0, 1)).softmax(0)
        # D =  - (softmax * torch.log(softmax)).sum()
        
        # alternative div loss # 1 (better)
        # softmax = (logits.softmax(dim=-1) * tm_z.unsqueeze(2)).sum(dim=(0, 1)) / z_sl.sum()
        # D =  - (softmax * torch.log(softmax)).sum()

        # import IPython; IPython.embed()
        # alternative div loss # 2 (best observed with 0.04 rec)
        # softmax = z_btd.sum(dim=(0, 1)) / z_sl.sum() # or softmax normalization
        # D =  - (softmax * torch.log(softmax + 1e-10)).sum()

        # alternative div loss # 3 (uniform KL)
        conj_post = z_btd.sum(dim=(0, 1)) / z_sl.sum() # or softmax normalization
        kl = (conj_post * torch.log(conj_post / (self.prior + 1e-10) + 1e-10)).sum()

        # alternative div loss # 3 (per step uniform KL)1/2
        # s = z_btd.argmax(dim=2, keepdim=True)
        # p_z = torch.log(self.prior).reshape([1, 1, -1]).repeat(s.shape)
        # q_z = logits - logits.logsumexp(dim=-1, keepdim=True)
        # q_log_prob = q_z.gather(-1, s).squeeze(-1)
        # q_log_prob = q_log_prob * tm_z
        # p_log_prob = p_z.gather(-1, s).squeeze(-1)
        # p_log_prob = p_log_prob * tm_z
        # kl = (q_log_prob - p_log_prob).sum() / z_sl.sum()
        
        
        
        #kl = - log_prob.sum() / z_sl.sum()

        # add diversity loss:
        loss = rec_loss + kl
        #loss = rec_loss

        
        outputs = SimpleNamespace(
            logits=logits,
            z=z_btd
        )

        metrics = [
            LossMetric(rec_loss, weight_by=x_sl.sum(), name="rec"),
            LossMetric(kl, weight_by=z_sl.sum(), name="kl"),
            WindowMeanMetric(rec_loss, name="rec10"),
            WindowMeanMetric(kl, name="kl10"),
            WindowMeanMetric(H, name="H10"),
            SeqAccuracyMetric(z_btd.argmax(2), y, tm_z, name="acc"),
            WindowMeanMetric(torch.Tensor([tau]), name="tau")
        ]

        return loss, metrics, outputs

# build unigram prior
prior = None
if prior is not None:
    chars = []
    for ((x, x_sl), (y, y_sl)), metadata in train_loader:
        m = sequence_mask(y_sl)
        chars.append(y[m])
    chars = torch.cat(chars, dim=0)
    counts = torch.bincount(chars, minlength=len(token_map))
    prior = counts / counts.sum()


model = Model(prior=prior)


model.to(device)
model.set_transform_device(device)

optimizer = getattr(torch.optim, args.optimizer)
optimizer = optimizer(model.parameters(), lr=args.lr_max, **args.optimizer_kwargs)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs - args.warm_up), eta_min=args.lr_min)

tracker = Tracker()

for epoch in tracker.epochs(args.epochs):

    # update hyperparams (post)
    p = (epoch - 1) / (args.epochs - 1)
    tau = 1.1 - p

    # training
    model.train()
    for ((x, x_sl), (y, y_sl)), metadata in tracker.steps(train_loader):

        x = x.to(device)
        y = y.to(device)

        loss, metrics, outputs = model(x, x_sl, y, y_sl, tau=tau, hard=p)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update(metrics)

    tracker.reset()

    # evaluation
    model.eval()
    refs, hyps, lens = [], [], []
    with torch.no_grad():
        for ((x, x_sl), (y, y_sl)), metadata in tracker.steps(val_loader):
            
            x = x.to(device)
            y = y.to(device)

            loss, metrics, outputs = model(x, x_sl, y, y_sl, tau=tau, hard=1.0)
            refs.append(y.cpu())
            hyps.append(outputs.logits.argmax(-1).cpu())
            lens.append(y_sl)
            tracker.update(metrics)

    tracker.reset()

    # update hyperparams (post)
    if epoch >= args.warm_up:
        lr_scheduler.step()

def optimal_perm_error(refs, hyps, lens, token_map):

    lens = torch.cat(lens, dim=0)
    max_len = lens.max().item()

    prefs, phyps = [], []
    for r, h in zip(refs, hyps):
        p = max_len - r.size(1)
        b = r.size(0)
        pad = torch.zeros([b,p], dtype=torch.int64)
        prefs.append(torch.cat([r, pad], dim=1))
        phyps.append(torch.cat([h, pad], dim=1))
    refs = torch.cat(prefs, dim=0)
    hyps = torch.cat(phyps, dim=0)

    m = sequence_mask(lens, dtype=torch.bool)

    hyps = hyps[m]
    refs = refs[m]

    c, t = 0, 0
    bins = []
    while len(hyps) > 0:
        i = torch.mode(hyps).values.item()
        f = (hyps == i)
        b = torch.bincount(refs[f], minlength=len(token_map))
        bins.append(b.unsqueeze(0))
        c += b.max().item()
        t += f.sum().item()
        hyps = hyps[~f]
        refs = refs[~f]

    return c / t, torch.cat(bins, 0)

acc, hist = optimal_perm_error(refs, hyps, lens, token_map)

import numpy as np
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

vegetables = [str(j) for j in list(range(hist.size(0)))]
farmers = [str(j) for j in list(range(hist.size(1)))]

harvest = hist.numpy()


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

fig.tight_layout()
plt.show()
