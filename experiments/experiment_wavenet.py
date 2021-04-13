import argparse
import logging

import torch

from torch.utils.data import DataLoader

import vseq
import vseq.data
import vseq.models
import vseq.training
import vseq.utils
import vseq.utils.device

from vseq.data import BaseDataset, DataModule, transforms
from vseq.data.batcher import AudioBatcher
from vseq.data.datapaths import LIBRISPEECH_DEV_CLEAN, LIBRISPEECH_TRAIN
from vseq.data.text_cleaners import clean_librispeech
from vseq.data.token_map import TokenMap
from vseq.data.tokenizers import char_tokenizer
from vseq.data.tokens import DELIMITER_TOKEN, ENGLISH_STANDARD
from vseq.data.transforms import Compose, EncodeInteger, RandomSegment, Scale
from vseq.evaluation import LLMetric, Aggregator
from vseq.utils.rand import set_seed
from vseq.utils.training import epochs


torch.autograd.set_detect_anomaly(True)
LOGGER = logging.getLogger(name=__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=500, type=int, help="number of epochs")
parser.add_argument("--lr", default=2e-3, type=float, help="base learning rate")
# parser.add_argument("--test_every", default=1, type=int, help="test every x epochs")
parser.add_argument("--seed", default=42, type=int, help="random seed")
parser.add_argument("--num_workers", default=16, type=int, help="number of dataloader workers")
# parser.add_argument("--task_name", default="", type=str, help="run task_name suffix")
# parser.add_argument("--task_tags", default=[], type=str, nargs="+", help="tags for the task")
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
    ("flac", Compose(RandomSegment(length=64000), Scale(-1, 1)), AudioBatcher()),
]


train_dataset = BaseDataset(
    source=LIBRISPEECH_TRAIN,
    modalities=modalities,
)
val_dataset = BaseDataset(
    source=LIBRISPEECH_DEV_CLEAN,
    modalities=modalities,
)

# train_sampler = FrameSampler(source=LIBRISPEECH_TRAIN, sample_rate=16000, max_seconds=40)
# val_sampler = EvalSampler(source=LIBRISPEECH_DEV_CLEAN, sample_rate=16000, max_seconds=40)

train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=args.num_workers,
    shuffle=True,
    batch_size=4
    # batch_sampler=train_sampler,
)
val_loader = DataLoader(
    dataset=val_dataset,
    collate_fn=val_dataset.collate,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=4
    # batch_sampler=val_sampler,
)

delimiter_token_idx = token_map.get_index(DELIMITER_TOKEN)
model = vseq.models.WaveNet(
    layer_size=10,
    stack_size=2,
    in_channels=1,
    res_channels=64,
    out_classes=256
)

print(model)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

summary = model.summary((64000, 1), 1, x_sl=[64000])
print(summary)

model.load('./wavenet')

# x = model.generate(n_samples=1, n_frames=16000)

# torchaudio.save('./wavenet/sample.wav', x[0].cpu(), sample_rate=16000, channels_first=True, compression=None)

import IPython; IPython.embed()


metric_ll = LLMetric(tags={'loss'})
aggregator = Aggregator(metric_ll)

for epoch in epochs(args.epochs):

    aggregator.set(train_dataset.source)
    for (x, x_sl), metadata in train_loader:
        x = x.to(device)

        loss, output = model(x, x_sl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_ll.update(output.ll, weight=output.ll.numel())

        aggregator.print()

    aggregator.log()
    aggregator.reset()

    aggregator.set(val_dataset.source)
    for (x, x_sl), metadata in train_loader:
        x = x.to(device)

        loss, output = model(x, x_sl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_ll.update(output.ll, weight=output.ll.numel())

        aggregator.print()

    aggregator.log()
    aggregator.reset()











# try:
#     for epoch in range(args.epochs):

#         mean_loss = 0
#         for step, ((x, x_sl), metadata) in enumerate(train_loader, start=1):
#             x = x.to(device)

#             # print(x.shape, x.numel)

#             loss, output = model(x, x_sl)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             mean_loss = mean_loss * ((step - 1) / step) + loss.item() / step

#             print(f"{epoch=} | batch={step}/{len(train_loader)} | {loss=:.3f} | {mean_loss=:.3f}", end='\r')
#         model.save('./wavenet')

#         print()

# except KeyboardInterrupt:
#     pass

# import IPython; IPython.embed(using=False)
