import argparse
import os
import rich

from torch.utils.data import DataLoader

import vseq
import vseq.data
import vseq.models
import vseq.utils
import vseq.utils.device

from vseq.data import BaseDataset
from vseq.data.batchers import AudioBatcher
from vseq.data.datapaths import TIMIT_TEST, TIMIT_TRAIN
from vseq.data.loaders import AudioLoader
from vseq.data.transforms import Compose, MuLawDecode, MuLawEncode
from vseq.data.samplers.batch_samplers import LengthTrainSampler, LengthEvalSampler
from vseq.evaluation import Tracker


NUM_EPOCHS = 2
BATCH_SIZE = 0
NUM_WORKERS = 1


decode_transform = []
encode_transform = []
encode_transform.append(MuLawEncode(bits=8))  #args.num_bits))
decode_transform.append(MuLawDecode(bits=8))  #args.num_bits))

encode_transform = Compose(*encode_transform)
decode_transform = Compose(*decode_transform)

batcher = AudioBatcher(padding_module=160)
loader = AudioLoader("wav", cache=False)
modalities = [(loader, encode_transform, batcher)]

train_dataset = BaseDataset(
    source=TIMIT_TRAIN,
    modalities=modalities,
)
valid_dataset = BaseDataset(
    source=TIMIT_TEST,
    modalities=modalities,
)
rich.print(train_dataset)


train_sampler = LengthTrainSampler(
    source=TIMIT_TRAIN,
    field="length.wav.samples",
    max_len=16000 * BATCH_SIZE if BATCH_SIZE > 0 else "max",
    max_pool_difference=16000 * 0.3,
    min_pool_size=512,
)
valid_sampler = LengthEvalSampler(
    source=TIMIT_TEST,
    field="length.wav.samples",
    max_len=16000 * BATCH_SIZE if BATCH_SIZE > 0 else "max",
)
train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=NUM_WORKERS,
    batch_sampler=train_sampler,
    pin_memory=True,
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    collate_fn=valid_dataset.collate,
    num_workers=NUM_WORKERS,
    batch_sampler=valid_sampler,
    pin_memory=True,
)

tracker = Tracker(print_every=1.0)

for epoch in tracker.epochs(NUM_EPOCHS):
    for (x, x_sl), metadata in tracker.steps(train_loader):
        pass

    for (x, x_sl), metadata in tracker.steps(valid_loader):
        pass

tracker = Tracker(print_every=int(1))

iterations = 10000
for epoch in tracker.epochs(NUM_EPOCHS):
    for i in tracker.steps(range(iterations), source="dataless", max_steps=iterations):
        pass
    print((tracker.end_time["dataless"] - tracker.start_time["dataless"]) / iterations * 1e6, "µs per iteration")
