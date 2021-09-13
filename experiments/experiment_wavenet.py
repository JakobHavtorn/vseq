# sourcery skip: remove-unreachable-code
import os
import argparse
import logging
from types import SimpleNamespace

import torch
import wandb
import rich

from torch.utils.data import DataLoader

import vseq.models

from vseq.data import BaseDataset
from vseq.data.batchers import AudioBatcher, SpectrogramBatcher
from vseq.data.datapaths import (
    TIMIT_TRAIN,
    TIMIT_TEST,
    LIBRISPEECH_DEV_CLEAN,
    LIBRISPEECH_TRAIN,
)
from vseq.data.loaders import AudioLoader
from vseq.data.transforms import (
    Compose,
    Quantize,
    RandomSegment,
    Scale,
    MuLawEncode,
    StackWaveform,
    MuLawDecode,
)
from vseq.evaluation.tracker import Tracker
from vseq.utils.argparsing import str2bool
from vseq.utils.device import get_device
from vseq.utils.rand import set_seed, get_random_seed


def prep_audio_for_save(audio: torch.TensorType) -> torch.TensorType:
    # TODO: this assumes mono channel
    _audio = audio.squeeze().cpu().flatten()
    if len(_audio.shape) == 1:
        _audio = _audio.unsqueeze(-1)
    return _audio


N_SAMPLES_SAVE = 5  # TODO: make arg

LOGGER = logging.getLogger(name=__file__)
# fmt: off
parser = argparse.ArgumentParser()

parser.add_argument("--input_length", default=16000, type=int, help="input length")
parser.add_argument("--batch_size", default=4, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--n_layers", default=10, type=int, help="number of layers per stack")
parser.add_argument("--n_stacks", default=4, type=int, help="number of stacks")
parser.add_argument("--res_channels", default=64, type=int, help="number of channels in residual connections")
parser.add_argument("--input_encoding", default="mu_law", type=str, choices=["mu_law", "raw"], help="whether to encode the input")
parser.add_argument("--input_embedding", default="stacked", type=str, choices=["frames", "quantized", "stacked", "spectrogram"], help="input encoding")
parser.add_argument("--num_bits", default=8, type=int, help="number of bits for mu_law encoding (note the data bits depth)")
parser.add_argument("--stack_frames", default=1, type=int, help="Number of audio frames to stack in feature vector if input_coding is frames")
parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
parser.add_argument("--cache_dataset", default=False, type=str2bool, help="if True, cache the dataset in RAM")
parser.add_argument("--num_workers", default=8, type=int, help="number of dataloader workers")
parser.add_argument("--wandb_group", default=None, type=str, help="custom group for this experiment (optional)")
parser.add_argument("--seed", default=None, type=int, help="seed for random number generators. Random if -1.")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
parser.add_argument("--save_freq", default=10, type=int, help="number of epochs to go between saves")
parser.add_argument("--delete_last_model", default=False, type=str2bool, help="if True, delete the last model saved")
parser.add_argument("--dataset", default="timit", choices=['librispeech','timit'], help="which dataset to use", type=str)
parser.add_argument("--model_load", default=None, type=str, help="model to load")
parser.add_argument("--likelihood", default="dmol", type=str, choices=["conv", "dmol"], help="likelihood type")



args = parser.parse_args()
# fmt: on
if args.seed is None:
    args.seed = get_random_seed()
set_seed(args.seed)

device = get_device() if args.device == "auto" else torch.device(args.device)


# if args.likelihood == "conv" and args.input_embedding != 'quantized':
#     raise # ??? 

### COPY ARGS TO MODEL ARGS
wavenet_args = SimpleNamespace(
    in_channels=1,
    n_layers=args.n_layers,
    n_stacks=args.n_stacks,
    res_channels=args.res_channels,
    out_classes=256,
    num_embeddings=None,
    likelihood=args.likelihood,
)


# If the input embedding is frames or a spectrogram, don't use embeddings.
# If we use a quantized input we instantiate an embedding of res_channels

_transforms = [RandomSegment(args.input_length)]
transform_decode = None
if args.input_encoding == "mu_law":
    _transforms.append(MuLawEncode(args.num_bits))
    transform_decode = MuLawDecode(args.num_bits)


if args.input_embedding == "quantized":
    embedding_str = args.input_embedding
    wavenet_args.num_embeddings = 2 ** args.num_bits
    _transforms.append(Quantize(bits=args.num_bits))
    _Batcher = AudioBatcher
    dataset_sort = True

elif args.input_embedding == "stacked":
    embedding_str = args.input_embedding + "_" + str(args.stack_frames)
    wavenet_args.stack_waveform = True

    _transforms.append(StackWaveform(n_frames=args.stack_frames))
    dataset_sort = False
    _Batcher = SpectrogramBatcher
    wavenet_args.in_channels = args.stack_frames

elif args.input_embedding == "spectrogram":
    embedding_str = args.input_embedding + "_" + str(args.spectrogram_size)
    raise NotImplementedError
    _Batcher = SpectrogramBatcher
    dataset_sort = False
    wavenet_args.in_channels = args.spectrogram_size
else:  # frames
    embedding_str = args.input_embedding
    _Batcher = AudioBatcher
    dataset_sort = True

# Defining transforms
wavenet_transform = Compose(*_transforms)
rich.print(wavenet_transform)

#######################
### MODEL INSTANTIATION
#######################
rich.print(wavenet_args)

if args.model_load is not None:
    # TODO: Add parsing here
    rich.print(f"Loading model from: {args.model_load}")
    model = vseq.models.WaveNet.load(args.model_load)
else:
    model = vseq.models.WaveNet(**vars(wavenet_args))
rich.print(model)


model_name_str = f"wavenet-{embedding_str}-{wavenet_args.n_layers}layers-{wavenet_args.n_stacks}stacks-{wavenet_args.res_channels}res-{model.receptive_field}RF"
print(f"Initialized model with name: {model_name_str}")
wandb.init(
    entity="vseq", project="wavenet", group=args.wandb_group, name=model_name_str
)
wandb.config.update(args)
rich.print(vars(args))


if args.input_embedding == "stacked":
    rich.print(
        f"RECEPTIVE FIELD: {model.receptive_field} * {args.stack_frames} = {model.receptive_field * args.stack_frames}"
    )
else:
    rich.print(f"RECEPTIVE FIELD: {model.receptive_field}")


modalities = [
    (
        AudioLoader("wav"),
        wavenet_transform,
        _Batcher(min_length=model.receptive_field + 1),
    ),
]
if args.dataset == "timit":
    TRAIN_SET = TIMIT_TRAIN
    VAL_SET = TIMIT_TEST
elif args.dataset == "librispeech":
    TRAIN_SET = LIBRISPEECH_TRAIN
    VAL_SET = LIBRISPEECH_DEV_CLEAN

train_dataset = BaseDataset(source=TRAIN_SET, modalities=modalities, sort=dataset_sort)
val_dataset = BaseDataset(source=VAL_SET, modalities=modalities, sort=dataset_sort)

train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=args.num_workers,
    shuffle=True,
    batch_size=args.batch_size,
    pin_memory=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    collate_fn=val_dataset.collate,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=args.batch_size,
    pin_memory=True,
)


# exit()
(x, x_sl), metadata = next(iter(train_loader))
# rich.print(x.shape)

model = model.to(device)
model.summary(input_data=x, x_sl=x_sl)

# exit()

wandb.watch(model, log="all", log_freq=len(train_loader))


# x = x.to(device)
# print(model.summary(input_example=x, x_sl=x_sl))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

tracker = Tracker()

for epoch in tracker.epochs(args.epochs):

    model.train()
    for (x, x_sl), metadata in tracker.steps(train_loader):
        x = x.to(device)

        loss, metrics, output = model(x, x_sl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update(metrics)

    model.eval()
    with torch.no_grad():
        for (x, x_sl), metadata in tracker.steps(val_loader):
            x = x.to(device)

            loss, metrics, output = model(x, x_sl)

            tracker.update(metrics)

        # reconstructions = [
        #     wandb.Audio(
        #         output.x_hat[i].cpu().flatten().numpy(),
        #         caption=f"Reconstruction {i}",
        #         sample_rate=16000,
        #     )
        #     for i in range(2)
        # ]

        # samples = [
        #     wandb.Audio(
        #         x[i].flatten().cpu().numpy(), caption=f"Sample {i}", sample_rate=16000
        #     )
        #     for i in range(2)
        # ]

        # tracker.log(samples=samples, reconstructions=reconstructions)

        tracker.log()
    if epoch % args.save_freq == 0:  # and epoch != 0 :  # epoch 10, 20, ...
        if args.delete_last_model and os.path.exists(
            f"./models/{model_name_str}-epoch-{epoch-args.save_freq}"
        ):
            # delete past model
            os.removedirs(f"./models/{model_name_str}-epoch-{epoch-args.save_freq}")
        model_save_path = f"./models/{model_name_str}-epoch-{epoch}"
        model.save(model_save_path)
        rich.print(f"Saved model to {model_save_path}")

        rich.print("X SHAPE")
        rich.print(x.shape)

        rich.print("X HAT SHAPE")
        rich.print(output.x_hat.shape)
        output.x_hat = (
            transform_decode(output.x_hat)
            if transform_decode is not None
            else output.x_hat
        )
        x = transform_decode(x) if transform_decode is not None else x
        for i in range(min(args.batch_size, N_SAMPLES_SAVE)):
            # save reference (true) samples

            torchaudio.save(
                f"./wavenet_samples/{model_name_str}-epoch-{epoch}-tru_sample_{i}.wav",
                prep_audio_for_save(x[i]),
                sample_rate=16000,
                channels_first=False,
                encoding="ULAW",
            )

            # save reconstruction of true samples
            torchaudio.save(
                f"./wavenet_samples/{model_name_str}-epoch-{epoch}-rec_sample_{i}.wav",
                prep_audio_for_save(output.x_hat[i]),
                sample_rate=16000,
                channels_first=False,
                encoding="ULAW",
            )

        # save generated samples
        # samples = [wandb.Audio(x[i].flatten().cpu().numpy(), caption=f"Sample {i}", sample_rate=16000) for i in range(2)]

        x_gen = model.generate(
            n_samples=N_SAMPLES_SAVE, n_frames=128000 // args.stack_frames
        ).cpu()
        x_gen = transform_decode(x_gen) if transform_decode else x_gen
        for i in range(N_SAMPLES_SAVE):
            torchaudio.save(
                f"./wavenet_samples/{model_name_str}-epoch-{epoch}-gen_sample_{i}.wav",
                x_gen[i],
                sample_rate=16000,
                channels_first=False,
                encoding="ULAW",
            )
