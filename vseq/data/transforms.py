import math
import random

from typing import Callable, Optional

import torch
import torch.nn as nn
import numpy as np
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from vseq.data.tokens import LIBRI_PHONESET_SPECIAL, UNKNOWN_TOKEN


class Transform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError()


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class TextCleaner(Transform):
    def __init__(self, cleaner_fcn: Callable):
        super().__init__()
        self.cleaner_fcn = cleaner_fcn

    def forward(self, x):
        return self.cleaner_fcn(x)


class EncodeInteger(Transform):
    def __init__(self, tokenizer, token_map):
        super().__init__()
        self.tokenizer = tokenizer
        self.token_map = token_map

    def forward(self, x: str):
        x = self.tokenizer(x)
        x = self.token_map.encode(x)
        return x

class EncodeIntegerAlignment(Transform):
    def __init__(self, token_map, strip_accent=True, receptive_field=400, stride=320, sample_rate=16000):
        super().__init__()
        self.token_map = token_map
        self.strip_accent = strip_accent
        self.receptive_field = receptive_field
        self.stride = stride
        self.sample_rate = sample_rate

    def forward(self, x: str):
        N = len(x)
        r, s = self.receptive_field, self.stride
        label, mask, new_align, total, new_end = [], [], [], 0, 0
        for idx, interval in enumerate(x):
            end = round(interval.maxTime * self.sample_rate)
            if idx + 1 == N:
                steps = int((end - r) // s + 1) - total
            else:
                steps = round((end - r) / s + 1) - total
            total += steps
            mark = interval.mark.strip("012") if self.strip_accent else interval.mark
            if mark in self.token_map.token2index:
                label += [self.token_map.token2index[mark]] * steps
                mask += [True] * steps
                new_end += steps
            else:
                mask += [False] * steps
        
        return (torch.LongTensor(label), torch.BoolTensor(mask))

class DecodeInteger(Transform):
    def __init__(self):
        super().__init__()
        raise NotImplementedError()


class RandomSegment(Transform):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def forward(self, x):
        high = max(x.size(0) - self.length, 1)
        start_idx = torch.randint(low=0, high=high, size=(1,))
        return x[start_idx : start_idx + self.length]


class Scale(Transform):
    def __init__(self, low=-1, high=1, min_val=None, max_val=None):
        """Scale an input to be in [low, high] by normalizing with data min and max values"""
        super().__init__()
        assert (low is not None) == (high is not None), "must set both low and high or neither"
        self.low = low
        self.high = high
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        x_min = self.min_val if self.min_val is not None else x.min()
        x_max = self.max_val if self.max_val is not None else x.max()

        x_scaled = (x - x_min) / (x_max - x_min)

        if self.low == 0 and self.high == 1:
            return x_scaled

        return self.low + x_scaled * (self.high - self.low)


class MuLawEncode(Transform):
    def __init__(self, bits: int = 8):
        """Encode PCM audio via µ-law companding to some number of bits (8 by default)"""
        super().__init__()
        self.bits = bits
        self.mu = 2 ** bits - 1
        self._divisor = math.log(self.mu + 1)

    def forward(self, x: torch.Tensor):
        return x.sign() * torch.log(1 + self.mu * x.abs()) / self._divisor

class LogMelSpectrogram(Transform):

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        n_mels: int = 80,
        normalize_frq_bins: bool = True
    ) -> None:
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.normalize_frq_bins = normalize_frq_bins

        self.MelSpectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: specgram_mel_db of size (..., ``n_mels``, time).
        """
        mel_specgram = self.MelSpectrogram(waveform)
        logmel_specgram = 10 * torch.log10(torch.clamp_min(mel_specgram, 1e-10))
        
        if self.normalize_frq_bins:
            logmel_specgram -= torch.mean(logmel_specgram, -1, keepdim=True)
            logmel_specgram /= torch.std(logmel_specgram, -1, keepdim=True) + 1e-10

        return logmel_specgram


class AstroSpeech(Transform):
    def __init__(
        self,
        num_tokens,
        whitespace_idx,
        sample_rate=8000,
        duration=500,
        fade=150,
        min_mel=400,
        mel_delta=60,
        speaker_shift=0,
        token_shift=0,
        volume_range=None
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.whitespace_idx = whitespace_idx
        self.sample_rate = sample_rate
        self.duration = duration
        self.fade = fade
        self.min_mel = min_mel
        self.mel_delta = mel_delta
        self.speaker_shift = speaker_shift
        self.token_shift = token_shift
        self.volume_range = volume_range

        # base char frequencies (in numpy because torch can be altered in-place)
        self.mels = (np.arange(num_tokens - 1.0).astype(np.float32) * mel_delta) + min_mel

        # cosine filter
        end = 1/2 * (1 + torch.cos(torch.arange(fade) / (fade - 1) * math.pi))
        start = torch.flip(end, dims=[0])
        middle = torch.ones(duration - (2 * fade))
        self.cosine_filter = torch.cat([start, middle, end])
    
    def mel_to_frq(self, m):
        return (np.exp(m / 1127) - 1) * 700

    def forward(self, x: list):
        
        w = torch.zeros(self.duration)
        signal = []
        for i in x:
            if i == self.whitespace_idx:
                signal.append(w)
            else:
                j = i if i < self.whitespace_idx else i - 1
                m = self.mels[j]
                if self.token_shift > 0:
                    m = m + random.uniform(-self.token_shift, self.token_shift)
                f = self.mel_to_frq(m)
                # TODO: Add frq shifts and volume pertubation to f if specified 
                r = torch.arange(0, self.duration) * (2 * math.pi) / self.sample_rate * f
                c = torch.sin(r) * self.cosine_filter
                signal.append(c)
        
        return torch.cat(signal)



        

class MuLawDecode(Transform):
    def __init__(self, bits: int = 8):
        """Decode PCM audio via µ-law companding from some number of bits (8 by default)"""
        super().__init__()
        self.bits = bits
        self.mu = 2 ** bits - 1
        self._divisor = math.log(self.mu + 1)

    def forward(self, x: torch.Tensor):
        return x.sign() * (torch.exp(x.abs() * self._divisor) - 1) / self.mu


class Quantize(Transform):
    def __init__(
        self,
        low: float = -1.0,
        high: float = 1.0,
        bits: int = 8,
        bins: Optional[int] = None,
        force_out_int64: bool = True,
    ):
        """Quantize a tensor of values between `low` and `high` using a number of `bits`.

        The return value is an integer tensor with values in [0, 2**bits - 1].

        If `bits` is 32 or smaller, the integer tensor is of type `IntTensor` (32 bits).
        If `bits` is 33 or larger, the integer tensor is of type `LongTensor` (64 bits).

        We can force `LongTensor` (64 bit) output if `force_out_int64` is `True`.

        Args:
            low (float, optional): [description]. Defaults to -1.0.
            high (float, optional): [description]. Defaults to 1.0.
            bits (int, optional): [description]. Defaults to 8.
            bins (Optional[int], optional): [description]. Defaults to None.
        """
        super().__init__()
        assert (bits is None) != (bins is None), "Must set one and only one of `bits` and `bins`"
        self.low = low
        self.high = high
        self.bits = bins // 8 if bits is None else bits
        self.bins = 2 ** bits if bins is None else bins
        self.boundaries = torch.linspace(start=-1, end=1, steps=self.bins)
        self.out_int32 = (self.bits <= 32) and (not force_out_int64)

    def forward(self, x: torch.Tensor):
        return torch.bucketize(x, self.boundaries, out_int32=self.out_int32, right=False)


class Binarize(Transform):
    def __init__(self, resample: bool = False, threshold: float = None):
        super().__init__()
        assert bool(threshold) != bool(resample), "Must set exactly one of threshold and resample"
        self.resample = resample
        self.threshold = threshold

    def forward(self, x):
        if self.resample:
            return torch.bernoulli(x)

        return x > self.threshold


class Dequantize(Transform):
    """Dequantize a quantized data point by adding uniform noise.

    Sppecifically, assume the quantized data is x in {0, 1, 2, ..., D} for some D e.g. 255 for int8 data.
    Then, the transformation is given by definition of the dequantized data z as

        z = x + u
        u ~ U(0, 1)

    where u is sampled uniform noise of same shape as x.

    The dequantized data is in the continuous interval [0, D + 1]

    If the value is to scaled subsequently, the maximum value attainable is hence D + 1 due to the uniform noise.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + torch.rand_like(x)
