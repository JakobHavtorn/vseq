import math

from typing import Optional

import torch
import torch.nn as nn

from torchtyping import TensorType


class STFT(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: Optional[torch.Tensor] = None,
        center: bool = True,
        pad_mode:str = "reflect",
        normalized: bool = True,
        eps: float = 1e-6
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = True
        self.return_complex = True
        self.eps = eps

        self.out_features = n_fft // 2 + 1

    def forward(self, audio: TensorType["B", "T"]) -> TensorType["B", 2, "N", "T"]:

        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=self.return_complex,
        )

        r = stft.abs()
        phi = stft.angle()

        r_log = (r + self.eps).log()
        phi_scaled = (phi + math.pi) / (2 * math.pi)

        stft = torch.stack([r_log, phi_scaled], dim=1)

        return stft


class ISTFT(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: Optional[torch.Tensor] = None,
        center: bool = True,
        pad_mode:str = "reflect",
        normalized: bool = True,
        eps: float = 1e-6
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = True
        self.return_complex = True
        self.eps = eps

        self.out_features = n_fft // 2 + 1

    def forward(self, stft: TensorType["B", 2, "N", "T"], lengths) -> TensorType["B", "T"]:

        r_log, phi_scaled = torch.chunk(stft, chunks=2, dim=1)

        r = r_log.exp() - self.eps
        phi = phi_scaled * 2 * math.pi - math.pi
        stft = torch.polar(r, phi)

        audio = torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=self.return_complex,
            length=lengths,
        )

        return audio
