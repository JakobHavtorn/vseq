from typing import Union, List
import numpy as np


import torch
from torch.utils.data import Dataset


class SimpleSinusoidDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 1024,
        sample_length: int = 64000,
        amplitude: float = 0.5,
        frequency: List[float] = [440],
        sample_rate: int = 8000,
        cycle_amplitude: Union[bool, int] = False,
    ):
        super().__init__()
        self.amplitude = amplitude
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.sample_length = sample_length
        self.n_samples = n_samples

        self.source = "gen_sine"

        self.periods = np.array([sample_rate / freq for freq in frequency])

        # Do actual generation here

        self.base_sequence_length = 2 * self.sample_length + int(2 * max(self.periods))

        base_sequence = np.mean(
            [
                np.sin(np.pi * 2 * (np.arange(self.base_sequence_length) / period))
                for period in self.periods
            ],
            axis=0,
        )

        if cycle_amplitude:
            n = int(cycle_amplitude)
            amplitude *= np.abs(
                np.sin(np.linspace(0, n * np.pi * 2, self.base_sequence_length))
            )

        self.base_sequence = torch.FloatTensor(amplitude * base_sequence)

    def __getitem__(self, index):
        # gen sequence skewed by index right or left
        cut_index = index % (int(max(self.periods)) + self.sample_length)
        return (
            self.base_sequence[cut_index : cut_index + self.sample_length],
            self.sample_length,
        )

    def __len__(self):
        return self.n_samples


class SequentialSinusoidDataset(Dataset):
    pass
