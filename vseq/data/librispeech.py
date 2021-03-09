from typing import List, Dict

import torchaudio

from .base_dataset import BaseDataset
from .transforms import Transform
from .data_types import Audio


class Librispeech(BaseDataset):
    def __init__(self, transforms: List[Dict[DataType, Transform]]):
        self.transforms = transforms

        self.data_types = set([dt for dt in self.transforms.keys()])

    def __getitem__(self, idx):
        input_data = dict()
        for data_type in self.data_types:
            input_data[data_type] =  data_type.load(self.examples[idx])

        outputs = []
        for data_type, transform in self.transforms.items():
            x = input_data[data_type]
            outputs += [transform(x)]
        
        return tuple(outputs), self.examples[idx]

    def collate(self, batch Tuple[torch.Tensor]):
        for data_type, transform in self.transforms.items():
            pass


Librispeech(
    transforms=[
        dict(Audio('.wav')=torchaudio.transforms.MelSpectrogram())
    ]
)

