import os

from typing import List, Tuple, Callable, Any
from vseq.data.load import MetaData, load_text

from torch import Tensor
from torch.utils.data import Dataset

from .transforms import Transform
from .load import extensions_to_load
from .datapaths import DATAPATHS_MAPPING


class BaseDataset(Dataset):
    def __init__(self, source: str, modalities: List[Tuple[str, Callable, Transform]], sort: bool = True):
        super().__init__()
        self.source = source
        self.extensions, self.transforms, self.collaters = zip(*modalities)
        self.sort = sort

        self.num_modalities = len(modalities)

        self.source_filepath = DATAPATHS_MAPPING[source] if source in DATAPATHS_MAPPING else source
        self.unique_extensions = set(self.extensions)
        self.examples = self.load_examples(self.source_filepath)

    @staticmethod
    def load_examples(source_filepath):
        with open(source_filepath, "r") as source_file_buffer:
            lines = source_file_buffer.readlines()
        examples = [l.split(',')[0] for l in lines]
        return examples

    def __getitem__(self, idx):
        example_path = self.examples[idx]

        input_data = dict()
        unique_metadata = dict()
        for ext in self.unique_extensions:
            input_data[ext], unique_metadata[ext] = extensions_to_load[ext](example_path + f".{ext}")

        data = []
        metadata = []
        for ext, transform in zip(self.extensions, self.transforms):
            x = input_data[ext]
            y = transform(x)

            data.append(y)
            metadata.append(unique_metadata[ext])

        if len(data) == 1:
            return data[0], metadata[0]
        return tuple(data), tuple(metadata)

    def collate(self, batch: List[Tuple[Tuple[Any], Tuple[MetaData]]]):
        """Arrange a list of outputs from `__getitem__` into a batch via the collater function of each transform"""
        if self.sort:
            sort_key = (lambda x: x[1][0].length) if self.num_modalities > 1 else (lambda x: x[1].length)
            batch = sorted(batch, key=sort_key)

        data, metadata = zip(*batch)
        if self.num_modalities == 1:
            return self.collaters[0](data), metadata

        data = zip(*data)  # [[audio] * batch_size, [text] * batch_size]
        metadata = list(zip(*metadata))

        outputs = []
        for collater, modality_data in zip(self.collaters, data):
            o = collater(modality_data)
            outputs.append(o)

        return outputs, metadata

    def __len__(self):
        return len(self.examples)
