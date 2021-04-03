import os

from typing import List, Tuple, Any
from vseq.data.load import MetaData, load_text

from torch import Tensor
from torch.utils.data import Dataset

from .batcher import Batcher
from .transforms import Transform
from .load import EXTENSIONS_TO_LOADFCN
from .datapaths import DATAPATHS_MAPPING


class BaseDataset(Dataset):
    def __init__(self, source: str, modalities: List[Tuple[str, Batcher, Transform]], sort: bool = True):
        """Dataset class that serves examples from files listed in a `source` as defined by `modalities`.

        `modalities` defines how to obtain an example from a specific file extension via a `Transform` and a `Batcher`.

        Args:
            source (str): Dataset shorthand name or path to source file
            modalities (List[Tuple[str, Batcher, Transform]]): File extensions, batcher and transforms
            sort (bool, optional): If True, sort the first modality according to its batcher. Defaults to True.
        """        
        super().__init__()
        self.source = source
        self.extensions, self.transforms, self.batchers = zip(*modalities)
        self.sort = sort

        self.num_modalities = len(modalities)

        self.source_filepath = DATAPATHS_MAPPING[source] if source in DATAPATHS_MAPPING else source
        self.unique_extensions = set(self.extensions)
        self.examples = self.load_examples(self.source_filepath)

    @staticmethod
    def load_examples(source_filepath):
        with open(source_filepath, "r") as source_file_buffer:
            lines = source_file_buffer.readlines()
        examples = [l.split(',')[0].strip() for l in lines]
        return examples

    def __getitem__(self, idx):
        example_path = self.examples[idx]

        # Load data for each modality
        input_data = dict()
        unique_metadata = dict()
        for ext in self.unique_extensions:
            input_data[ext], unique_metadata[ext] = EXTENSIONS_TO_LOADFCN[ext](example_path + f".{ext}")

        # Transform modalities according to transforms
        data = []
        metadata = []
        for ext, transform in zip(self.extensions, self.transforms):
            x = input_data[ext]
            y = transform(x) if transform else x

            data.append(y)
            metadata.append(unique_metadata[ext])

        if len(data) == 1:
            return data[0], metadata[0]
        return tuple(data), tuple(metadata)

    def collate(self, batch: List[Tuple[Any, Any]]):
        """Arrange a list of outputs from `__getitem__` into a batch via the batcher of each transform"""
        if self.sort:
            sort_modality_idx = 0 if self.num_modalities > 1 else None
            batch = self.batchers[0].sort(batch, sort_modality_idx=sort_modality_idx)

        data, metadata = zip(*batch)
        if self.num_modalities == 1:
            return self.batchers[0](data), metadata

        data = zip(*data)  # [[audio] * batch_size, [text] * batch_size]
        metadata = list(zip(*metadata))

        outputs = []
        for batcher, modality_data in zip(self.batchers, data):
            o = batcher(modality_data)
            outputs.append(o)

        return outputs, metadata

    def __len__(self):
        return len(self.examples)
