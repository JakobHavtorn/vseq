from enum import unique
import csv
import os

from tqdm import tqdm
from typing import List, Tuple, Any
from vseq.data.load import MetaData

from torch import Tensor
from torch.utils.data import Dataset

from .load import Loader
from .transforms import Transform
from .batcher import Batcher
from .datapaths import DATAPATHS_MAPPING

class BaseDataset(Dataset):
    def __init__(self, source: str, modalities: List[Tuple[Loader, Transform, Batcher]], sort: bool = True):
        """Dataset class that serves examples from files listed in a `source` as defined by `modalities`.

        `modalities` defines how to obtain an example from a specific file extension via a `Transform` and a `Batcher`.

        Args:
            source (str): Dataset shorthand name or path to source file
            modalities (List[Tuple[Loader, Batcher, Transform]]): File extensions, batcher and transforms
            sort (bool, optional): If True, sort the first modality according to its batcher. Defaults to True.
        """        
        super().__init__()
        self.source = source
        self.loaders, self.transforms, self.batchers = zip(*modalities)
        self.sort = sort

        self.num_modalities = len(modalities)

        self.source_filepath = DATAPATHS_MAPPING[source] if source in DATAPATHS_MAPPING else source
        self.unique_loaders = set(self.loaders)
        self.examples = self.load_examples(self.source_filepath)
        
    def load_examples(self, source_filepath):
        """Load example_ids from source file"""
        
        with open(source_filepath, newline='') as source_file_buffer:
            reader = csv.DictReader(source_file_buffer)
            _is_batch_dataset = ("n_examples" in reader.fieldnames)
            source_rows = list(reader)
        
        if _is_batch_dataset:
            return self._load_and_cache_batch_dataset(source_rows)
        return [row["filename"] for row in source_rows]
    
    def _load_and_cache_batch_dataset(self, source_rows):
        """Caches data for each loader upfront."""
        
        # load examples
        examples = []
        for row in source_rows:
            examples += [f"{row['filename']}-{idx}" for idx in range(int(row["n_examples"]))]
        
        # cache dataset for each loader
        print("\nLoading and caching data:")
        with tqdm(total=len(source_rows) * len(self.unique_loaders)) as pbar:
            for loader in self.unique_loaders:
                loader.enable_cache()
                for row in source_rows:
                    loader.load_and_cache_batch(row["filename"])
                    pbar.update()
                assert len(examples) == len(loader.load.memory), "Not all examples were cached correctly."

        return examples
            
    def __getitem__(self, idx):
        """Get all modalities of a single example"""
        
        example_id = self.examples[idx]

        # load data
        loader_data = {}
        for loader in self.unique_loaders:
            loader_data[loader.id] = loader(example_id)

        # transform data
        data, metadata = [], []
        for loader, transform in zip(self.loaders, self.transforms):
            x, m = loader_data[loader.id]
            y = transform(x) if transform else x
            data.append(y)
            metadata.append(m)

        # return data
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
