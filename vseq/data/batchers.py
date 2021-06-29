from typing import List, Tuple, Any, Optional

import torch


def get_modulo_padding(length: int, module: int):
    """Return the size of padding to add to `length` to make it evenly divisible by `modulo`"""
    return (module - length % module) % module


def get_modulo_length(length: int, module: int):
    """Return the smallest number larger than `length` evenly divisible by `modulo`"""
    return length + get_modulo_padding(length, module)


class Batcher:
    """Base class for Batchers. These must define `collate` and optionally `sort` methods."""

    def __init__(self) -> None:
        pass

    def __call__(self, batch: List[torch.Tensor]):
        return self.collate(batch)

    def collate(self, batch: List[torch.Tensor]):
        """Convert a list of Tensors into a single Tensor.

        Args:
            batch (List[torch.Tensor]): Batch of Tensors to convert.
        """
        raise NotImplementedError()

    def sort(
        self, batch: List[Tuple[Any, Any]], sort_modality_idx: Optional[int] = None
    ):
        """Sort the order of examples within the batch optionally specifying which modality to sort if more than one.

        Args:
            batch (List[Tuple[Any, Any]]): The batch to sort, generally as a list of tuples of data and metadata.
            sort_modality_idx (bool, optional): Index of the modality to sort. Defaults to None.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()


class TensorBatcher(Batcher):
    def __init__(self):
        """Generic concatenating batcher for equally sized tensors of arbitrary dimensions"""
        super().__init__()

    def collate(self, batch: List[torch.Tensor]):
        """Concatenate a number of equally sized tensors (B, D1, D2, D3, ...)"""
        sequence_lengths = [tensor.numel() for tensor in batch]
        shapes = [tensor.shape for tensor in batch]

        assert all(sequence_lengths[0] == seq_len for seq_len in sequence_lengths)
        assert all(shapes[0] == shape for shape in shapes)

        collated_batch = torch.cat(batch, dim=0)

        return collated_batch, torch.LongTensor(sequence_lengths)


class AudioBatcher(Batcher):
    def __init__(self) -> None:
        super().__init__()

    def collate(self, batch: List[torch.Tensor]):
        """Zero pad batch of audio waveforms to maximum temporal length and concatenate"""
        sequence_lengths = [audio.shape[0] for audio in batch]

        T = max(sequence_lengths)
        N = len(batch)

        collated_batch = torch.zeros((N, T), dtype=batch[0].dtype)
        for i, seq_len in enumerate(sequence_lengths):
            collated_batch[i, :seq_len] = batch[i]
        return collated_batch, torch.LongTensor(sequence_lengths)

    def sort(self, batch: List[torch.Tensor], sort_modality_idx: Optional[int] = None):
        if sort_modality_idx is not None:
            sort_key = lambda x: len(x[0][sort_modality_idx])
        else:
            sort_key = lambda x: len(x[0])
        return sorted(batch, key=sort_key, reverse=True)


class SpectrogramBatcher(Batcher):
    def __init__(self, min_length: int = None) -> None:
        self.min_length = min_length
        super().__init__()

    def collate(self, batch: List[torch.Tensor]):
        """Zero pad batch of spectrograms (T, F) to maximum temporal length and concatenate"""
        sequence_lengths = [sample.shape[0] for sample in batch]

        T = max(sequence_lengths)
        if self.min_length is not None:
            T = max(self.min_length, T)
        N = len(batch)
        F = batch[0].shape[1]

        collated_batch = torch.zeros((N, T, F), dtype=batch[0].dtype)
        for i, seq_len in enumerate(sequence_lengths):
            collated_batch[i, T-seq_len:, :] = batch[i] # right side padding for WaveNet hardcoded
            # if self.min_length is not None and self.min_length > seq_len: # this is clearly wrong and bad
            #     sequence_lengths[i] = self.min_length 
        

        return collated_batch, torch.LongTensor(sequence_lengths)


class TextBatcher(Batcher):
    def __init__(
        self, pad_value: int = 0, min_sample_length: Optional[int] = None
    ) -> None:
        self.pad_value = pad_value
        self.min_sample_length = min_sample_length

    def collate(self, batch: List[torch.Tensor]):
        """Pad batch of int (encoded text) to maximum temporal length and return LongTensors"""
        sequence_lengths = [len(text) for text in batch]

        T = max(sequence_lengths)
        if self.min_sample_length is not None:
            T = max(T, self.min_sample_length)

        collated_batch = [
            text + [self.pad_value] * (T - t)
            for t, text in zip(sequence_lengths, batch)
        ]

        return torch.LongTensor(collated_batch), torch.LongTensor(sequence_lengths)

    def sort(self, batch: List[torch.Tensor], sort_modality_idx: Optional[int] = None):
        if sort_modality_idx is not None:
            sort_key = lambda x: len(x[0][sort_modality_idx])
        else:
            sort_key = lambda x: len(x[0])

        return sorted(batch, key=sort_key, reverse=True)
