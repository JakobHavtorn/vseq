from typing import List, Tuple, Any, Optional

import torch


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

    def sort(self, batch: List[Tuple[Any, Any]], sort_modality_idx: Optional[int] = None):
        """Sort the order of examples within the batch optionally specifying which modality to sort if more than one.

        Args:
            batch (List[Tuple[Any, Any]]): The batch to sort, generally as a list of tuples of data and metadata.
            sort_modality_idx (bool, optional): Index of the modality to sort. Defaults to None.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()


class AudioBatcher(Batcher):
    def __init__(self) -> None:
        super().__init__()

    def collate(self, batch: List[torch.Tensor]):
        """Zero pad batch of audio waveforms to maximum temporal length and concatenate"""
        sequence_lengths = [audio.shape[0] for audio in batch]

        T = max(sequence_lengths)
        N = len(batch)

        padded_batch = torch.zeros((N, T), dtype=batch[0].dtype)
        for i, seq_len in enumerate(sequence_lengths):
            padded_batch[i, :seq_len] = batch[i]

        return padded_batch, torch.LongTensor(sequence_lengths)

    def sort(self, batch: List[torch.Tensor], sort_modality_idx: Optional[int] = None):
        if sort_modality_idx is not None:
            sort_key = lambda x: len(x[0][sort_modality_idx])
        else:
            sort_key = lambda x: len(x[0])

        return sorted(batch, key=sort_key, reverse=True)

class AlignmentBatcher(Batcher):
    def __init__(self) -> None:
        super().__init__()

    def collate(self, batch: List[torch.Tensor]):
        """Zero pad batch of audio waveforms to maximum temporal length and concatenate"""
        align, mask = zip(*batch)
        
        align_sl= [a.shape[0] for a in align]
        mask_sl = [m.shape[0] for m in mask]

        T_align = max(align_sl)
        T_mask = max(mask_sl)
        N = len(batch)

        padded_batch_align = torch.zeros((N, T_align), dtype=align[0].dtype)
        padded_batch_mask = torch.zeros((N, T_mask), dtype=mask[0].dtype)
        for i, (a_sl, m_sl) in enumerate(list(zip(align_sl, mask_sl))):
            padded_batch_align[i, :a_sl] = align[i]
            padded_batch_mask[i, :m_sl] = mask[i]

        return (padded_batch_align, padded_batch_mask), torch.LongTensor(align_sl)

    def sort(self, batch: List[torch.Tensor], sort_modality_idx: Optional[int] = None):
        if sort_modality_idx is not None:
            sort_key = lambda x: len(x[0][sort_modality_idx])
        else:
            sort_key = lambda x: len(x[0])

        return sorted(batch, key=sort_key, reverse=True)


class SpectrogramBatcher(Batcher):
    def __init__(self) -> None:
        super().__init__()

    def collate(self, batch: List[torch.Tensor]):
        """Zero pad batch of spectrograms (F, T) to maximum temporal length and concatenate"""
        sequence_lengths = [spectrogram.shape[1] for spectrogram in batch]

        T = max(sequence_lengths)
        N = len(batch)
        F = batch[0].shape[0]

        padded_batch = torch.zeros((N, F, T), dtype=torch.float32)
        for i, seq_len in enumerate(sequence_lengths):
            padded_batch[i, :, :seq_len] = batch[i]

        return padded_batch, torch.LongTensor(sequence_lengths)
    
    def sort(self, batch: List[torch.Tensor], sort_modality_idx: Optional[int] = None):
        if sort_modality_idx is not None:
            sort_key = lambda x: x[0][sort_modality_idx].shape[1]
        else:
            sort_key = lambda x: x[0].shape[1]

        return sorted(batch, key=sort_key, reverse=True)


class TextBatcher(Batcher):
    def __init__(self, pad_value: int = 0) -> None:
        self.pad_value = pad_value

    def collate(self, batch: List[torch.Tensor]):
        """Pad batch of int (encoded text) to maximum temporal length and return LongTensors"""
        sequence_lengths = [len(text) for text in batch]

        T = max(sequence_lengths)

        padded_batch = []
        for t, text in zip(sequence_lengths, batch):
            padded_batch.append(text + [self.pad_value] * (T - t))

        return torch.LongTensor(padded_batch), torch.LongTensor(sequence_lengths)

    def sort(self, batch: List[torch.Tensor], sort_modality_idx: Optional[int] = None):
        if sort_modality_idx is not None:
            sort_key = lambda x: len(x[0][sort_modality_idx])
        else:
            sort_key = lambda x: len(x[0])

        return sorted(batch, key=sort_key, reverse=True)
