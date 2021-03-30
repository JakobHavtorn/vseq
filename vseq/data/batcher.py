from typing import List, Tuple, Any

import torch


class Batcher:
    def __init__(self) -> None:
        pass

    def __call__(self, batch: List[torch.Tensor]):
        return self.collate(batch)

    def collate(self, batch: List[torch.Tensor]):
        raise NotImplementedError()

    def sort(self, batch: List[Tuple[Any, Any]], sort_modality_idx=None):
        raise NotImplementedError()


class AudioBatcher(Batcher):
    def __init__(self) -> None:
        super().__init__()

    def collate(self, batch: List[torch.Tensor]):
        """Zero pad batch of audio waveforms to maximum temporal length and concatenate"""
        sequence_lengths = [audio.shape[0] for audio in batch]

        T = max(sequence_lengths)
        N = len(batch)

        padded_batch = torch.zeros((N, T), dtype=torch.float32)
        for i, seq_len in enumerate(sequence_lengths):
            padded_batch[i, :seq_len] = batch[i]

        return padded_batch, sequence_lengths


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

        return padded_batch, sequence_lengths


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

    def sort(self, batch: List[torch.Tensor], sort_modality_idx=None):
        if sort_modality_idx is not None:
            sort_key = lambda x: len(x[0][sort_modality_idx])
        else:
            sort_key = lambda x: len(x[0])

        return sorted(batch, key=sort_key, reverse=True)
