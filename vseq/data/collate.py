from typing import List

import torch


def collate_audio(batch: List[torch.Tensor]):
    """Zero pad batch of audio waveforms to maximum temporal length and concatenate"""
    sequence_lengths = [audio.shape[0] for audio in batch]

    T = max(sequence_lengths)
    N = len(batch)

    padded_batch = torch.zeros((N, T), dtype=torch.float32)
    for i, seq_len in enumerate(sequence_lengths):
        padded_batch[i, :seq_len] = batch[i]

    return padded_batch, sequence_lengths


def collate_spectrogram(batch: List[torch.Tensor]):
    """Zero pad batch of spectrograms (F, T) to maximum temporal length and concatenate"""
    sequence_lengths = [spectrogram.shape[1] for spectrogram in batch]

    T = max(sequence_lengths)
    N = len(batch)
    F = batch[0].shape[0]

    padded_batch = torch.zeros((N, F, T), dtype=torch.float32)
    for i, seq_len in enumerate(sequence_lengths):
        padded_batch[i, :, :seq_len] = batch[i]

    return padded_batch, sequence_lengths


def collate_text(batch: List[int], pad_value=-1):
    """Pad batch of int (encoded text) to maximum temporal length and return LongTensors"""
    sequence_lengths = [len(text) for text in batch]

    T = max(sequence_lengths)

    padded_batch = []
    for t, text in zip(sequence_lengths, batch):
        padded_batch.append(text + [-1] * (T - t))

    return torch.LongTensor(padded_batch), torch.LongTensor(sequence_lengths)
