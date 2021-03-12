from typing import List

import torch


def collate_audio(batch: List[torch.Tensor]):
    """Zero pad batch of audio waveforms to maximum temporal length and concatenate"""
    sequence_lengths = [audio.shape[0] for audio in batch]

    T_max = max(sequence_lengths)
    N = len(batch)

    padded_batch = torch.zeros((N, T_max), dtype=torch.float32)
    for i, seq_len in enumerate(sequence_lengths):
        padded_batch[i, :seq_len] = batch[i]

    return padded_batch, sequence_lengths


def collate_spectrogram(batch: List[torch.Tensor]):
    """Zero pad batch of spectrograms to maximum temporal length and concatenate"""
    sequence_lengths = [spectrogram.shape[1] for spectrogram in batch]

    T_max = max(sequence_lengths)
    N, F = len(batch), batch[0].shape[0]

    padded_batch = torch.zeros((N, F, T_max), dtype=torch.float32)
    for i, seq_len in enumerate(sequence_lengths):
        padded_batch[i, :, :seq_len] = batch[i]

    return padded_batch, sequence_lengths


def collate_text(batch: List[str]):
    pass
