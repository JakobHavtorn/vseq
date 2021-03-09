

import torchaudio


def load_audio(filepath):
    """Load an audio file returning the audio and sample rate in metadata. Supports same filetypes as torchadio.load"""
    audio, sample_rate = torchaudio.load(filepath)
    metadata = {'sample_rate': sample_rate}
    return audio, metadata


LOAD_FUNCTIONS = {
    'wav': load_audio,
    '.mp3': load_audio,
}


class DataType:
    def __init__(self) -> None:
        self.load = LOAD_FUNCTIONS[self.extension]

    @property
    def extension(self):
        return self._extension

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o)


class Audio(DataType):
    def __init__(self, extension='wav') -> None:
        super().__init__()
        self._extension = extension

    def collate(self, batch):
        pass


class Spectrogram(DataType):
    def __init__(self, extension='.spec'):
        self._extension = extension

    def collate(self, batch: List[torch.Tensor]):
        """Zero pad batch of spectrograms to maximum temporal length and concatenate"""
        sequence_lengths = [spectrogram.shape[1] for spectrogram in batch]

        T_max = max(sequence_lengths)
        N, F = len(batch), batch[0].shape[0]

        padded_batch = torch.zeros((N, F, T_max), dtype=torch.float32)
        for i, seq_len in enumerate(sequence_lengths):
            padded_batch[i, :, :seq_len] = batch[i]

        return padded_batch, sequence_lengths


