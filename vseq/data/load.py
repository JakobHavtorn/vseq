from dataclasses import dataclass
from vseq import data

import torch
import torchaudio


@dataclass
class MetaData():
    length: int
    filepath: str


@dataclass
class AudioMetaData(MetaData):
    sample_rate: int
    channels: int
    bits_per_sample: int
    encoding: str


@dataclass
class TextMetaData(MetaData):
    pass


def load_audio(filepath, sum_channels: bool = True):
    """Load an audio file returning the audio and sample rate in metadata. Supports same filetypes as torchadio.load"""
    metadata = torchaudio.info(filepath)
    audio, _ = torchaudio.load(filepath)

    if sum_channels:
        audio = audio.sum(axis=0)

    metadata = AudioMetaData(
        sample_rate=metadata.sample_rate,
        channels=metadata.num_channels,
        bits_per_sample=metadata.bits_per_sample,
        encoding=metadata.encoding,
        length=metadata.num_frames,
        filepath=filepath
    )
    return audio, metadata


def load_text(filepath):
    """Load a text file and return """
    with open(filepath, 'r') as text_file:
        string = text_file.read()

    metadata = TextMetaData(length=len(string), filepath=filepath)
    return string, metadata


extensions_to_load = dict(
    wav=load_audio,
    flac=load_audio,
    mp3=load_audio,
    txt=load_text,
)