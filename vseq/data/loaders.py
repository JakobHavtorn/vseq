import csv
import datetime
import os
import re
import uuid

from dataclasses import dataclass
from typing import Callable, Union
from vseq.data.datapaths import DATASETS, TIMIT

import numpy as np
import torch
import torchaudio


def memoize(func: Callable):
    cache = dict()

    def memoized_func(example_id):
        if example_id in cache:
            return cache[example_id]
        result = func(example_id)
        cache[example_id] = result
        return result

    memoized_func.memory = cache
    return memoized_func


@dataclass
class MetaData:
    length: int
    file_path: str


@dataclass
class AudioMetaData(MetaData):
    sample_rate: int
    channels: int
    bits_per_sample: int
    encoding: str
    file_path: str
    example_id: str = None


@dataclass
class TextMetaData(MetaData):
    file_path: str
    word_length: int = None
    char_length: int = None
    example_id: str = None
    line_idx: int = None


def load_text(file_path: str):
    with open(file_path, "r") as text_file:
        text = text_file.read()

    metadata = TextMetaData(
        length=len(text),
        char_length=len(text),
        word_length=len(text.split()),
        file_path=file_path
    )
    return text, metadata


def load_audio(file_path: str, sum_channels: bool = False):
    metadata = torchaudio.info(file_path)
    audio, _ = torchaudio.load(file_path)

    if sum_channels:
        audio = audio.sum(axis=0)

    metadata = AudioMetaData(
        sample_rate=metadata.sample_rate,
        channels=metadata.num_channels,
        bits_per_sample=metadata.bits_per_sample,
        encoding=metadata.encoding,
        length=metadata.num_frames,
        file_path=file_path,
    )
    return audio, metadata


def load_numpy(file_path: str, length_dim: int = 0, **kwargs):
    tensor = torch.from_numpy(np.load(file_path, **kwargs))
    metadata = MetaData(
        length=tensor.size(length_dim),
        file_path=file_path,
    )
    return tensor, metadata


def load_timit_alignment(file_path: str):
    """Method to load alignment metadata for TIMIT from .PHN, .WRD or even .TXT files """
    with open(file_path, "r") as text_file:
        reader = csv.reader(text_file, delimiter=" ")
        source_rows = list(reader)

    start_frame = [int(r[0]) for r in source_rows]
    end_frame = [int(r[1]) for r in source_rows]
    token = [r[2] for r in source_rows]

    metadata = TextMetaData(
        length=len(source_rows),
        file_path=file_path
    )
    return (start_frame, end_frame, token), metadata


class Loader():

    def __init__(self, extension: Union[None, str], cache: bool = False):
        """
        Base Loader for any data type.

        Args:
            extension (str): Extension of data files without delimiter.
            cache (bool): Whether to enable caching.
        """
        self.extension = extension
        self.cache = False

        self.suffix = f"{os.extsep}{extension}" if extension is not None else ""
        self.id = str(uuid.uuid4())

        if cache:
            self.enable_cache()

    def enable_cache(self):
        """Enables caching for the loader."""
        if not self.cache:
            self.cache = True  
            self.load = memoize(self.load)
    
    def __call__(self, example_id):
        """Calls the potentially memoized load method."""
        return self.load(example_id)

    def load(self, example_id):
        raise NotImplementedError

    def __repr__(self):
        name = self.__class__.__name__
        extension = self.extension
        cache = self.cache
        return f"{name}({extension=}, {cache=}, id={self.id})"


class AudioLoader(Loader):

    def __init__(self, extension: Union[None, str], cache: bool = False, sum_channels: bool = True):
        """
        Loader for audio data.

        Args:
            extension (str): Extension of data files (e.g., "wav" or "flac").
            cache (bool): Whether to enable caching.
        """
        super().__init__(extension=extension, cache=cache)
        self.sum_channels = sum_channels

    def load(self, example_id):
        """Load a single audio file."""
        file_path = example_id + self.suffix
        audio, metadata = load_audio(file_path, self.sum_channels)
        metadata.example_id = example_id
        return audio, metadata


class TextLoader(Loader):

    def __init__(self, extension: Union[None, str], cache: bool = False):
        """
        Loader for text data.

        Args:
            extension (str): Extension of data files (e.g., "txt").
            cache (bool): Whether to enable caching.
        """
        super().__init__(extension=extension, cache=cache)

    def load(self, example_id):
        """Load a single text file"""
        file_path = example_id + self.suffix
        text, metadata = load_text(file_path)
        metadata.example_id = example_id
        return text, metadata

    def load_and_cache_batch(self, batch_id):
        """Load a text file with multiple examples and cache them."""

        assert self.cache, "Caching not enabled for loader."

        file_path = batch_id + self.suffix
        with open(file_path, "r") as text_file:
            strings = text_file.read().splitlines()

        batch_data = {}
        for idx, string in enumerate(strings):
            example_id = f"{batch_id}-{idx}"
            metadata = TextMetaData(
                length=len(string),
                char_length=len(string),
                word_length=len(string.split()),
                example_id=example_id,
                file_path=file_path,
                line_idx=idx
            )
            batch_data[example_id] = (string, metadata)

        self.load.memory.update(batch_data)


class NumpyLoader(Loader):

    def __init__(self, extension: Union[None, str], cache: bool = False, length_dim: int = 0, **kwargs):
        """
        Loader for numpy data.

        Args:
            extension (str): Extension of data files (e.g., "wav" or "flac").
            cache (bool): Whether to enable caching.
        """
        super().__init__(extension=extension, cache=cache)
        self.length_dim = length_dim
        self.kwargs = kwargs

    def load(self, example_id):
        """Load a single audio file."""
        file_path = example_id + self.suffix
        tensor, metadata = load_numpy(file_path, length_dim=self.length_dim, **self.kwargs)
        metadata.example_id = example_id
        return tensor, metadata


class TIMITAlignmentLoader(Loader):
    def __init__(self, extension: Union[None, str] = "PHN", cache: bool = True):
        super().__init__(extension=extension, cache=cache)
        if extension not in ["PHN", "WRD"]:
            raise ValueError(f"Invalid extension for TIMIT alignment data {extension}. Should be one of ['PHN', 'WRD']")

    def load(self, example_id):
        file_path = example_id + self.suffix
        (start_frame, end_frame, token), metadata = load_timit_alignment(file_path)
        metadata.example_id = example_id
        return (start_frame, end_frame, token), metadata


@dataclass
class TIMITSpeakerMetadata:
    speaker_id: str
    sex: str
    dialect: str
    use: str
    recorded: datetime.datetime
    birthday: datetime.datetime
    height: datetime.datetime
    race: str
    education: str
    comments: str


def timit_load_speaker_info(spkrinfo_path: str):
    # h_inch += h_ft * 12
    # h_cm = round(h_inch * 2.54, 1)

    dialects = {
        "1": "New England",
        "2": "Northern",
        "3": "North Midland",
        "4": "South Midland",
        "5": "Southern",
        "6": "New York City",
        "7": "Western",
        "8": "Army Brat",
    }

    with open(spkrinfo_path, "r") as buffer:
        text = buffer.readlines()

    text = [re.sub(r"\s+", " ", t) for t in text if t[0] != ";"]  # Remove comment lines
    reader = csv.reader(text, delimiter=" ")
    source_rows = list(reader)

    speaker_metadata = dict()
    for row in source_rows:
        speaker_metadata[row[0]] = TIMITSpeakerMetadata(
            speaker_id=row[0],
            sex=row[1],
            dialect=dialects[row[2]],
            use=row[3],
            recorded=datetime.datetime.strptime(row[4], '%m/%d/%y') if "?" not in row[4] else None,
            birthday=datetime.datetime.strptime(row[5], '%m/%d/%y') if "?" not in row[5] else None,
            height=row[6],
            race=row[7],
            education=row[8],
            comments=row[9],
        )
    return speaker_metadata


class TIMITSpeakerLoader(Loader):
    def __init__(self, spkrinfo_path: str = DATASETS[TIMIT].speaker_info):
        super().__init__(extension=None, cache=False)
        self.spkrinfo_path = spkrinfo_path
        self.speaker_info = timit_load_speaker_info(spkrinfo_path)

    def load(self, example_id):
        speaker_id = example_id.split("/")[-2][1:]  # Remove gender indicator
        return self.speaker_info[speaker_id], None
