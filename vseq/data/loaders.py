import os
import uuid
from time import time
import logging

LOGGER = logging.getLogger(name=__file__)

from dataclasses import dataclass
from typing import Callable, Union

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
    word_length: int
    char_length: int
    file_path: str
    example_id: str = None
    line_idx: int = None


def load_text(file_path: str):
    with open(file_path, "r") as text_file:
        text = text_file.read()

    metadata = TextMetaData(
        length=len(text),
        char_length=len(text),
        word_length=len(text.split()),
        file_path=file_path,
    )
    return text, metadata


def load_audio(file_path: str, sum_channels: bool = False):
    try:
        metadata = torchaudio.info(file_path)
        audio, _ = torchaudio.load(file_path)
    except RuntimeError as exc:
        if "Error opening audio file" in str(exc):
            raise RuntimeError(f"Error opening audio file: {file_path}")
        else:
            raise exc

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


class Loader:
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
    def __init__(
        self,
        extension: Union[None, str],
        cache: bool = False,
        sum_channels: bool = True,
    ):
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
                line_idx=idx,
            )
            batch_data[example_id] = (string, metadata)

        self.load.memory.update(batch_data)


class NumpyLoader(Loader):
    def __init__(
        self,
        extension: Union[None, str],
        cache: bool = False,
        length_dim: int = 0,
        **kwargs,
    ):
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
        tensor, metadata = load_numpy(
            file_path, length_dim=self.length_dim, **self.kwargs
        )
        metadata.example_id = example_id
        return tensor, metadata
