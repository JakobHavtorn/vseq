from functools import partial

from typing import Iterable, List, Optional

import torch

from torch.utils.data.dataset import IterableDataset
from vseq.data.tokens import BLANK_TOKEN, DELIMITER_TOKEN, START_TOKEN, END_TOKEN, UNKNOWN_TOKEN


def get_with_fallback(dictionary, key):
    return dictionary.get(key, dictionary[UNKNOWN_TOKEN])


def get(dictionary, key):
    return dictionary[key]


class TokenMap:
    def __init__(
        self,
        tokens: List,
        add_start: bool = False,
        add_end: bool = False,
        add_delimit: bool = False,
        add_unknown: bool = False,
        add_blank: bool = False,
    ) -> None:

        assert not (add_delimit and (add_end or add_start)), "Cannot use start or end token with delimiter token."

        self.add_start = add_start
        self.add_end = add_end
        self.add_delimit = add_delimit
        self.add_unknown = add_unknown
        self.add_blank = add_blank
        self.prefix = ""
        self.suffix = ""

        if add_start:
            tokens.append(START_TOKEN)
            self.prefix = START_TOKEN
        if add_end:
            tokens.append(END_TOKEN)
            self.suffix = END_TOKEN
        if add_delimit:
            tokens.append(DELIMITER_TOKEN)
            self.prefix = DELIMITER_TOKEN
            self.suffix = DELIMITER_TOKEN
        if add_unknown:
            tokens.append(UNKNOWN_TOKEN)
        if add_blank:
            tokens.insert(BLANK_TOKEN, 0)  # Blank token always at index 0

        self.tokens = tokens

        self.token2index = {t: i for i, t in enumerate(tokens)}
        self.index2token = {i: t for i, t in enumerate(tokens)}

        self.get_index = partial(get_with_fallback, self.token2index) if add_unknown else partial(get, self.token2index)
        self.get_token = partial(get, self.index2token)

    def encode(self, tokens: Iterable[list]):
        tokens = list(self.prefix) + tokens + list(self.suffix)
        return [self.get_index(t) for t in tokens]

    def decode(self, indices: Iterable[int], join_separator: Optional[str] = None):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        if join_separator is None:
            return [self.index2token[i] for i in indices]
        return join_separator.join([self.index2token[i] for i in indices])

    def decode_batch(
        self, indices_batch: Iterable[Iterable[int]], sl: Iterable[int], join_separator: Optional[str] = None
    ):
        if isinstance(indices_batch, torch.Tensor):
            indices_batch = indices_batch.tolist()
        assert len(indices_batch) == len(sl), "Batch must be first in `indices_batch`"

        batch = []
        N = len(sl)
        for n in range(N):
            indices = indices_batch[n][: sl[n]]
            batch.append(self.decode(indices, join_separator=join_separator))
        return batch

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        token_str = f"tokens={self.tokens}" if len(self.tokens) < 50 else f"|tokens|={len(self.tokens)}"
        return f"TokenMap({token_str})"
