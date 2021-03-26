from typing import Iterable, List
from vseq.data.tokens import DELIMITER_TOKEN, START_TOKEN, END_TOKEN


class TokenMap:
    def __init__(self, tokens: List) -> None:
        self.tokens = tokens
        self.token2index = {t: i for i, t in enumerate(tokens)}
        self.index2token = {i: t for i, t in enumerate(tokens)}

    def encode(self, tokens: Iterable[list], prefix: str="", suffix: str=""):
        if prefix or suffix:
            tokens = list(prefix) + tokens + list(suffix)
        return [self.token2index[t] for t in tokens]

    def decode(self, indices: Iterable[int], separator: str = ""):
        return separator.join([self.index2token[i] for i in indices])

    def decode_batch(self, indices_batch, sl, separator: str = ""):
        batch = []
        N = len(sl)
        for n in range(N):
            indices = indices_batch[n][:sl[n]]
            batch.append(self.decode(indices, separator=separator))
        return batch


    def __len__(self):
        return len(self.tokens)
