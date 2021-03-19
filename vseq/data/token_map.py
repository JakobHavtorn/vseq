from typing import Iterable, List


class TokenMap:
    def __init__(self, tokens: List) -> None:
        self.tokens = tokens
        self.token2index = {t: i for i, t in enumerate(tokens)}
        self.index2token = {i: t for i, t in enumerate(tokens)}

    def encode(self, tokens: Iterable[str]):
        return [self.token2index[t] for t in tokens]

    def decode(self, indices: Iterable[int]):
        return [self.index2token[i] for i in indices]

    def __len__(self):
        return len(self.tokens)
