from typing import Iterable, List
from vseq.data.tokens import DELIMITER_TOKEN, START_TOKEN, END_TOKEN


class TokenMap:
    def __init__(self, tokens: List) -> None:
        self.tokens = tokens
        self.token2index = {t: i for i, t in enumerate(tokens)}
        self.index2token = {i: t for i, t in enumerate(tokens)}

    def encode(self, tokens: Iterable[list], delimit: bool=False):
        if delimit:
            tokens = self.delimit(tokens)
        return [self.token2index[t] for t in tokens]

    def delimit(self, tokens: Iterable[list]):
        if DELIMITER_TOKEN in self.tokens:
            tokens.insert(0, DELIMITER_TOKEN)
            tokens.insert(-1, DELIMITER_TOKEN)
        elif all(token in self.tokens for token in [START_TOKEN, END_TOKEN]):
            tokens.insert(0, START_TOKEN)
            tokens.insert(-1, END_TOKEN)
        else:
            raise ValueError("The TokenMap doesn't contain the required delimiters tokens.")
        return tokens

    def decode(self, indices: Iterable[int]):
        return [self.index2token[i] for i in indices]

    def __len__(self):
        return len(self.tokens)
