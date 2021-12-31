from typing import List, Dict
import numpy as np
import json


class Alphabet:
    def __init__(self, path: str) -> None:
        self.pad: int = 0
        with open(path, 'r', encoding='utf-8') as f:
            alphabet = f.readline().strip("\n").strip("\r\t").strip()
            alphabet = ' ' + alphabet
        self.char_dict: Dict = {c: i + 1 for i, c in enumerate(alphabet)}
        self.int_dict: Dict = {i + 1: c for i, c in enumerate(alphabet)}
        self.int_dict[self.pad] = '<pad>'

    def encode(self, s: str) -> np.ndarray:
        es: List = [self.char_dict.get(ch, self.pad) for ch in s]
        return np.asarray(es, dtype=np.int32)

    def decode(self, es: np.ndarray) -> str:
        s = ''.join([self.int_dict[item] for item in es if item != self.pad])
        return s

    def size(self):
        return len(self.int_dict)


class DocumentLabel:
    def __init__(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            label = json.loads(f.readline().strip("\n").strip("\r\t"))
        self.char_dict: Dict = {c: i for i, c in enumerate(label)}
        self.int_dict: Dict = {i: c for i, c in enumerate(label)}

    def encode(self, s: str):
        return self.char_dict[s]

    def decode(self, n: int):
        return self.int_dict[n]

    def size(self):
        return len(self.int_dict)
