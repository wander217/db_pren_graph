from collections import OrderedDict
import numpy as np


class DBICDAR:
    def __init__(self, shrink_ratio):
        self.shrink_ratio:float = shrink_ratio

    def build(self, data: dict) -> dict:
        polygon: list = []
        ignore: list = []
        annotation: np.ndarray = data['annotation']
        img: np.ndarray = data['image']
        is_train: bool = data['is_train']
        shape: np.ndarray = np.array(data['shape'])

        for target in annotation:
            polygon.append(np.array(target['polygon']))
            ignore.append(target['ignore'])

        return OrderedDict(
            image=img,
            polygon=polygon,
            shape=shape,
            ignore=np.array(ignore, dtype=np.uint8),
            is_train=is_train
        )
