from torch.utils.data import Dataset
from os.path import join
import json
import numpy as np
import cv2 as cv
import models.det.dataloader.preprocess as prep

IMAGE_DIR = "image/"
TARGET_FILE = "target.json"


class DBDataset(Dataset):
    def __init__(self, path: str, preprocess: dict):
        super().__init__()
        self.image_dir: str = join(path, IMAGE_DIR)
        self.target_file: str = join(path, TARGET_FILE)
        self.is_train: bool = 'train' in path
        self.preprocess: list = []
        if preprocess is not None:
            for key, item in preprocess.items():
                cls = getattr(prep, key)
                self.preprocess.append(cls(**item))

        self.image_path: list = []
        self.target: list = []
        self.__load_data()

    def __load_data(self):
        '''
            Tải dữ liệu từ file lên
        '''
        with open(self.target_file, 'r', encoding='utf-8') as file:
            target_list = json.loads(file.readline().strip('\n').strip('\r\t'))
        for target in target_list:
            self.image_path.append(join(self.image_dir, target['image']))
            polygon_list: list = []
            for polygon in target['target']:
                if len(polygon['label'].strip()) != 0:
                    polygon['polygon'] = np.array(polygon['polygon']).reshape(-1, 2)
                    polygon_list.append(polygon)
            self.target.append(polygon_list)

    def __getitem__(self, index: int) -> dict:
        data: dict = {}
        image_path: str = self.image_path[index]
        image: np.ndarray = cv.imread(image_path)
        data['is_train'] = self.is_train
        data['target'] = self.target[index]
        data['image'] = image
        for preprocess in self.preprocess:
            data = preprocess.build(data)
        return data

    def __len__(self) -> int:
        return len(self.image_path)
