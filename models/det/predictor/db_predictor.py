import numpy as np
import torch
from models.det.structure import DBNetwork
import yaml
from typing import Dict, List
from .operator import resize, normalize, expand, box_finding, find_key
import cv2 as cv


class DBPredictor:
    def __init__(self, config: str, pretrained):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        with open(config) as f:
            data: Dict = yaml.safe_load(f)
        self.model = DBNetwork(**data['structure'], device=self.device)
        print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        state_dict = torch.load(pretrained, map_location=self.device)
        self.model.load_state_dict(state_dict['model'])
        self.limit: int = 1024

    def predict(self, image: np.ndarray) -> List:
        self.model.eval()
        with torch.no_grad():
            org_h, org_w, _ = image.shape
            res_image, new_w, new_h = resize(image, self.limit)
            image = normalize(res_image)
            pred = self.model.predict(image)
            prob_map = pred.cpu().detach().numpy()[0][0]
            bitmap = np.uint8((prob_map * 255.)).astype(np.int32) > 0
            contour_list, _ = cv.findContours(np.uint8(bitmap * 255.),
                                              cv.RETR_LIST,
                                              cv.CHAIN_APPROX_SIMPLE)
            contour_num: int = min(len(contour_list), 1000)
            box_list: List = []
            for i in range(contour_num):
                contour: np.ndarray = contour_list[i]
                box, min_edge = box_finding(contour)
                if min_edge < 3:
                    continue
                box = expand(box).reshape((-1, 1, 2))
                box, min_edge = box_finding(box)
                if min_edge < 5:
                    continue
                # Đảm bảo box trong kích cỡ của ảnh gốc
                box[:, 0] = np.clip(box[:, 0].astype(np.float32) * org_w / new_w, 0, org_w - 1)
                box[:, 1] = np.clip(box[:, 1].astype(np.float32) * org_h / new_h, 0, org_h - 1)
                x1_min, y1_min, x1_max, y1_max = find_key(box.astype(np.int16))
                box_list.append([x1_min, y1_min, x1_max, y1_max])
        return box_list
