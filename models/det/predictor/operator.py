import numpy as np
import cv2 as cv
import math
import torch
from torch import Tensor
from shapely.geometry import Polygon
import pyclipper
from typing import List, Tuple


def resize(image: np.ndarray, limit: int = 960) -> Tuple:
    org_h, org_w, _ = image.shape
    new_w: int = limit
    new_h: int = int((org_h / org_w) * new_w)
    new_h = math.floor(new_h / 32) * 32
    new_image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)
    return new_image, new_w, new_h


def normalize(image: np.ndarray) -> Tensor:
    mean = [122.67891434, 116.66876762, 104.00698793]
    image = image.astype(np.float64)
    image = (image - mean) / 255.
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    return image.unsqueeze(0)


def expand(box: np.ndarray, ratio: float = 2) -> np.ndarray:
    polygon = Polygon(box)
    dist: float = polygon.area * ratio / polygon.length
    subject: list = [tuple(point) for point in box]
    expand = pyclipper.PyclipperOffset()
    expand.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expand_polygon: np.ndarray = np.array(expand.Execute(dist)[0])
    return expand_polygon


def sort_pointer(point_list: List):
    idx1, idx2, idx3, idx4 = 0, 1, 2, 3
    if point_list[1][1] > point_list[0][1]:
        idx1 = 0
        idx4 = 1
    else:
        idx1 = 1
        idx4 = 0

    if point_list[3][1] > point_list[2][1]:
        idx2 = 2
        idx3 = 3
    else:
        idx2 = 3
        idx3 = 2

    ans: list = [
        point_list[idx1], point_list[idx2],
        point_list[idx3], point_list[idx4]
    ]
    return ans


def box_finding(contour: np.ndarray) -> tuple:
    # Tìm bounding box nhỏ nhất bao quanh contour
    # rồi sắp xếp tăng dần theo trục x: x1 <= x2 < x3 <= x4
    box: tuple = cv.minAreaRect(contour)
    point_list: list = sorted(list(cv.boxPoints(box)), key=lambda x: x[0])

    # Quy định lại vị trí của các điểm
    # theo thứ tự : (((1,2),(4,3))
    ans = sort_pointer(point_list)
    # Trả lại các điểm của bounding box
    # và min(width,height) của bounding box
    return np.array(ans), min(box[1])


def find_key(box: List) -> Tuple:
    new_box = np.array(box)
    x_min = new_box[:, 0].min()
    x_max = new_box[:, 0].max()
    y_min = new_box[:, 1].min()
    y_max = new_box[:, 1].max()
    return x_min, y_min, x_max, y_max
