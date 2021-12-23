import numpy as np
import pyclipper
from shapely.geometry import Polygon
import cv2 as cv


class DBThreshLabel:
    def __init__(self, expand_ratio: float, thresh_min: float, thresh_max: float):
        self.expand_ratio: float = expand_ratio
        self.thresh_min: float = thresh_min
        self.thresh_max: float = thresh_max

    def build(self, data: dict) -> dict:
        image: np.ndarray = data['image']
        polygon_list: list = data['polygon']
        ignore: np.ndarray = data['ignore']

        thresh_map = np.ones(image.shape[:2], dtype=np.float32)
        thresh_mask = np.zeros(image.shape[:2], dtype=np.float32)

        for i in range(len(polygon_list)):
            if not ignore[i]:
                thresh_map, thresh_mask = self.__calc(polygon_list[i], thresh_map, thresh_mask)
        thresh_map = 1 - thresh_map
        thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min
        data.update(thresh_map=thresh_map, thresh_mask=thresh_mask)
        return data

    def __calc(self, polygon: np.ndarray, thresh_map: np.ndarray, thresh_mask: np.ndarray) -> tuple:
        '''
            Hàm tính thresh_map và thresh_mask cho từng polygon
        '''
        polygon_shape = Polygon(polygon)
        dist = polygon_shape.area * (1 - np.power(self.expand_ratio, 2)) / polygon_shape.length
        subject: list = [tuple(point) for point in polygon]
        expand = pyclipper.PyclipperOffset()
        expand.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expand_polygon: np.ndarray = np.array(expand.Execute(abs(dist))[0])
        cv.fillPoly(thresh_mask, [expand_polygon.astype(np.int32)], 1.)

        x_min: int = expand_polygon[:, 0].min()
        x_max: int = expand_polygon[:, 0].max()
        y_min: int = expand_polygon[:, 1].min()
        y_max: int = expand_polygon[:, 1].max()
        w: int = x_max - x_min + 1
        h: int = y_max - y_min + 1

        x_axis: np.ndarray = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        y_axis: np.ndarray = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        dist_map: np.ndarray = np.zeros((polygon.shape[0], h, w), dtype=np.float32)
        for i in range(polygon.shape[0]):
            i_n = (i + 1) % polygon.shape[0]
            abs_dist = self.__calc_dist(x_axis, y_axis, polygon[i], polygon[i_n])
            dist_map[i] = np.clip(abs_dist / dist, 0., 1.)
        dist_map = np.min(dist_map, axis=0)
        x_valid_min: int = min(max(x_min, 0), thresh_map.shape[1] - 1)
        x_valid_max: int = min(max(x_max, 0), thresh_map.shape[1] - 1)
        y_valid_min: int = min(max(y_min, 0), thresh_map.shape[0] - 1)
        y_valid_max: int = min(max(y_max, 0), thresh_map.shape[0] - 1)
        thresh_map[y_valid_min:y_valid_max + 1, x_valid_min:x_valid_max + 1] = np.fmin(
            1 - dist_map[y_valid_min - y_min:y_valid_max - y_max + h,
                x_valid_min - x_min:x_valid_max - x_max + w],
            thresh_map[y_valid_min:y_valid_max + 1, x_valid_min:x_valid_max + 1])
        return thresh_map, thresh_mask

    def __calc_dist(self, x_axis: np.ndarray, y_axis: np.ndarray, p1: list, p2: list) -> np.ndarray:
        '''
            Tính khoảng cách từ mọi điểm trên trục x,y đến đoạn thẳng p1p2
        '''
        sq_dist_1: np.ndarray = np.square(x_axis - p1[0]) + np.square(y_axis - p1[1])
        sq_dist_2: np.ndarray = np.square(x_axis - p2[0]) + np.square(y_axis - p2[1])
        sq_dist_3: np.ndarray = np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1])
        cosin: np.ndarray = (sq_dist_1 + sq_dist_2 - sq_dist_3) / (2. * np.sqrt(sq_dist_1 * sq_dist_2))
        sq_sin: np.ndarray = 1 - np.square(cosin)
        sq_sin = np.nan_to_num(sq_sin)
        dist: np.ndarray = np.sqrt(sq_dist_1 * sq_dist_2 * sq_sin / sq_dist_3)
        dist[cosin > 0] = np.sqrt(np.fmin(sq_dist_1, sq_dist_2))[cosin > 0]
        return dist
