import numpy as np
import cv2 as cv
from shapely.geometry import Polygon
import pyclipper


class DBProbLabel:
    def __init__(self, **kwargs):
        self.min_text_size = kwargs['min_text_size']
        self.shrink_ratio = kwargs['shrink_ratio']

    def build(self, data: dict) -> dict:
        image: np.ndarray = data['image']
        polygon_list: list = data['polygon']
        ignore_list: np.ndarray = data['ignore']
        h, w, _ = image.shape

        h, w = image.shape[:2]
        if data['is_train']:
            polygon_list, ignore_list = self.__valid(polygon_list, ignore_list)
        prob_map: np.ndarray = np.zeros((1, h, w), dtype=np.float32)
        prob_mask: np.ndarray = np.ones((h, w), dtype=np.float32)

        for i in range(len(polygon_list)):
            polygon: np.ndarray = np.array(polygon_list[i])
            ph = max(polygon[:, 1]) - min(polygon[:, 1])
            pw = max(polygon[:, 0]) - min(polygon[:, 0])

            if ignore_list[i] or min(pw, ph) < self.min_text_size:
                cv.fillPoly(prob_mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore_list[i] = True
            else:
                polygon_shape = Polygon(polygon)
                dist = polygon_shape.area * \
                    (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject:list = [tuple(point) for point in polygon]
                shrinking = pyclipper.PyclipperOffset()
                shrinking.AddPath(subject, pyclipper.JT_ROUND,
                                  pyclipper.ET_CLOSEDPOLYGON)
                shrink_polygon = shrinking.Execute(-abs(dist))

                if len(shrink_polygon) == 0:
                    cv.fillPoly(prob_mask, polygon.astype(
                        np.int32)[np.newaxis, :, :], 0)
                    ignore_list[i] = True
                    continue

                shrink_polygon = np.array(shrink_polygon[0]).reshape(-1, 2)
                cv.fillPoly(prob_map[0], [shrink_polygon.astype(np.int32)], 1)

        data.update(prob_map=prob_map, prob_mask=prob_mask)
        return data

    def __valid(self, polygon_list: list, ignore_list: np.ndarray) -> tuple:
        '''
            Input:
                - polygon_list: Danh sách các polygon trong ảnh
                - ignore_list : Danh sách quy định polygon bị ignore
            Ouput:
                - polygon_list: Danh sách các polygon được chuẩn hóa
                - ignore: Danh sách quy định polygon bị ignore 
        '''
        if len(polygon_list) == 0:
            return polygon_list, ignore_list
        assert len(ignore_list) == len(polygon_list)
        for i in range(len(polygon_list)):
            area = self.__polygon_area(polygon_list[i])
            if abs(area) < 1:
                ignore_list[i] = True
            if area > 0:
                polygon_list[i] = polygon_list[i][::-1, :]
        return polygon_list, ignore_list

    def __polygon_area(self, polygon: np.ndarray) -> float:
        '''
            Input:
                - polygon: Danh sách các đỉnh của polygon
            Ouput:
                - Diện tích của polygon chưa abs
        '''
        area = 0
        for i in range(polygon.shape[0]):
            i_n = (i+1) % polygon.shape[0]
            area += (polygon[i_n, 0]*polygon[i, 1] -
                     polygon[i_n, 1]*polygon[i, 0])
        return area / 2.
