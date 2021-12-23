import numpy as np
import cv2 as cv


class DBRandomCrop:
    def __init__(self, size: list, max_tries: int, min_crop: float):
        self.size: list = size
        self.max_tries: int = max_tries
        self.min_crop: float = min_crop

    def build(self, data: dict) -> dict:
        img: np.ndarray = data['image']
        org_annotation: list = data['annotation']
        polygon_list: list = [target['polygon']
                              for target in org_annotation if not target['ignore']]
        crop_x, crop_y, crop_w, crop_h = self.__crop_area(img, polygon_list)

        scale_w: float = self.size[0] / crop_w
        scale_h: float = self.size[1] / crop_h
        scale: float = min(scale_w, scale_h)
        h = int(scale * crop_h)
        w = int(scale * crop_w)

        ex_image: np.ndarray = np.zeros((self.size[1], self.size[0], img.shape[2]), img.dtype)
        ex_image[:h, :w] = cv.resize(img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))

        target_list: list = []
        for target in org_annotation:
            polygon = np.array(target['polygon'])
            if not self.__is_outside(polygon, crop_x, crop_y, crop_w, crop_h):
                new_polygon: list = ((polygon - (crop_x, crop_y)) * scale).tolist()
                target_list.append({**target, 'polygon': new_polygon})
        if len(target_list) != 0:
            data['annotation'] = target_list
            data['image'] = ex_image
        return data

    def __crop_area(self, img: np.ndarray, polygon_list: list) -> tuple:
        '''
            Hàm thực hiện cắt ảnh.
        '''
        h, w, _ = img.shape
        h_axis: np.ndarray = np.zeros(h, dtype=np.int32)
        w_axis: np.ndarray = np.zeros(w, dtype=np.int32)

        for polygon in polygon_list:
            tmp: np.ndarray = np.round(polygon, decimals=0).astype(np.int32)
            w_axis = self.__mask_down(w_axis, tmp, 0)
            h_axis = self.__mask_down(h_axis, tmp, 1)

        h_not_mask: np.ndarray = np.where(h_axis == 0)[0]
        w_not_mask: np.ndarray = np.where(w_axis == 0)[0]
        if len(h_not_mask) == 0 or len(w_not_mask) == 0:
            return 0, 0, w, h
        h_region: list = self.__split_region(h_not_mask)
        w_region: list = self.__split_region(w_not_mask)

        w_min: float = self.min_crop * w
        h_min: float = self.min_crop * h

        for _ in range(self.max_tries):
            x_min, x_max = self.__crop(w_region, w_not_mask, w)
            y_min, y_max = self.__crop(h_region, h_not_mask, h)
            if x_max - x_min + 1 < w_min or y_max - y_min + 1 < h_min:
                continue
            for polygon in polygon_list:
                if not self.__is_outside(polygon, x_min, y_min, x_max - x_min + 1, y_max - y_min + 1):
                    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1
        return 0, 0, w, h

    def __mask_down(self, axis: np.ndarray, polygon: np.ndarray, type: int) -> np.ndarray:
        '''
            Hàm thực hiện đánh dấu vị trí box trong trục
        '''
        p_axis: np.ndarray = polygon[:, type]
        mn: int = np.min(p_axis)
        mx: int = np.max(p_axis)
        axis[mn:mx + 1] = 1
        return axis

    def __split_region(self, axis: np.ndarray) -> list:
        '''
            Chia cắt các vùng theo trục
        '''
        region: list = []
        start_point: int = 0
        if axis.shape[0] == 0:
            return region
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region.append(axis[start_point:i])
                start_point = i
        if start_point < axis.shape[0]:
            region.append(axis[start_point:])
        return region

    def __is_outside(self, polygon: list, x: float, y: float, w: float, h: float) -> bool:
        '''
            Kiểm tra box có chứa polygon không
        '''
        tmp: np.ndarray = np.array(polygon)
        x_min = tmp[:, 0].min()
        x_max = tmp[:, 0].max()
        y_min = tmp[:, 1].min()
        y_max = tmp[:, 1].max()
        if x_min >= x and x_max < x + w and y_min >= y and y_max < y + h:
            return False
        return True

    def __crop(self, region_list: list, axis_not_mask: np.ndarray, size: int) -> tuple:
        '''
            Cắt ngẫu nhiên một đoạn từ trục
        '''
        if len(region_list) > 1:
            id_list: list = list(np.random.choice(len(region_list), size=2))
            value_list: list = []
            for id in id_list:
                region: int = region_list[id]
                x: int = int(np.random.choice(region, 1))
                value_list.append(x)
            x_min = np.clip(min(value_list), 0, size - 1)
            x_max = np.clip(max(value_list), 0, size - 1)
        else:
            x_list: np.ndarray = np.random.choice(axis_not_mask, size=2)
            x_min = np.clip(np.min(x_list), 0, size - 1)
            x_max = np.clip(np.max(x_list), 0, size - 1)
        return x_min, x_max
