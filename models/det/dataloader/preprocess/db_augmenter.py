import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import math
import numpy as np
import cv2 as cv
from typing import List


class DBAugmenter:
    def __init__(self, **kwargs):
        module_list: list = []
        for key, item in kwargs.items():
            module = getattr(iaa, key)
            if module is not None:
                module_list.append(module(**item))
        self.preprocess = None
        if len(module_list) != 0:
            self.preprocess = iaa.Sequential(module_list)
        self.new_size = kwargs['Resize']['size']

    def build(self, data: dict) -> dict:
        image: np.ndarray = data['image']
        shape: tuple = image.shape
        only_resize: bool = not data['is_train']

        if self.preprocess is not None:
            aug = self.preprocess.to_deterministic()
            # Nếu là valid thì chỉ resize
            if not data['is_train']:
                data['image'] = self.__resize(image)
            else:
                data['image'] = aug.augment_image(image)
            data = self.__make_annotation(aug, data, shape, only_resize)
        data.update(shape=data['image'].shape[:2])
        return data

    def __resize(self, image: np.ndarray) -> tuple:
        '''
            Resize ảnh
        '''
        org_h, org_w, _ = image.shape
        new_h: float = self.new_size['height']
        # Nếu giữ tỉ lệ thì đảm bảo cho luôn chia hết cho 32
        # Do trong bước upsampling nếu không chia hết cho 32
        # ảnh sẽ bị lẻ kích thước dẫn đến sai kích thước input
        new_w = org_w / org_h * new_h
        new_w = math.floor(new_w / 32) * 32
        image = cv.resize(image, (new_w, new_h))
        return image

    def __make_annotation(self, aug, data: dict, shape: tuple, only_resize: bool) -> dict:
        '''
            Điều chỉnh tọa độ polygon theo ảnh đã resize
        '''
        if aug is None:
            return data

        target_list: list = []
        for target in data['target']:
            if only_resize:
                new_polygon: List = [(point[0], point[1]) for point in target['polygon']]
            else:
                # Nếu trong quá trình train thì biến đổi
                # tọa độ các đỉnh theo aug
                polygon = np.array(target['polygon'])
                x_min = polygon[:, 0].min()
                x_max = polygon[:, 0].max()
                y_min = polygon[:, 1].min()
                y_max = polygon[:, 1].max()
                bbox = BoundingBoxesOnImage([
                    BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max)
                ], shape=shape)
                new_bbox_list = aug.augment_bounding_boxes(bbox)
                new_bbox_list = new_bbox_list.remove_out_of_image().clip_out_of_image()
                new_bbox_list = new_bbox_list.bounding_boxes
                if len(new_bbox_list) == 0:
                    continue
                new_bbox = new_bbox_list[0]
                key_points = new_bbox.to_keypoints()
                new_polygon: List = [(point.x, point.y) for point in key_points]
            label: str = target['label']
            target_list.append({
                'label': label,
                'polygon': new_polygon,
                'ignore': False
            })
        data['annotation'] = target_list
        return data
