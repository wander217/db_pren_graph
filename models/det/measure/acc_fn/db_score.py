import cv2 as cv
import numpy as np
from copy import deepcopy
from shapely.geometry import Polygon
import pyclipper
from torch import Tensor


class DBScore:
    def __init__(self, box_num: int, edge_thresh: float,label: str):
        self.box_num: int = box_num
        self.edge_thresh: float = edge_thresh
        self.label: str = label

    def calc(self, pred: dict, batch: dict) -> tuple:
        # Phân ngưỡng prob_map
        # đảm bảo những vùng có xác suất
        # xuất hiện của chữ lớn hơn thresh
        prob_map_list: Tensor = pred[self.label]
        seg_map_list: Tensor = (prob_map_list * 255.).int() > 0

        # Khởi tạo mảng lưu trữ bounding box và score
        box_list: list = []
        score_list: list = []
        batch_size: int = batch['image'].size(0)
        for i in range(batch_size):
            box, score = self.__finding(
                prob_map_list[i],
                seg_map_list[i],
                batch['shape'][i]
            )
            box_list.append(box)
            score_list.append(score)
        return box_list, score_list

    def __finding(self, prob_map: Tensor, seg_map: Tensor, dest_size: Tensor) -> tuple:
        new_prob_map: np.ndarray = prob_map.cpu().detach().numpy()[0]
        new_seg_map: np.ndarray = seg_map.cpu().detach().numpy()[0]
        org_size: tuple = new_prob_map.shape
        new_dest_size: np.ndarray = dest_size.cpu().detach().numpy()

        # Tìm contour bao quanh phần xác định có chữ
        contour_list, _ = cv.findContours(
            (new_seg_map * 255.).astype(np.uint8),
            cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
        )

        # lưu trữ tọa độ và điểm của contour thỏa mãn
        countour_num: int = min(len(contour_list), self.box_num)
        box_list: np.ndarray = np.zeros((countour_num, 4, 2), dtype=np.int16)

        for i in range(countour_num):
            contour: np.ndarray = contour_list[i]
            box, min_edge = self.__box_finding(contour)
            # Nếu độ dài cạnh nhỏ nhất không thỏa mãn
            # thì loại  box đó luôn
            if min_edge < self.edge_thresh:
                continue
            # Mở rộng box ra để đảm bảo lấy chọn
            # được khoảng của chữ
            box = self.__expand(box).reshape(-1, 1, 2)
            box, min_edge = self.__box_finding(box)
            if min_edge < self.edge_thresh + 2:
                continue
            # Đảm bảo box trong kích cỡ của ảnh gốc
            box[:, 0] = np.clip(box[:, 0] / org_size[1] *
                                new_dest_size[1], 0, new_dest_size[1] - 1)
            box[:, 1] = np.clip(box[:, 1] / org_size[0] *
                                new_dest_size[0], 0, new_dest_size[0] - 1)
            box_list[i, :, :] = box.astype(np.int16)
        return box_list

    def __box_finding(self, contour: np.ndarray) -> tuple:
        # Tìm bounding box nhỏ nhất bao quanh contour
        # rồi sắp xếp tăng dần theo trục x: x1 <= x2 < x3 <= x4
        box: tuple = cv.minAreaRect(contour)
        point_list: list = sorted(list(cv.boxPoints(box)), key=lambda x: x[0])

        # Quy định lại vị trí của các điểm
        # theo thứ tự : (((1,2),(4,3))
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
        # Trả lại các điểm của bounding box
        # và min(width,height) của bounding box
        return np.array(ans), min(box[1])

    def __calc_score(self, prob_map: np.ndarray, box: np.ndarray) -> float:
        h, w = prob_map.shape[:2]
        copy_box: np.ndarray = deepcopy(box)
        # Làm tròn tọa độ của box và chuẩn hóa nó
        # trong khoảng của prob_map
        x_min: int = np.clip(
            np.floor(copy_box[:, 0].min()).astype(np.int32), 0, w - 1)
        x_max: int = np.clip(
            np.ceil(copy_box[:, 0].max()).astype(np.int32), 0, w - 1)
        y_min: int = np.clip(
            np.floor(copy_box[:, 1].min()).astype(np.int32), 0, h - 1)
        y_max: int = np.clip(
            np.ceil(copy_box[:, 1].max()).astype(np.int32), 0, h - 1)

        # Chuyển tọa độ của các điểm trong box
        # để có thể đánh dấu vào trong mask được
        copy_box[:, 0] = copy_box[:, 0] - x_min
        copy_box[:, 1] = copy_box[:, 1] - y_min
        mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        cv.fillPoly(mask, copy_box.reshape(1, -1, 2).astype(np.int32), 1)
        # Trả lại xác suất trung bình của phần được đánh dấu
        return cv.mean(prob_map[y_min:y_max + 1, x_min:x_max + 1], mask=mask)[0]

    def __expand(self, box: np.ndarray, ratio: float = 1.5) -> np.ndarray:
        polygon = Polygon(box)
        dist: float = polygon.area * ratio / polygon.length
        subject: list = [tuple(point) for point in box]
        expand = pyclipper.PyclipperOffset()
        expand.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expand_polygon: np.ndarray = np.array(expand.Execute(dist)[0])
        return expand_polygon
