import copy
from typing import List
from models.det.trainer import DBNetwork
from models.det.dataloader.preprocess import DBNormalize
from os.path import join
from os import mkdir
import cv2 as cv
import torch
import yaml
import math
import numpy as np
from copy import deepcopy
from shapely.geometry import Polygon
import pyclipper
import time

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
file_config: str = r'D:\invoice-extractor\data\det\config\eb0-config.yaml'
with open(file_config) as stream:
    data = yaml.safe_load(stream)
net = DBNetwork(device=device, **data['structure'])
state_dict = torch.load(r'D:\invoice-extractor\data\det\pretrained\checkpoint466.pth', map_location=device)
net.load_state_dict(state_dict['model'])
net.eval()
model = net.model
path_dir = r"D:\backup\book"
with torch.no_grad():
    for count in range(7, 8):
        img = cv.imread(join(path_dir, "book{}.jpg".format(count)))
        d = 0.5
        org_h, org_w, _ = img.shape
        new_h = math.floor(org_h / 32) * 32
        new_w = math.floor(org_w / 32) * 32
        limit = 960
        # if max(new_w, new_h) < limit:
        #     if org_w > org_h:
        new_w = limit
        new_h = int((org_h / org_w) * new_w)
        new_h = math.floor(new_h / 32) * 32
        # else:
        #     new_h = limit
        #     new_w = int((org_w / org_h) * new_h)
        #     new_w = math.floor(new_w / 32) * 32
        print(new_w, new_h)
        img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_CUBIC)
        # img = cv.erode(img, np.ones((1, 1)).astype(np.uint8), 1)
        batch = {
            'image': img
        }
        norm = DBNormalize(mean=[122.67891434, 116.66876762, 104.00698793])
        batch = norm.build(batch, retype_polygon=False)
        start = time.time()
        pred = model(batch['image'].reshape(-1, 3, new_h, new_w))
        print(time.time() - start)
        prob_map = pred['binary_map'].cpu().detach().numpy()[0][0]
        bitmap = np.uint8((prob_map * 255)).astype(np.int32) > 0
        contour_list, _ = cv.findContours(np.uint8(bitmap * 255), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        countour_num: int = min(len(contour_list), 1000)
        box_list: np.ndarray = np.zeros((countour_num, 4, 2), dtype=np.int16)
        score_list: np.ndarray = np.zeros((countour_num,), dtype=np.float32)


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


        def calc_score(prob_map: np.ndarray, box: np.ndarray) -> float:
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


        def expand(box: np.ndarray, ratio: float = 2) -> np.ndarray:
            polygon = Polygon(box)
            dist: float = polygon.area * ratio / polygon.length
            subject: list = [tuple(point) for point in box]
            expand = pyclipper.PyclipperOffset()
            expand.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expand_polygon: np.ndarray = np.array(expand.Execute(dist)[0])
            return expand_polygon


        for i in range(countour_num):
            contour: np.ndarray = contour_list[i]
            box, min_edge = box_finding(contour)
            # Nếu độ dài cạnh nhỏ nhất không thỏa mãn
            # thì loại  box đó luôn
            if min_edge < 3:
                continue
            # Tính score từ prob_map và box
            # score bằng giá trị trung bình của tổng xác suất trong box
            # Nếu score thấp hơn ngưỡng thì loại bỏ
            score: float = calc_score(prob_map, box.reshape(-1, 2))
            box = expand(box).reshape((-1, 1, 2))
            box, min_edge = box_finding(box)
            if min_edge < 5:
                continue
            # Đảm bảo box trong kích cỡ của ảnh gốc
            box[:, 0] = np.clip(box[:, 0] / img.shape[1] * new_w, 0, new_w - 1)
            box[:, 1] = np.clip(box[:, 1] / img.shape[0] * new_h, 0, new_h - 1)
            box_list[i, :, :] = box.astype(np.int16)
            score_list[i] = score


        def find_key(box: List):
            new_box = np.array(box)
            x_min = new_box[:, 0].min()
            x_max = new_box[:, 0].max()
            y_min = new_box[:, 1].min()
            y_max = new_box[:, 1].max()
            return x_min, y_min, x_max, y_max


        new_box_list = []
        rectangle = []
        thresh = 30
        extend = 3
        for i in range(len(box_list)):
            p1 = Polygon(box_list[i])
            if p1.area <= 0:
                continue
            x1_min, y1_min, x1_max, y1_max = find_key(box_list[i])
            new_box_list.append(np.array([
                [x1_min, y1_min], [x1_max, y1_min],
                [x1_max, y1_max], [x1_min, y1_max]
            ]))
            rectangle.append([x1_min, x1_max, y1_min, y1_max])
        save_path = join(r"C:\Users\TrinhThinh\Documents\real_text\book1", str(count))
        mkdir(save_path)
        new_image = copy.deepcopy(img)
        for i in range(len(new_box_list)):
            x_min, x_max, y_min, y_max = rectangle[i]
            tmp = img[y_min:y_max + 1, x_min:x_max + 1, :]
            img_name = "img{}.jpg".format(i)
            cv.rectangle(new_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv.imwrite(join(save_path, img_name), tmp)
        cv.imshow("aaa", new_image)
        cv.waitKey(0)
