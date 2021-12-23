import numpy as np
from shapely.geometry import Polygon
from torch import Tensor


class DBAccurancy:
    def __init__(self, ignore_thresh: float, acc_thresh: float):
        self.ignore_thresh: float = ignore_thresh
        self.acc_thresh: float = acc_thresh
        self.result: list = []

    def calc(self, pred_box_batch: list, batch: dict):
        # Lấy thông tin từ mẻ
        target_box_batch: Tensor = batch['polygon']
        ignore_batch: Tensor = batch['ignore']

        # Duyệt từng phần tử trong mẻ
        for target_polygon_list, ignore_list, pred_polygon_list \
                in zip(target_box_batch, ignore_batch, pred_box_batch):
            # Tạo lại target của từng ảnh trong mẻ
            target: list = [dict(polygon=target_polygon_list[i], ignore=ignore_list[i])
                            for i in range(len(target_polygon_list))]
            pred: list = []
            # Loại bỏ những box không đạt đủ điểm
            for i in range(pred_polygon_list.shape[0]):
                pred.append(dict(polygon=pred_polygon_list[i, :, :].tolist()))
            self.result.append(self.__evaluate(pred, target))

    def __union_area(self, polygon1, polygon2) -> float:
        return Polygon(polygon1).union(Polygon(polygon2)).area

    def __intersection_area(self, polygon1, polygon2) -> float:
        return Polygon(polygon1).intersection(Polygon(polygon2)).area

    def __calc_iou(self, polygon1, polygon2) -> float:
        intersection = self.__intersection_area(polygon1, polygon2)
        union = self.__union_area(polygon1, polygon2)
        return intersection / union

    def __evaluate(self, pred: list, target: list) -> dict:
        target_polygon: list = []
        target_ignore: list = []
        pred_polygon: list = []
        pred_ignore: list = []
        pred_match: int = 0

        # Lọc thông tin về target polygon và ignore
        for i in range(len(target)):
            polygon = target[i]['polygon']
            ignore: Tensor = target[i]['ignore']
            polygon_shape = Polygon(polygon)
            # Loại bỏ polygon không khép kín hoặc quá phức tạp
            if not polygon_shape.is_valid or not polygon_shape.is_simple:
                continue
            # Ghi lại polygon nào bị bỏ qua
            target_polygon.append(polygon)
            if ignore:
                target_ignore.append(len(target_polygon) - 1)

        # Lọc thông tin của pred
        for i in range(len(pred)):
            polygon = pred[i]['polygon']
            polygon_shape = Polygon(polygon)
            # Loại bỏ polygon không khép kín hoặc quá phức tạp
            if not polygon_shape.is_valid or not polygon_shape.is_simple:
                continue
            pred_polygon.append(polygon)
            # Nếu có phần polygon bị ignore
            # thì check diện tích phần giao giữa
            # phần ignore và polygon dự đoán
            if len(target_ignore) > 0:
                for pos in target_ignore:
                    ignore_polygon = target_polygon[pos]
                    intersection_area = self.__intersection_area(
                        polygon, ignore_polygon)
                    ignore_area = Polygon(ignore_polygon).area
                    area = 0 if ignore_area == 0 else intersection_area / ignore_area
                    # Nếu diện tích phần được giao chiếm hơn ignore_thresh
                    # thì phần dự đoán cũng bị đánh dấu bỏ qua
                    if area > self.ignore_thresh:
                        pred_ignore.append(len(pred_polygon) - 1)
                        break

        if len(pred_polygon) > 0 and len(target_polygon) > 0:
            # Lập bảng để tính giao giữa các polygon dự đoán và polygon ở target
            table = np.empty([len(target_polygon), len(pred_polygon)])
            # Lập bảng đánh dấu phần nào đã có polygon giao với
            pred_mask = np.zeros(len(pred_polygon), dtype=np.uint8)
            target_mask = np.zeros(len(target_polygon), dtype=np.uint8)
            # Tính diện tích phần giao giữa các polygon
            for i in range(len(target_polygon)):
                for j in range(len(pred_polygon)):
                    table[i][j] = self.__calc_iou(
                        target_polygon[i], pred_polygon[j]
                    )
            # Thực hiện tính số lượng box được tìm thấy chính xác
            for i in range(len(target_polygon)):
                for j in range(len(pred_polygon)):
                    if target_mask[i] == 0 and i not in target_ignore \
                            and pred_mask[j] == 0 and j not in pred_ignore:
                        if table[i][j] > self.acc_thresh:
                            target_mask[i] = 1
                            pred_mask[j] = 1
                            pred_match += 1
                            break
        # Tính tổng số polygon không bị bỏ qua cho target và pred
        total_target = len(target_polygon) - len(target_ignore)
        total_pred = len(pred_polygon) - len(pred_ignore)
        # Trả lại kết quả
        return dict(
            pred_match=pred_match,
            total_target=total_target,
            total_pred=total_pred
        )

    def gather(self):
        pred_match, total_target, total_pred = 0., 0., 0.
        for item in self.result:
            pred_match += item['pred_match']
            total_target += item['total_target']
            total_pred += item['total_pred']
        self.__reset()
        recall: float = 0 if total_target == 0 else float(
            pred_match) / total_target
        precision: float = 0 if total_pred == 0 else float(
            pred_match) / total_pred
        hmean: float = 0 if recall + precision == 0 else 2 * \
                                                         recall * precision / (recall + precision)
        return {
            'precision': precision,
            'recall': recall,
            'hmean': hmean
        }

    def __reset(self):
        self.result.clear()
