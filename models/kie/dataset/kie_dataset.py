import dgl
import numpy as np
import torch
from torch.utils.data import Dataset
from .alphabet import Alphabet, DocumentLabel
from os.path import join
import json
from typing import List, Dict

IMAGE = "image/"
TARGET = "target.json"


class KIEDataset(Dataset):
    def __init__(self, path: str, alphabet: Alphabet, doc: DocumentLabel):
        self.doc: DocumentLabel = doc
        self.alphabet: Alphabet = alphabet
        self.node_label: List = []
        self.graph_list: List = []
        self.box_list: List = []
        self.edge_list: List = []
        self.text_list: List = []
        self.text_length: List = []
        self.e_data: List = []
        self.src_list: List = []
        self.dst_list: List = []
        self._prepare(join(path, TARGET))

    def _get_annotation(self, sample: Dict):
        text_length_list: List = []
        text_encode_list: List = []
        bbox_list: List = []
        label_encode_list: List = []
        for ele in sample['target']:
            text: str = ele['text']
            label: str = ele['label']
            # flatten ra
            bbox = np.array(ele['bbox'], dtype=np.int32).reshape(-1).tolist()
            # encode text
            text_encode: np.ndarray = self.alphabet.encode(text)
            text_length_list.append(text_encode.shape[0])
            text_encode_list.append(text_encode.tolist())
            # Thêm chiều rộng
            bbox.append(np.max(bbox[0::2]) - np.min(bbox[0::2]))
            # Thêm chiều dài
            bbox.append(np.max(bbox[1::2]) - np.min(bbox[1::2]))
            bbox_list.append(bbox)
            # encode label
            label_encode: int = self.doc.encode(label)
            label_encode_list.append(label_encode)
        return np.array(text_encode_list), \
               np.array(text_length_list), \
               np.array(bbox_list), \
               np.array(label_encode_list)

    def _norm(self, ele: np.ndarray) -> np.ndarray:
        ele_min: int = ele.min(axis=0)
        ele_max: int = ele.max(axis=0)
        ele = (ele - ele_min) / (ele_max - ele_min)
        ele = (ele - 0.5) / 0.5
        return ele

    def _prepare(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            sample_list: List = json.loads(f.readline().strip("\r\t").strip("\n"))
        for sample in sample_list:
            texts, text_lengths, boxes, labels = self._get_annotation(sample)
            node_nums: int = labels.shape[0]
            src: List = []
            dst: List = []
            edge_data = []
            for i in range(node_nums):
                for j in range(node_nums):
                    if i == j:
                        continue
                    edata: List = []
                    # Tính khoảng cách trung bình của y
                    y_dist: int = np.mean(boxes[i][:8][1::2]) - np.mean(boxes[j][:8][1::2])
                    # Tính khoảng cách trung bình của x
                    x_dist: int = np.mean(boxes[i][:8][0::2]) - np.mean(boxes[j][:8][0::2])
                    h: int = boxes[i, 9]
                    if np.abs(y_dist) > 3 * h:
                        continue
                    edata.append(y_dist)
                    edata.append(x_dist)
                    edge_data.append(edata)
                    src.append(i)
                    dst.append(j)
            edge_data = np.array(edge_data)
            g = dgl.DGLGraph()
            g.add_nodes(node_nums)
            g.add_edges(src, dst)

            boxes = torch.from_numpy(self._norm(boxes)).float()
            edge_data = torch.from_numpy(self._norm(edge_data)).float()
            g.edata['feat'] = edge_data
            g.ndata['feat'] = boxes

            self.graph_list.append(g)
            self.box_list.append(boxes)
            self.edge_list.append(edge_data)
            self.src_list.append(src)
            self.dst_list.append(dst)
            self.text_length.append(text_lengths)
            self.text_list.append(texts)
            self.node_label.append(labels)

    def __getitem__(self, index: int):
        graph = self.graph_list[index]
        node = self.node_label[index]
        text = self.text_list[index]
        text_length = self.text_length[index]
        return graph, node, text, text_length

    def __len__(self):
        return len(self.node_label)
