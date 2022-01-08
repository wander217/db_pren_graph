import torch
import numpy as np
import dgl
import yaml
from typing import Dict, List
from models.kie.structure import GateGCNNet
from models.kie.dataset import Alphabet, DocumentLabel


class KIEPredictor:
    def __init__(self, config: str, pretrained: str):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        with open(config) as f:
            data: Dict = yaml.safe_load(f)
        self.alphabet: Alphabet = Alphabet(**data['alphabet'])
        self.doc: DocumentLabel = DocumentLabel(**data['label'])
        self.model = GateGCNNet(self.alphabet.size(),
                                n_class=self.doc.size(),
                                **data['model'])
        # print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        state_dict = torch.load(pretrained, map_location=self.device)
        self.model.load_state_dict(state_dict['model'])

    def _process_ocr(self, ocr_result: List):
        texts = []
        text_lengths = []
        boxes = []
        for item in ocr_result:
            x_min, y_min, x_max, y_max, label = item
            label_encoded = self.alphabet.encode(label)
            texts.append(label_encoded)
            text_lengths.append(label_encoded.shape[0])
            box_info = [x_min, y_min, x_max, y_min, x_max, y_max,
                        x_min, y_max, x_max - x_min, y_max - y_min]
            boxes.append(box_info)
        return np.array(texts), np.array(text_lengths), np.array(boxes)

    def _norm(self, ele: np.ndarray):
        ele_min = ele.min(axis=0)
        ele_max = ele.max(axis=0)
        ele = (ele - ele_min) / (ele_max - ele_min)
        ele = (ele - 0.5) / 0.5
        return ele

    def _preprocess(self, ocr_result: List):
        texts, text_lengths, boxes = self._process_ocr(ocr_result)
        origin_boxes = boxes
        node_nums = texts.shape[0]
        src = []
        dst = []
        edge_data = []
        for i in range(node_nums):
            for j in range(node_nums):
                if i == j:
                    continue
                edata = []
                y_dist = np.mean(boxes[i][:8][1::2]) - np.mean(boxes[j][:8][1::2])
                x_dist = np.mean(boxes[i][:8][0::2]) - np.mean(boxes[j][:8][0::2])
                h = boxes[i, 9]
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

        boxes = self._norm(boxes)
        edge_data = self._norm(edge_data)
        boxes = torch.from_numpy(boxes).float()
        edge_data = torch.from_numpy(edge_data).float()
        g.edata['feat'] = edge_data
        g.ndata['feat'] = boxes

        tab_sizes_n = g.number_of_nodes()
        tab_snorm_n = torch.FloatTensor(tab_sizes_n, 1).fill_(1. / float(tab_sizes_n))
        snorm_n = tab_snorm_n.sqrt()

        tab_sizes_e = g.number_of_edges()
        tab_snorm_e = torch.FloatTensor(tab_sizes_e, 1).fill_(1. / float(tab_sizes_e))
        snorm_e = tab_snorm_e.sqrt()

        max_length = text_lengths.max()
        new_text = [np.expand_dims(np.pad(t, (0, max_length - t.shape[0]), 'constant'), axis=0) for t in texts]
        texts = np.concatenate(new_text)

        texts = torch.from_numpy(np.array(texts))
        text_lengths = torch.from_numpy(np.array(text_lengths))
        graph_node_size = g.number_of_nodes()
        graph_edge_size = g.number_of_edges()

        return (g, boxes, edge_data, snorm_n, snorm_e,
                texts, text_lengths, origin_boxes,
                graph_node_size, graph_edge_size)

    def predict(self, ocr_result: List):
        prep_data = self._preprocess(ocr_result)
        batch_graph = prep_data[0]
        batch_x = prep_data[1].to(self.device)
        batch_e = prep_data[2].to(self.device)
        texts = prep_data[5].to(self.device)
        text_length = prep_data[6].to(self.device)
        batch_snorm_n = prep_data[3].to(self.device)
        batch_snorm_e = prep_data[4].to(self.device)
        graph_node_size = prep_data[8]
        graph_edge_size = prep_data[9]
        batch_score = self.model.forward(batch_graph,
                                         batch_x, batch_e,
                                         texts, text_length,
                                         batch_snorm_n,
                                         batch_snorm_e,
                                         graph_node_size,
                                         graph_edge_size)
        batch_score = batch_score.cpu().softmax(1)
        values, pred = batch_score.max(1)
        result = [(self.doc.decode(pred[i].item()), values[i].item()) for i in range(len(pred))]
        return result
