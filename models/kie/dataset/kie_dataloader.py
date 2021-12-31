import dgl
import torch
from torch.utils.data import DataLoader
from torch import Tensor
from typing import List, Dict
import numpy as np
from .kie_dataset import KIEDataset
from .alphabet import Alphabet, DocumentLabel


class KIECollate:
    def __init__(self):
        pass

    def __call__(self, batch: List):
        graphs, labels, texts, text_lengths = map(list, zip(*batch))
        labels = np.concatenate(labels)
        text_lengths = np.concatenate(text_lengths)
        max_len: int = text_lengths.max()
        texts = np.concatenate(texts)
        new_text: List = [np.expand_dims(np.pad(t, (0, max_len - t.shape[0]), 'constant'), axis=0) for t in texts]
        texts = np.concatenate(new_text)

        tab_size_n: List = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n: List = [torch.full((size, 1), 1. / float(size), dtype=torch.float32) for size in tab_size_n]
        snorm_n: Tensor = torch.cat(tab_snorm_n).sqrt()
        tab_size_e: List = [graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e: List = [torch.full((size, 1), 1. / float(size), dtype=torch.float32) for size in tab_size_e]
        snorm_e: Tensor = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        graph_node_size = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        graph_edge_size = [graphs[i].number_of_edges() for i in range(len(graphs))]

        return batched_graph, torch.from_numpy(labels), snorm_n, snorm_e, \
               torch.from_numpy(texts), torch.from_numpy(text_lengths), \
               graph_node_size, graph_edge_size


class KIELoader:
    def __init__(self, num_workers: int,
                 batch_size: int,
                 drop_last: bool, shuffle: bool,
                 pin_memory: bool, dataset: Dict,
                 alphabet: Alphabet, doc: DocumentLabel):
        self.dataset: KIEDataset = KIEDataset(**dataset, alphabet=alphabet, doc=doc)
        self.num_workers: int = num_workers
        self.batch_size: int = batch_size
        self.drop_last: bool = drop_last
        self.shuffle: bool = shuffle
        self.pin_memory: bool = pin_memory

    def build(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            collate_fn=KIECollate()
        )
