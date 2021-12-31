import torch
from torch import nn, Tensor
from .graph import GateGCNLayer, MLPReadout, DenseLayer
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
import torch.nn.functional as F
from typing import List
from dgl import DGLGraph


class GateGCNNet(nn.Module):
    def __init__(self, vocab: int,
                 n_dim_node: int, n_dim_edge: int,
                 n_hidden: int, n_class: int,
                 dropout: float, n_layer: int):
        super().__init__()
        self.text_embedding: nn.Module = nn.Embedding(vocab, n_hidden)
        self.h_embedding: nn.Module = nn.Linear(n_dim_node, n_hidden)
        self.e_embedding: nn.Module = nn.Linear(n_dim_edge, n_hidden)
        self.layers: nn.ModuleList = nn.ModuleList([
            GateGCNLayer(n_hidden, n_hidden, dropout) for _ in range(n_layer)
        ])
        self.dense: nn.ModuleList = nn.ModuleList([
            DenseLayer(n_hidden + i * n_hidden, n_hidden) for i in range(1, n_layer + 1)
        ])
        self.lstm: nn.Module = nn.LSTM(input_size=n_hidden, hidden_size=n_hidden,
                                       num_layers=1, batch_first=True,
                                       bidirectional=True)
        self.mlp: nn.Module = MLPReadout(n_hidden, n_class, 2)

    def _lstm_text_embedding(self, text: Tensor, text_length: Tensor):
        # Đảm bảo text được padding cùng size
        packed_sequence: PackedSequence = pack_padded_sequence(text, text_length.cpu(), True, False)
        output, (h_last, c_last) = self.lstm(packed_sequence)
        return h_last.mean(0)

    def _concat(self, h_list: List, i: int) -> Tensor:
        h_concat: Tensor = torch.cat(h_list, dim=1)
        output: Tensor = self.dense[i](h_concat)
        return output

    def forward(self,
                g: DGLGraph,
                h: Tensor, e: Tensor,
                text: Tensor,
                text_length: Tensor,
                snorm_n: int, snorm_e: int,
                graph_node_size: List,
                graph_edge_size: List) -> Tensor:
        h_embedding: Tensor = self.h_embedding(h)
        e_embedding: Tensor = self.e_embedding(e)
        text_embedding: Tensor = self.text_embedding(text)
        text_embedding = self._lstm_text_embedding(text_embedding, text_length)
        text_embedding = F.normalize(text_embedding)

        e: Tensor = e_embedding
        h: Tensor = h_embedding + text_embedding
        all_h: List = [h]
        for i, conv in enumerate(self.layers):
            h1, e = conv(g, h, e, snorm_n, snorm_e, graph_node_size, graph_edge_size)
            all_h.append(h1)
            h = self._concat(all_h, i)
        output: Tensor = self.mlp(h)
        return output
