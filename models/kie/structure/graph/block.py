import torch
from torch import nn, Tensor
from typing import List, Dict, Tuple
import torch.nn.functional as F
import copy
from dgl import DGLGraph


class GraphNorm(nn.Module):
    def __init__(self, nfeature: int, eps: float = 1e-5):
        super().__init__()
        self.eps: float = eps
        self.gamma: nn.Parameter = nn.Parameter(torch.ones(nfeature))
        self.beta: nn.Parameter = nn.Parameter(torch.zeros(nfeature))

    def norm(self, x: Tensor):
        """
            Thực hiện graph norm cho từng graph trong batch
            :param x:
            :return:
        """
        mean: Tensor = x.mean(dim=0, keepdim=True)
        var: Tensor = x.std(dim=0, keepdim=True)
        return (x - mean) / (var + self.eps)

    def forward(self, x: Tensor, graph_size: int):
        """
            Thực hiện graph norm cho batch
            :param x:
            :param graph_size: N : kích cỡ của 1 graph
            :return:
        """
        # chia graph ra
        x_list: Tensor = torch.split(x, graph_size)
        norm_list: List = [self.norm(x) for x in x_list]
        output: Tensor = torch.cat(norm_list, dim=0)
        return self.gamma * output + self.beta


class DenseLayer(nn.Module):
    def __init__(self, ninput: int, noutput: int):
        super().__init__()
        self.norm: nn.Module = nn.LayerNorm(ninput)
        self.fc: nn.Module = nn.Linear(ninput, noutput)

    def forward(self, x: Tensor) -> Tensor:
        output: Tensor = self.norm(x)
        output = F.relu(output)
        output = self.fc(output)
        return output


class GateGCNLayer(nn.Module):
    def __init__(self, ninput: int, noutput: int, dropout: float):
        super().__init__()
        self.fc1: nn.Module = nn.Linear(ninput, noutput, bias=True)
        self.fc2: nn.Module = copy.deepcopy(self.fc1)
        self.fc3: nn.Module = copy.deepcopy(self.fc1)
        self.fc4: nn.Module = copy.deepcopy(self.fc1)
        self.fc5: nn.Module = copy.deepcopy(self.fc1)

        self.norm_h: nn.Module = GraphNorm(noutput)
        self.norm_e: nn.Module = GraphNorm(noutput)
        self.residual: bool = noutput == ninput
        self.dropout: float = dropout

    def _message_func(self, edges) -> Dict:
        Bh_j = edges.src['Bh']
        e_ij = edges.data['Ce'] + edges.src['Dh'] + edges.dst['Eh']
        edges.data['e'] = e_ij
        return {'Bh_j': Bh_j, 'e_ij': e_ij}

    def _reduce_fn(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij']
        sigma_ij = torch.sigmoid(e)
        h = Ah_i + torch.sum(sigma_ij * Bh_j, dim=1) / (torch.sum(sigma_ij, dim=1) + 1e-6)
        return {'h': h}

    def forward(self, g: DGLGraph,
                h: Tensor, e: Tensor,
                snorm_n: int, snorm_e: int,
                graph_node_size: int,
                graph_edge_size: int) -> Tuple:
        h_in: Tensor = h
        e_in: Tensor = e

        g.ndata['h'] = h
        g.ndata['Ah'] = self.fc1(h)
        g.ndata['Bh'] = self.fc2(h)
        g.ndata['Dh'] = self.fc4(h)
        g.ndata['Eh'] = self.fc5(h)
        g.edata['e'] = e
        g.edata['Ce'] = self.fc3(e)
        g.update_all(self._message_func, self._reduce_fn)
        h = g.ndata['h']
        e = g.edata['e']

        h = h * snorm_n
        h = self.norm_h(h, graph_node_size)
        h = F.relu(h)

        e = e * snorm_e
        e = self.norm_e(e, graph_edge_size)
        e = F.relu(e)

        if self.residual:
            h = h_in + h
            e = e_in + e

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        return h, e


class MLPReadout(nn.Module):
    def __init__(self, ninput: int, nouput: int, L: int):
        super().__init__()
        fc_list: List = [nn.Linear(ninput // 2 ** l, ninput // 2 ** (l + 1), bias=True) for l in range(L)]
        fc_list.append(nn.Linear(ninput // 2 ** L, nouput, bias=True))
        self.fc = nn.ModuleList(fc_list)
        self.L: int = L

    def forward(self, x: Tensor) -> Tensor:
        output: Tensor = x
        for i in range(self.L):
            output = self.fc[i](output)
            output = F.relu(output)
        output = self.fc[self.L](output)
        return output
