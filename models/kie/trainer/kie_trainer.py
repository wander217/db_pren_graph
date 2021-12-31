import numpy as np
import torch
from torch import nn, optim, Tensor
from models.kie.structure import GateGCNNet
from models.kie.dataset import KIELoader, Alphabet, DocumentLabel
from models.kie.criterion import OHEMLoss
from models.kie.utils import GCNLogger, GCNCheckpoint
from typing import Dict, List, Tuple
from .averager import Averager
from sklearn.metrics import confusion_matrix


class GCNTrainer:
    def __init__(self,
                 model: Dict,
                 alphabet_path: str,
                 label_path: str,
                 optimizer: Dict,
                 total_epoch: int,
                 start_epoch: int,
                 train: Dict,
                 valid: Dict,
                 criterion: Dict,
                 checkpoint: Dict,
                 logger: Dict):
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        self.alphabet: Alphabet = Alphabet(alphabet_path)
        self.doc: DocumentLabel = DocumentLabel(label_path)
        self.model: nn.Module = GateGCNNet(self.alphabet.size(), n_class=self.doc.size(), **model)
        self.model = self.model.to(self.device)
        self.criterion: nn.Module = OHEMLoss(**criterion)
        self.criterion = self.criterion.to(self.device)
        cls = getattr(optim, optimizer['name'])
        self.optimizer: optim.Optimizer = cls(self.model.parameters(), **optimizer['params'])
        self.total_epoch: int = total_epoch
        self.start_epoch: int = start_epoch
        self.train_loader = KIELoader(**train, alphabet=self.alphabet, doc=self.doc).build()
        self.valid_loader = KIELoader(**valid, alphabet=self.alphabet, doc=self.doc).build()
        self.logger: GCNLogger = GCNLogger(**logger)
        self.checkpoint: GCNCheckpoint = GCNCheckpoint(**checkpoint)
        self.step: int = 0

    def train(self):
        self.load()
        self.logger.partition_report()
        self.logger.time_report("Starting:")
        self.logger.partition_report()
        h_mean: float = 0.
        best_epoch: int = 1
        for epoch in range(self.start_epoch, self.total_epoch + 1):
            self.logger.partition_report()
            self.logger.time_report("Epoch {}:".format(epoch))
            self.logger.partition_report()

            train_rs = self.train_step()
            valid_rs = self.valid_step()
            if h_mean < valid_rs['avg_hmean']:
                h_mean = valid_rs['avg_hmean']
                self.checkpoint.save_best(self.model)
                best_epoch = epoch
            self.logger.partition_report()
            self.logger.time_report("Measure:")
            self.logger.measure_report(train_rs, valid_rs, best_epoch)
            self.logger.partition_report()
            self.save(epoch)
        self.logger.partition_report()
        self.logger.time_report("Finish:")
        self.logger.partition_report()

    def save(self, epoch: int):
        self.logger.partition_report()
        self.logger.time_report("Saving:")
        self.checkpoint.save(self.model, self.optimizer, epoch)
        self.logger.time_report("Saving complete!")
        self.logger.partition_report()

    def train_step(self):
        self.model.train()
        train_loss: Averager = Averager()
        for batch, (batch_graphs, batch_label, batch_snorm_n,
                    batch_snorm_e, text, text_length,
                    graph_node_size, graph_edge_size) in enumerate(self.train_loader):
            batch_graphs = batch_graphs.to(self.device)
            batch_x = batch_graphs.ndata['feat'].to(self.device)
            batch_e = batch_graphs.edata['feat'].to(self.device)
            text = text.to(self.device)
            text_length = text_length.to(self.device)
            batch_snorm_e = batch_snorm_e.to(self.device)
            batch_snorm_n = batch_snorm_n.to(self.device)
            batch_label = batch_label.to(self.device)
            self.optimizer.zero_grad()
            batch_score: Tensor = self.model(batch_graphs, batch_x,
                                             batch_e, text, text_length,
                                             batch_snorm_n, batch_snorm_e,
                                             graph_node_size, graph_edge_size)
            loss = self.criterion(batch_score, batch_label)
            loss.backward()
            self.optimizer.step()
            train_loss.update(loss.item(), 1)
        return {"train_loss": train_loss.calc()}

    def valid_step(self):
        self.model.eval()
        valid_loss: Averager = Averager()
        valid_hmean: Averager = Averager()
        all_score: List = []
        all_label: List = []
        with torch.no_grad():
            for batch, (batch_graphs, batch_label, batch_snorm_n,
                        batch_snorm_e, text, text_length,
                        graph_node_size, graph_edge_size) in enumerate(self.valid_loader):
                batch_graphs = batch_graphs.to(self.device)
                batch_x = batch_graphs.ndata['feat'].to(self.device)
                batch_e = batch_graphs.edata['feat'].to(self.device)
                text = text.to(self.device)
                text_length = text_length.to(self.device)
                batch_snorm_e = batch_snorm_e.to(self.device)
                batch_snorm_n = batch_snorm_n.to(self.device)
                batch_label = batch_label.to(self.device)
                batch_score: Tensor = self.model(batch_graphs, batch_x,
                                                 batch_e, text, text_length,
                                                 batch_snorm_n, batch_snorm_e,
                                                 graph_node_size, graph_edge_size)
                loss = self.criterion(batch_score, batch_label)
                valid_loss.update(loss.item(), 1)
                all_score.append(batch_score)
                all_label.append(batch_label)
            recall, precision, h_mean = self._acc(torch.cat(all_score, dim=0), torch.cat(all_label, dim=0))
            result = []
            for i in range(self.doc.size()):
                result.append({
                    "name": self.doc.decode(i),
                    "recall": recall[i],
                    "precision": precision[i],
                    "h_mean": h_mean[i]
                })
                valid_hmean.update(h_mean[i], 1)
        return {
            "valid_loss": valid_loss.calc(),
            "valid_result": result,
            "avg_hmean": valid_hmean.calc()
        }

    def load(self):
        state_dict: Dict = self.checkpoint.load()
        if state_dict is not None:
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.start_epoch = state_dict['epoch'] + 1

    def _acc(self, score: Tensor, target: Tensor) -> Tuple:
        s: np.ndarray = target.cpu().detach().numpy()
        c: np.ndarray = np.argmax(torch.nn.Softmax(dim=1)(score).cpu().detach().numpy(), axis=1)
        cm: np.ndarray = confusion_matrix(s, c).astype(np.float32)
        target = target.cpu().detach().numpy()
        nb_non_empty_class: int = 0
        recall: np.ndarray = np.zeros(self.doc.size())
        precision: np.ndarray = np.zeros(self.doc.size())
        h_mean: np.ndarray = np.zeros(self.doc.size())

        for r in range(self.doc.size()):
            cluster: np.ndarray = np.where(target == r)[0]
            if cluster.shape[0] != 0:
                recall[r] = cm[r, r] / float(cluster.shape[0])
                if np.sum(cm[:, r]) > 0:
                    precision[r] = cm[r, r] / np.sum(cm[:, r])
                else:
                    precision[r] = 0.0

                if (precision[r] + recall[r]) > 0:
                    h_mean[r] = 2 * recall[r] * precision[r] / (recall[r] + precision[r])
                else:
                    h_mean[r] = 0.

                if cm[r, r] > 0:
                    nb_non_empty_class += 1
            else:
                recall[r] = 0
                precision[r] = 0
                h_mean[r] = 0
        return recall, precision, h_mean
