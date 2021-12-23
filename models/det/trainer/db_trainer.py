import torch
from torch import optim
from models.det.measure import DBAccurancy, DBScore
from models.det.dataloader import DBLoader
from models.det.tool import DBCheckpoint, DBLogger
from models.det.structure import DBNetwork
from .element import Holder, Optimizer


class DBTrainer:
    def __init__(self, total_epoch: int, start_epoch: int, **kwargs):
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')

        self.network = DBNetwork(device=self.device, **kwargs['structure'])
        self.network.to(self.device)
        self.train_loader = DBLoader(**kwargs['train']).build()
        #         self.valid_loader = DBLoader(**kwargs['valid']).build()
        self.optimizer = Optimizer(
            **kwargs['optimizer']).build(self.network.parameters())
        self.checkpoint = DBCheckpoint(**kwargs['checkpoint'])
        self.logger = DBLogger(**kwargs['logger'])
        self.accurancy = DBAccurancy(**kwargs['acc_fn']['accurancy'])
        self.score = DBScore(**kwargs['acc_fn']['score'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=total_epoch * len(self.train_loader)
        )

        self.total_epoch: int = 1 + total_epoch
        self.start_epoch: int = start_epoch

    def train(self):
        self.__load_step()

        self.logger.report_partiton()
        self.logger.report_time("Start")
        self.logger.report_partiton()
        self.logger.report_space()
        for i in range(self.start_epoch, self.total_epoch):
            self.logger.report_partiton()
            self.logger.report_time("Epoch {}".format(i))
            train_rs: dict = self.__train_step()
            self.__save_step(train_rs, i)

        self.logger.report_partiton()
        self.logger.report_time("Finish")
        self.logger.report_partiton()

    def __load_step(self):
        state_dict: dict = self.checkpoint.load()
        if state_dict is not None:
            self.network.load_state_dict(state_dict['model'])

    #             self.optimizer.load_state_dict(state_dict['optimizer'])
    #             self.scheduler.load_state_dict(state_dict['scheduler'])
    #             self.start_epoch = state_dict['epoch'] + 1

    def __train_step(self) -> dict:
        self.logger.report_partiton()
        self.logger.report_time("Training:")
        self.logger.report_partiton()

        self.network.train()
        train_loss: Holder = Holder()
        train_prob_loss: Holder = Holder()
        train_thresh_loss: Holder = Holder()
        train_binary_loss: Holder = Holder()
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss, _, metric = self.network(batch)
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            batch_size = batch['image'].shape[0]
            train_loss.update(loss.item() * batch_size, batch_size)
            train_prob_loss.update(
                metric['prob_loss'].item() * batch_size,
                batch_size
            )
            train_thresh_loss.update(
                metric['thresh_loss'].item() * batch_size,
                batch_size
            )
            train_binary_loss.update(
                metric['binary_loss'].item() * batch_size,
                batch_size
            )
        return {
            'train_loss': train_loss.average(),
            'train_prob_loss': train_prob_loss.average(),
            'train_thresh_loss': train_thresh_loss.average(),
            'train_binary_loss': train_binary_loss.average(),
        }

    def __valid_step(self) -> dict:
        self.logger.report_partiton()
        self.logger.report_time("Validation:")
        self.logger.report_partiton()

        self.network.eval()
        valid_loss: Holder = Holder()
        valid_score: Holder = Holder()
        with torch.no_grad():
            for batch in self.valid_loader:
                loss, pred, _ = self.network(batch)
                loss = loss.mean()
                batch_size = batch['image'].shape[0]
                valid_loss.update(loss.item() * batch_size, batch_size)
                box_list, score_list = self.score.calc(pred, batch)
                self.accurancy.calc(box_list, score_list, batch)
            result: dict = self.accurancy.gather()
        return {
            'valid_loss': valid_loss.average(),
            'valid_precision': result['precision'],
            'valid_recall': result['recall'],
            'valid_hmean': result['hmean']
        }

    def __save_step(self, train_rs: dict, epoch: int):
        self.logger.report_partiton()
        self.logger.report_time("Measure:")
        self.logger.report_measure(train_rs)
        self.logger.report_partiton()

        self.logger.report_time("Saving")
        self.checkpoint.save(
            model=self.network,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
        )
        self.logger.report_partiton()
        self.logger.report_space()
