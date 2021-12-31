import logging
from os.path import join, isdir
from os import mkdir
import time
import json


class GCNLogger:
    def __init__(self, workspace: str, level: str):
        workspace = join(workspace, 'logger')
        if not isdir(workspace):
            mkdir(workspace)
        workspace = join(workspace, 'recognizer')
        if not isdir(workspace):
            mkdir(workspace)

        self.level: int = logging.INFO if level == "INFO" else logging.DEBUG
        formater = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s')
        self.logger = logging.getLogger("message")
        self.logger.setLevel(self.level)

        file_handler = logging.FileHandler(join(workspace, "ouput.log"))
        file_handler.setFormatter(formater)
        file_handler.setLevel(self.level)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formater)
        stream_handler.setLevel(self.level)
        self.logger.addHandler(stream_handler)

        self.time: float = time.time()
        self.metric_path: str = join(workspace, "metric.txt")

    def time_report(self, name: str):
        tmp: float = time.time()
        self.__write(name + " - time: {}".format(tmp - self.time))
        self.time = tmp

    def measure_report(self, train_rs: dict, valid_rs: dict, best_epoch: int):
        train_log: str = 'train_loss: {}'.format(train_rs['train_loss'])
        self.__write(train_log)
        valid_log: str = 'valid_loss: {}'.format(valid_rs['valid_loss'])
        self.__write(valid_log)
        for item in valid_rs['valid_result']:
            valid_log = 'name: {},precision: {},recall: {},h_mean: {}'.format(
                item['name'], item['precision'], item['recall'], item['h_mean'])
            self.__write(valid_log)
        best_log: str = 'best_epoch: {}'.format(best_epoch)
        self.__write(best_log)
        metric = open(self.metric_path, 'a')
        metric.write(json.dumps({**train_rs, **valid_rs}))
        metric.write("\n")
        metric.close()

    def partition_report(self):
        self.__write("-" * 55)

    def space_report(self):
        self.__write("\n")

    def __write(self, message: str):
        if self.level == logging.INFO:
            self.logger.info(message)
        else:
            self.logger.debug(message)
