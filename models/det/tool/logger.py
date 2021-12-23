import logging
from os.path import join, isdir
from os import mkdir
import time
import json


class DBLogger:
    def __init__(self, workspace: str, level: str):
        workspace = join(workspace, 'logger')
        if not isdir(workspace):
            mkdir(workspace)
        workspace = join(workspace, 'detector')
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

    def report_time(self, name: str):
        tmp: float = time.time()
        self.__write(name + " - time: {}".format(tmp-self.time))
        self.time = tmp

    def report_measure(self, train_rs: dict):
        train_log: str = 'train_loss: {}, prob_loss: {} , thresh_loss: {}, binary_loss: {} '.format(
            train_rs['train_loss'], train_rs['train_prob_loss'],
            train_rs['train_thresh_loss'], train_rs['train_binary_loss'],
        )
        self.__write(train_log)
        metric = open(self.metric_path, 'a')
        metric.write(json.dumps(train_rs))
        metric.write("\n")
        metric.close()

    def report_partiton(self):
        self.__write("-"*55)

    def report_space(self):
        self.__write("\n")

    def __write(self, message: str):
        if self.level == logging.INFO:
            self.logger.info(message)
        else:
            self.logger.debug(message)