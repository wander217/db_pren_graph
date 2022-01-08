from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import os
import gdown
from models.det.predictor import DBPredictor
from models.kie.predictor import KIEPredictor

det_id = 'https://drive.google.com/uc?id=1VNdfZpEzpwVjWBnJQYDm0wehrcakF0Pt'
rec_id = 'https://drive.google.com/uc?id=1XsvP-gOl8azcMR5VkK_SCdpbpKMRuVh0'
kie_id = 'https://drive.google.com/uc?id=1bEhDefCj5IWbXV5xfUlluTbNquAqzvxm'


def download(url: str, out: str):
    return gdown.cached_download(url, out, md5=None, quiet=False)


def load_det():
    config_path = r'data/det/config/eb0-config.yaml'
    pretrained_path = r'data/det/pretrained/det_pretrained.pth'
    weight = pretrained_path
    if not os.path.isfile(pretrained_path):
        weight = gdown.download(det_id, pretrained_path)
    else:
        print("File exist: {}".format(pretrained_path))
    return DBPredictor(config_path, weight)


def load_rec():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
    config['cnn']['pretrained'] = False
    del config['cnn']['pretrained']
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = False
    return Predictor(config)


def load_kie():
    config_path = r'data/kie/config/gcn-config.yaml'
    pretrained_path = r'data/kie/pretrained/kie_pretrained.pth'
    weight = pretrained_path
    if not os.path.isfile(pretrained_path):
        weight = gdown.download(det_id, pretrained_path)
    else:
        print("File exist: {}".format(pretrained_path))
    return KIEPredictor(config_path, weight)


def loader():
    return load_det(), load_rec(), load_kie()
