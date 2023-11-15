from omegaconf import DictConfig
import tensorflow as tf
from tensorflow import keras
from keras import layers
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.models.GRUNet import GRUNet, EncoderLayer, ResidualBiGRU

def get_model(cfg: DictConfig):
    if cfg.model_type == 'single':
        if cfg.model == 'GRUNet':
            arch = [(8, 16, 12, 2),
                (16, 32, 10, 2),
                (32, cfg.hidden_units, 7, 2)]
            model = GRUNet(arch, cfg.hidden_units, cfg.n_layers, len(cfg.label.labels), gru_dropout_rate=cfg.gru_dropout_rate, dropout_rate=cfg.dropout_rate)
    elif cfg.model_type == 'dual':
        pass
    elif cfg.model_type == 'triple':
        pass
    model.build(input_shape=(None, cfg.duration, len(cfg.features)))
    return model


def load_model(cfg: DictConfig):
    if cfg.model_type == 'single':
        if cfg.model == 'GRUNet':
            custom_objects={"EncoderLayer": EncoderLayer, "ResidualBiGRU": ResidualBiGRU, "EncoderLayer": EncoderLayer, }
            model = keras.models.load_model(cfg.dir.model_save_dir+'/'+cfg.model+'.keras', custom_objects=custom_objects)
    elif cfg.model_type == 'dual':
        pass
    elif cfg.model_type == 'triple':
        pass
    model.build(input_shape=(None, cfg.duration, len(cfg.features)))
    return model
