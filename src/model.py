from omegaconf import DictConfig
import tensorflow as tf
from tensorflow import keras
from keras import layers
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.models.GRUNet import GRUNet, EncoderLayer, ResidualBiGRU
from src.models.decoder.RNNDecoder import LSTMDecoder, GRUDecoder
from src.models.dual_model import DualModel
from src.models.triple_model import TripleModel
from src.models.decoder.UNetDecoder import UNetDecoder, Down, SEModule, DoubleConv
from src.models.feature_extractor.CNN import CNN, SeparableCNN
from src.metrics import AveragePrecision

def get_model(cfg: DictConfig):
    if cfg.model.name == 'single':
        model_cfg = cfg.model.single
        if model_cfg.name == 'GRUNet':
            arch = [(8, 16, 12, 2),
                (16, 32, 10, 2),
                (32, model_cfg.params.hidden_units, 7, 2)]
            model = GRUNet(arch, model_cfg.params.hidden_units, model_cfg.params.n_layers, len(cfg.label.labels),
                           gru_dropout_rate=model_cfg.params.gru_dropout_rate, dropout_rate=model_cfg.params.dropout_rate)
        
    elif cfg.model.name == 'dual':
        # cfg.model.feature_extractor.params.strides[-1] = cfg.downsample_rate
        model = DualModel(cfg.model.feature_extractor, cfg.model.decoder,
                          (cfg.duration//cfg.downsample_rate, cfg.model.feature_extractor.params.base_filters[-1], len(cfg.model.feature_extractor.params.kernel_sizes)))
    elif cfg.model.name == 'triple':
        model = TripleModel(cfg.model.feature_extractor, cfg.model.decoder,
                          cfg.model.encoder_name, cfg.model.encoder_weights)
    model.build(input_shape=(None, cfg.duration, len(cfg.features)))
    return model


def load_model(cfg: DictConfig):
    if cfg.model.name == 'single':
        if cfg.model.single.name == 'GRUNet':
            custom_objects={"EncoderLayer": EncoderLayer, "ResidualBiGRU": ResidualBiGRU, "EncoderLayer": EncoderLayer, 'AveragePrecision': AveragePrecision}
            model = keras.models.load_model(cfg.dir.model_save_dir+'/'+cfg.model.single.name+cfg.save_extention, custom_objects=custom_objects)
    elif cfg.model.name == 'dual':
        custom_objects = {'DualModel': DualModel, 'CNN': CNN, 'UNetDecoder': UNetDecoder, 'DoubleConv':DoubleConv, 'SEModule':SEModule, 'Down':Down,
                          'LSTMDecoder': LSTMDecoder, 'GRUDecoder': GRUDecoder, 'SeparableCNN': SeparableCNN, 'AveragePrecision': AveragePrecision
                          }
        model = keras.models.load_model(cfg.dir.model_save_dir+'/'+cfg.model.model_name+cfg.save_extention, custom_objects=custom_objects)
    elif cfg.model.name == 'triple':
        custom_objects = {'DualModel': DualModel, 'CNN': CNN, 'UNetDecoder': UNetDecoder, 'DoubleConv':DoubleConv, 'SEModule':SEModule, 'Down':Down,
                          'LSTMDecoder': LSTMDecoder, 'GRUDecoder': GRUDecoder, 'SeparableCNN': SeparableCNN, 'AveragePrecision': AveragePrecision
                          }
        model = keras.models.load_model(cfg.dir.model_save_dir+'/'+cfg.model.model_name+cfg.save_extention, custom_objects=custom_objects)
    model.build(input_shape=(None, cfg.duration, len(cfg.features)))
    return model


