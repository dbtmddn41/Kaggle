from omegaconf import DictConfig, ListConfig, OmegaConf
import omegaconf
import tensorflow as tf
from tensorflow import keras
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.models.decoder.UNetDecoder import UNetDecoder
from src.models.feature_extractor.CNN import CNN

class DualModel(keras.Model):
    def __init__(self, feature_extractor_cfg: DictConfig, decoder_cfg:DictConfig):
        super().__init__()
        if not isinstance(feature_extractor_cfg, dict):
            feature_extractor_cfg = OmegaConf.to_object(feature_extractor_cfg)
        if not isinstance(decoder_cfg, dict):
            decoder_cfg = OmegaConf.to_object(decoder_cfg)
        self.feature_extractor_cfg = feature_extractor_cfg
        self.decoder_cfg = decoder_cfg
        self.feature_extractor = get_feature_extractor(feature_extractor_cfg)
        self.decoder = get_decoder(decoder_cfg)

    def call(self, inputs):
        x = self.feature_extractor(inputs)
        outputs = self.decoder(x)
        return outputs
    
    def get_config(self):
        config = super().get_config()
        config.update({"feature_extractor_cfg": self.feature_extractor_cfg,
                       'decoder_cfg': self.decoder_cfg})
        return config
    
    
def get_feature_extractor(cfg: dict):
    if cfg['name'] == 'CNN':
        feature_extractor = CNN(list(cfg['params']['base_filters']), list(cfg['params']['kernel_sizes']), list(cfg['params']['strides']), cfg['params']['pooling'])
    return feature_extractor

def get_decoder(cfg: DictConfig):
    if cfg['name'] == 'UNetDecoder':
        decoder = UNetDecoder(cfg['params']['n_classes'], cfg['params']['scale_factor'], cfg['params']['se'], cfg['params']['res'], cfg['params']['dropout'])
    return decoder
