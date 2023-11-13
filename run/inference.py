import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import hydra
from omegaconf import DictConfig
import tensorflow as tf
from src.dataset import get_dataset

@hydra.main(config_path="config", config_name="inference", version_base=None)
def main(cfg: DictConfig):
    test_ds = get_dataset(cfg, mode='test')
    for d in test_ds:
        print(d)
    
    
if __name__ == '__main__':
    main()