import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.dataset import get_dataset
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    ds = get_dataset(cfg, mode='train')
    for d in ds:
        if d['wakeup'] >= 11520 or d['onset'] >= 11520:
            print(d)

        # label = label.numpy()
        # print(label.argmax(axis=0))
        # print(label[4730:4750,2], label[10220:10230,2])

if __name__ == '__main__':
    main()