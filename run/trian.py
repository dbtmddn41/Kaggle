import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.dataset import get_dataset
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    ds = get_dataset(cfg, mode='train')
    # for d in ds.take(3):
    #     print(d)
    #     if (d['wakeup'] < 0).numpy().any() or ((d['onset'] < 0).numpy().any()):
    #         print(d)
    for d, label in ds.take(1):
        print(d, label)
        label = label.numpy()
        print(label.argmax(axis=0))
        a,b,c=label.argmax(axis=0)
        print(label[a-5:a+5,2], label[b-5:b+5,2])

if __name__ == '__main__':
    main()