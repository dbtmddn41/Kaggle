import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.dataset import get_dataset
from src.model import get_model
from src.metrics import ToggleMetrics, EventDetectionAveragePrecision, AveragePrecision
import hydra
from omegaconf import DictConfig
import tensorflow as tf

@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    train_ds = get_dataset(cfg, mode='train')
    validation_ds = get_dataset(cfg, mode='validation')
    model = get_model(cfg)
    callbacks = [ToggleMetrics()]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
                                                                        tf.keras.metrics.AUC(curve='PR', summation_method='minoring')
                                                                        # AveragePrecision(cfg.metric.AP_threshold)
                                                                        # , EventDetectionAveragePrecision()
                                                                         ])
    print(model.summary())
    # print(model.predict(train_ds.take(1)))
    model.fit(train_ds, epochs=cfg.epochs, validation_data=validation_ds, callbacks=callbacks)

    # for d, label in train_ds.take(10):
    #     print(d, label)
    #     label = label.numpy()
    #     print(label.argmax(axis=0))
    #     a,b,c=label.argmax(axis=0)
    #     print(label[a-5:a+5,2], label[max(0, b-5):b+5,2])

if __name__ == '__main__':
    main()