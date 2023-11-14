import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.dataset import get_dataset
from src.model import get_model
from src.metrics import ToggleMetrics, EventDetectionAveragePrecision, AveragePrecision
import hydra
from omegaconf import DictConfig
import tensorflow as tf
import numpy as np
from tensorflow import keras

@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    train_ds = get_dataset(cfg, mode='train')
    validation_ds = get_dataset(cfg, mode='validation')
    model = get_model(cfg)
    # callbacks = [ToggleMetrics()]
    lr = keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001, decay_steps=cfg.epochs*232*2, warmup_target=0.003, warmup_steps=100)
    adam = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AveragePrecision(0.0)])
    print(model.summary())
    callbacks = [
        # keras.callbacks.TensorBoard(cfg.dir.tensorboard_logs),
        keras.callbacks.ModelCheckpoint(cfg.dir.model_save_dir+'/'+cfg.model+'.keras', monitor='val_average_precision', save_best_only=True),
        keras.callbacks.EarlyStopping('val_average_precision', patience=6, start_from_epoch=10),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5),
    ]
    try:
        history = model.fit(train_ds, epochs=cfg.epochs, validation_data=validation_ds, callbacks=callbacks, class_weight={0:1.,1:1.,2:0.5})
    finally:
        best_model = keras.models.load_model(cfg.dir.model_save_dir+'/'+cfg.model+'.keras')
        preds = best_model.predict(validation_ds)
        y = np.concatenate([y for x, y in validation_ds], axis=0)
        m = EventDetectionAveragePrecision()
        m.update_state(y, preds)
        print(m.result())
if __name__ == '__main__':
    main()