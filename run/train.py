import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.dataset import get_dataset
from src.model import get_model, load_model
from src.metrics import ToggleMetrics, EventDetectionAveragePrecision, AveragePrecision
import hydra
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
import numpy as np
from tensorflow import keras
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    train_ds = get_dataset(cfg, mode='train')
    validation_ds = get_dataset(cfg, mode='validation')
    
    # callbacks = [ToggleMetrics()]
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=cfg.wandb.name)
    lr = keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001, decay_steps=cfg.epochs*232*2, warmup_target=0.003, warmup_steps=100)
    adam = keras.optimizers.Adam(learning_rate=lr)
    if cfg.loss == 'mixed':
        loss = CustomLoss(reduction=keras.losses.Reduction.NONE)   #self.focal_loss = keras.losses.BinaryFocalCrossentropy()
    elif cfg.loss == 'focal':
        loss = keras.losses.BinaryFocalCrossentropy()
    

    def save_model_checkpoint(epoch, logs):
        if logs['val_average_precision'] < save_model_checkpoint.best_val_average_precision:
            save_model_checkpoint.best_val_loss = logs['val_average_precision']
            model.save_weights(cfg.dir.model_save_dir+'/'+cfg.model.model_name+'.'+cfg.save_extention)
            print('Model checkpoint saved.')

    save_model_checkpoint.best_val_loss = float('inf')

# Initialize your model here

    callbacks=[]
    callbacks = [
        # keras.callbacks.TensorBoard(cfg.dir.tensorboard_logs),
        tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model_checkpoint),
        # keras.callbacks.ModelCheckpoint(cfg.dir.model_save_dir+'/'+cfg.model.model_name+'.'+cfg.save_extention, monitor='val_average_precision', save_best_only=True, mode='max', save_format="tf"),
        # keras.callbacks.EarlyStopping('val_average_precision', patience=6, start_from_epoch=10),
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.6, patience=2),
        WandbMetricsLogger(log_freq=5),
        WandbModelCheckpoint("models")
    ]
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if not cfg.finetune:
            print("create model...")
            model = get_model(cfg)
        else:
            print("load model...")
            model = load_model(cfg)
        model.compile(optimizer='adam', loss=loss, metrics=[AveragePrecision(0.0)])
        print(model.summary())
        history = model.fit(train_ds, epochs=(cfg.finetune+1)*cfg.epochs, validation_data=validation_ds, callbacks=callbacks,
                        initial_epoch=cfg.finetune*cfg.epochs)
    best_model = load_model(cfg)# keras.models.load_model(cfg.dir.model_save_dir+'/'+cfg.model.model_name+'.keras')
    print(best_model.evaluate(train_ds))
    preds = best_model.predict(validation_ds)
    y = np.concatenate([y for x, y in validation_ds], axis=0)
    m = EventDetectionAveragePrecision()
    m.update_state(y, preds)
    print(m.result())
    wandb.finish()

@keras.saving.register_keras_serializable('CustomBCE')
class CustomLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.focal_loss = keras.losses.BinaryFocalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    def call(self, y_true, y_pred):
        onset_loss = 2.*y_true[:, :, 0]*tf.math.log(y_pred[:, :, 0]) + (1 - y_true[:, :, 0])*tf.math.log(1 - y_pred[:, :, 0])
        wakeup_loss = 2.*y_true[:, :, 1]*tf.math.log(y_pred[:, :, 1]) + (1 - y_true[:, :, 1])*tf.math.log(1 - y_pred[:, :, 1])
        awake_loss = y_true[:, :, 2]*tf.math.log(y_pred[:, :, 2]) + (1 - y_true[:, :, 2])*tf.math.log(1 - y_pred[:, :, 2])
        balanced_bce =  tf.reduce_mean(-onset_loss-wakeup_loss-awake_loss)
        focal_loss_val = self.focal_loss(y_true, y_pred)
        return balanced_bce + focal_loss_val
    
    
if __name__ == '__main__':
    # keras.mixed_precision.set_global_policy('mixed_float16')
    main()
    
