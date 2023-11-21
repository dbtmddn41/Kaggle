import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import hydra
from omegaconf import DictConfig
import tensorflow as tf
import pandas as pd
from pathlib import Path
import numpy as np
import polars as pl
from tensorflow import keras
from src.dataset import get_dataset
from src.model import load_model
from scipy.signal import find_peaks

@hydra.main(config_path="config", config_name="inference", version_base=None)
def main(cfg: DictConfig):
    model = load_model(cfg)
    test_series = (
            pl.scan_parquet(Path(cfg.dir.data_dir) / cfg.dir.feature_eng_dir / cfg.series_parquet)
            .collect()
            .to_pandas()
        )
    pred_df = test_predict(cfg, test_series, model)
    submission = get_submission(pred_df, cfg.downsample_rate,score_th=cfg.score_th, distance=cfg.distance)
    submission.write_csv(Path(cfg.dir.output_dir) / "submission.csv")
    
def test_predict(cfg: DictConfig, series: pd.DataFrame, model)\
    -> None:
    series_ids = pd.unique(series['series_id'])
    predictions = pd.DataFrame()
    for id in series_ids:
        data = []
        curr_series = series.query('series_id == @id').reset_index().sort_values(by='step')
        for i in range(len(curr_series)//cfg.duration+1):
            start, end = cfg.duration * i, min(cfg.duration * (i+1), len(curr_series))
            series_data = curr_series.loc[start: end-1, cfg.features]
            if len(series_data) != cfg.duration:
                padding_df = pd.DataFrame([[0] * len(series_data.columns)] * (cfg.duration - len(series_data)), columns=series_data.columns)
                series_data = pd.concat([series_data, padding_df], ignore_index=True)
            data.append(series_data[cfg.features].values)
            if i % 32 == 0 or i == len(curr_series)//cfg.duration:
                data = np.array(data)
                preds = pd.DataFrame(np.concatenate(model.predict(data)[:,:,:2]), columns=['onset_score', 'wakeup_score'])
                if i == len(curr_series)//cfg.duration:
                    preds = preds.iloc[:len(curr_series)%cfg.duration]
                preds['series_id'] = id
                data = []
                predictions = pd.concat([predictions, preds])
    return predictions



def get_submission(preds: pd.DataFrame, downsample_rate: int, score_th: float = 0.02, distance: int = 120) -> pl.DataFrame:
    unique_series_ids = pd.unique(preds['series_id'])
    records = []
    for series_id in unique_series_ids:
        this_series = preds.query('series_id == @series_id')

        for i, event_name in enumerate(["onset", "wakeup"]):
            this_event_scores = this_series[event_name+'_score'].values
            steps = find_peaks(this_event_scores, height=score_th, distance=distance)[0]
            scores = this_event_scores[steps]

            for step, score in zip(steps, scores):
                records.append(
                    {
                        "series_id": series_id,
                        "step": step * downsample_rate,
                        "event": event_name,
                        "score": score,
                    }
                )

    if len(records) == 0:
        records.append(
            {
                "series_id": unique_series_ids[0],
                "step": 0,
                "event": "onset",
                "score": 0,
            }
        )

    sub_df = pl.DataFrame(records).sort(by=["series_id", "step"])
    row_ids = pl.Series(name="row_id", values=np.arange(len(sub_df)))
    sub_df = sub_df.with_columns(row_ids).select(["row_id", "series_id", "step", "event", "score"])
    return sub_df


@keras.saving.register_keras_serializable('CustomBCE')
class CustomLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.focal_loss = keras.losses.BinaryFocalCrossentropy()
    def call(self, y_true, y_pred):
        onset_loss = 300.*y_true[:, :, 0]*tf.math.log(y_pred[:, :, 0]) + 0.1*(1 - y_true[:, :, 0])*tf.math.log(1 - y_pred[:, :, 0])
        wakeup_loss = 300.*y_true[:, :, 1]*tf.math.log(y_pred[:, :, 1]) + 0.1*(1 - y_true[:, :, 1])*tf.math.log(1 - y_pred[:, :, 1])
        awake_loss = y_true[:, :, 2]*tf.math.log(y_pred[:, :, 2]) + (1 - y_true[:, :, 2])*tf.math.log(1 - y_pred[:, :, 2])
        balanced_bce =  tf.reduce_mean(-onset_loss-wakeup_loss-awake_loss)
        focal_loss_val = self.focal_loss(y_true, y_pred)
        return balanced_bce + focal_loss_val
    

if __name__ == '__main__':
    main()