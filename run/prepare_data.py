from tqdm import tqdm
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import hydra
from omegaconf import DictConfig
import polars as pl
from pathlib import Path
import joblib
import numpy as np
import tensorflow as tf

@hydra.main(config_path="config", config_name="prepare_data", version_base=None)
def main(cfg: DictConfig):
    if cfg.datatype == 'origin':
        series = (
            pl.scan_parquet(Path(cfg.dir.data_dir) / f'{cfg.phase}_series.parquet')
            .collect()
            .to_pandas()
        )
        events = pd.read_csv(Path(cfg.dir.data_dir) / f'{cfg.phase}_events.csv').dropna()
        events = events.replace({'onset':'1', 'wakeup':'2'})
    elif cfg.datatype == 'reduced':
        series = (
            pl.scan_parquet(Path(cfg.dir.reduced_data_dir) / f'{cfg.phase}_series.parquet')
            .collect()
            .to_pandas()
            .rename(columns = {'id_map' : 'series_id'})
        )
        events = (
            pl.scan_parquet(Path(cfg.dir.reduced_data_dir) / f'{cfg.phase}_events.parquet')
            .collect()
            .to_pandas()
            .dropna()
            .rename(columns = {'id_map' : 'series_id'})
        )
    preprocess_data(cfg, events, series, mode=cfg.phase)

def preprocess_data(cfg: DictConfig, events: pd.DataFrame, series: pd.DataFrame, mode: str)\
    -> None:
    series_ids = events.series_id.unique()
    data = []
    target = []
    for series_idx, id in enumerate(tqdm(series_ids)):
            
        curr_events = events.query('series_id == @id').reset_index(drop=True).sort_values(by='step')
        curr_series = series.query('series_id == @id').reset_index().sort_values(by='step')
        for i in range(len(curr_events)-1):
            start, end = max(0, curr_events.iloc[i].step-cfg.duration), min(curr_events.iloc[i].step+cfg.duration, len(curr_series))
            series_data = curr_series.loc[start: end-1, cfg.features]
            if len(series_data) != cfg.duration * 2:
                padding_df = pd.DataFrame([[0] * len(series_data.columns)] * (cfg.duration * 2 - len(series_data)), columns=series_data.columns)
                series_data = pd.concat([series_data, padding_df], ignore_index=True)
            event_target = curr_events.query('@start < step and step < @end')
            onsets = (event_target.query('event == 1').step.values - start).astype(np.uint32)
            wakeups = (event_target.query('event == 2').step.values - start).astype(np.uint32)
            data.append(series_data)
            target.append((onsets, wakeups))
        if series_idx != 0 and (series_idx % 50 == 0 or series_idx == len(series_ids)-1):
            convert_and_save(cfg, data, target, (series_idx-1)//50)
            data = []
            target = []
    return

def convert_and_save(cfg: DictConfig, data: list, target: list, idx: int) -> None:
    with tf.io.TFRecordWriter(
        str(Path(cfg.dir.processed_dir) / f"{cfg.phase}_data_{idx}.tfrec")
    ) as writer:
        for df, (onset, wakeup) in zip(data, target):
            example = create_example(df, onset, wakeup)
            writer.write(example.SerializeToString())

def create_example(df: pd.DataFrame, onset, wakeup):
    feature = {
        "onset": int64_feature_list(list(onset)),
        "wakeup": int64_feature_list(list(wakeup)),
    }
    feature.update(dict([(col, float_feature_list(list(df[col].values))) for col in df.columns]))
    return tf.train.Example(features=tf.train.Features(feature=feature))

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))
def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def int64_feature_list(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

if __name__ == "__main__":
    main()