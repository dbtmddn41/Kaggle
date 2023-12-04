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
    
    if cfg.phase == 'test':
        series = (
            pl.scan_parquet(Path(cfg.dir.data_dir) / f'test_series.parquet')
            .collect()
            .to_pandas()
        )
        series_ids = pd.unique(series['series_id'])
        id_map = pd.DataFrame(enumerate(series_ids), columns=['id_map', 'series_id'])
        series = series.merge(id_map, how='left', on='series_id')
        id_map = dict(enumerate(series_ids))
        preprocess_test_data(cfg, series, id_map)
        return
    
    if cfg.datatype == 'origin':
        series = (
            pl.scan_parquet(Path(cfg.dir.data_dir) / f'train_series.parquet')
            .collect()
            .to_pandas()
        )
        series_ids = pd.unique(series['series_id'])
        id_map = pd.DataFrame(enumerate(series_ids), columns=['id_map', 'series_id'])
        series = series.merge(id_map, how='left', on='series_id')
        events = pd.read_csv(Path(cfg.dir.data_dir) / f'train_events.csv').dropna().merge(id_map, how='left', on='series_id')
        events = events.replace({'onset':1, 'wakeup':2})
        id_map = dict([(id, i) for i, id in enumerate(series_ids)])
    elif cfg.datatype == 'reduced':
        id_map = (
            pl.scan_parquet(Path(cfg.dir.reduced_data_dir) / f'train_id_map.parquet')
            .collect()
            .to_pandas()
            )
        id_map = dict([(row['series_id'], row['id_map']) for _, row in id_map.iterrows()])
        series = (
            pl.scan_parquet(Path(cfg.dir.reduced_data_dir) / f'train_series.parquet')
            .collect()
            .to_pandas()
        )
        events = (
            pl.scan_parquet(Path(cfg.dir.reduced_data_dir) / f'train_events.parquet')
            .collect()
            .to_pandas()
            .dropna()
        )
    elif cfg.datatype == 'feature_engineering':
        series = (
            pl.scan_parquet(Path(cfg.dir.feature_eng_dir) / cfg.series_parquet)
            .collect()
            .to_pandas()
        )
        # series_ids = pd.unique(series['series_id'])
        id_map = series[['id_map', 'series_id']].drop_duplicates()
        # series = series.merge(id_map, how='left', on='series_id')
        events = pd.read_csv(Path(cfg.dir.data_dir) / cfg.event_csv).dropna().merge(id_map, how='left', on='series_id')
        events['timestamp'] = pd.to_datetime(events["timestamp"]).apply(lambda t: t.tz_localize(None))
        events = events.replace({'onset':1, 'wakeup':2})
        id_map = dict([(series_id,id_map) for i, (id_map, series_id) in id_map.iterrows()])
    if cfg.phase.startswith('train'):
        preprocess_train_data(cfg, events, series, id_map)
    elif cfg.phase == 'validation':
        preprocess_valid_data(cfg, events, series, id_map)

def preprocess_test_data(cfg: DictConfig, series: pd.DataFrame, id_map: dict)\
    -> None:
    series_ids = id_map.keys()
    data = []
    for series_idx, id in enumerate(tqdm(series_ids)):
        curr_series = series.query('id_map == @id').reset_index().sort_values(by='step')
        for i in range(len(curr_series)//cfg.duration+1):
            start, end = cfg.duration * i, min(cfg.duration * (i+1), len(curr_series))
            series_data = curr_series.loc[start: end-1, cfg.features]
            if len(series_data) != cfg.duration:
                padding_df = pd.DataFrame([[0] * len(series_data.columns)] * (cfg.duration - len(series_data)), columns=series_data.columns)
                series_data = pd.concat([series_data, padding_df], ignore_index=True)
            data.append(series_data)
        if series_idx != 0 and (series_idx % 10 == 0 or series_idx == len(series_ids)-1):
            convert_and_save(cfg, data, [], (series_idx-1)//10)
            data = []
    return

def preprocess_valid_data_old(cfg: DictConfig, events: pd.DataFrame, series: pd.DataFrame, id_map: dict)\
    -> None:
    series_ids = set(events.id_map.unique())
    series_ids &= set(map(lambda x: id_map[x], cfg.split.valid_series_ids))
    data = []
    target = []
    for series_idx, id in enumerate(tqdm(series_ids)):
        curr_events = events.query('id_map == @id').reset_index(drop=True).sort_values(by='step')
        time_delta = pd.Timedelta(hours=8)      #8시간은 그냥 고정값으로 쓰겠습니다.
        #reduce 버전은 timestamp가 str이 아니라 진짜 timestamp로 되어있음. reduce가 아닌 버전을 쓰려면 수정 필요.
        start_time, end_time = curr_events.iloc[0].timestamp - time_delta, curr_events.iloc[-1].timestamp + time_delta
        curr_series = series.query('id_map == @id').reset_index().sort_values(by='step')
        curr_series = curr_series.query('@start_time < timestamp and  timestamp < @end_time')
        for i in range(len(curr_series)//cfg.duration+1):
            start, end = cfg.duration * i, min(cfg.duration * (i+1), len(curr_series))
            series_data = curr_series.loc[start: end-1, cfg.features]
            if len(series_data) != cfg.duration:
                padding_df = pd.DataFrame([[0] * len(series_data.columns)] * (cfg.duration - len(series_data)), columns=series_data.columns)
                series_data = pd.concat([series_data, padding_df], ignore_index=True)
            event_target = curr_events.query('@start < step and step < @end')
            onsets = (event_target.query('event == 1').step.values - start).astype(np.uint32)
            wakeups = (event_target.query('event == 2').step.values - start).astype(np.uint32)
            data.append(series_data)
            target.append((onsets, wakeups))
        if series_idx != 0 and (series_idx % 5 == 0 or series_idx == len(series_ids)-1):
            convert_and_save(cfg, data, target, (series_idx-1)//5)
            data = []
            target = []
    return

def preprocess_valid_data(cfg: DictConfig, events: pd.DataFrame, series: pd.DataFrame, id_map: dict)\
    -> None:
    series_ids = set(events.id_map.unique())
    series_ids &= set(map(lambda x: id_map[x], cfg.split.valid_series_ids))
    data = []
    target = []
    for series_idx, id in enumerate(tqdm(series_ids)):
        curr_events = events.query('id_map == @id').reset_index(drop=True).sort_values(by='step')
        curr_series = series.query('id_map == @id').reset_index().sort_values(by='step')
        for i in range(len(curr_events)-1):
            start = curr_events.iloc[i].step
            start = np.random.randint(max(0, start-cfg.duration), start)
            end =  min(start+cfg.duration, len(curr_series))
            series_data = curr_series.loc[start: end-1, cfg.features]
            if len(series_data) != cfg.duration:
                padding_df = pd.DataFrame([[0] * len(series_data.columns)] * (cfg.duration - len(series_data)), columns=series_data.columns)
                series_data = pd.concat([series_data, padding_df], ignore_index=True)
            event_target = curr_events.query('@start < step and step < @end')
            onsets = (event_target.query('event == 1').step.values - start).astype(np.uint32)
            wakeups = (event_target.query('event == 2').step.values - start).astype(np.uint32)
            data.append(series_data)
            target.append((onsets, wakeups))
        if series_idx != 0 and (series_idx % 10 == 0 or series_idx == len(series_ids)-1):
            convert_and_save(cfg, data, target, (series_idx-1)//10)
            data = []
            target = []
    return

def preprocess_train_data(cfg: DictConfig, events: pd.DataFrame, series: pd.DataFrame, id_map: dict)\
    -> None:
    series_ids = set(events.id_map.unique())
    if cfg.phase != 'train_all':
        series_ids &= set(map(lambda x: id_map[x], cfg.split.train_series_ids))
    data = []
    target = []
    for series_idx, id in enumerate(tqdm(series_ids)):
        curr_events = events.query('id_map == @id').reset_index(drop=True).sort_values(by='step')
        curr_series = series.query('id_map == @id').reset_index().sort_values(by='step')
        for i in range(len(curr_events)-1):
            if curr_events.iloc[i].event ==1 and curr_events.iloc[i+1].event ==2 and curr_events.iloc[i].night==curr_events.iloc[i+1].night:
                start, end = max(0, curr_events.iloc[i].step-cfg.duration), min(curr_events.iloc[i].step+cfg.duration, len(curr_series))
                series_data = curr_series.loc[start: end-1, cfg.features]
                if len(series_data) != cfg.duration*2:
                    padding_df = pd.DataFrame([[0] * len(series_data.columns)] * (cfg.duration*2 - len(series_data)), columns=series_data.columns)
                    series_data = pd.concat([series_data, padding_df], ignore_index=True)
                event_target = curr_events.query('@start < step and step < @end')
                onsets = (event_target.query('event == 1').step.values - start).astype(np.uint32)
                wakeups = (event_target.query('event == 2').step.values - start).astype(np.uint32)
                data.append(series_data)
                target.append((onsets, wakeups))
        if series_idx != 0 and (series_idx % 5 == 0 or series_idx == len(series_ids)-1):
            convert_and_save(cfg, data, target, (series_idx-1)//5)
            data = []
            target = []
    return

def convert_and_save(cfg: DictConfig, data: list, target: list, idx: int) -> None:
    with tf.io.TFRecordWriter(
        str(Path(cfg.dir.processed_dir) / f"{cfg.phase}_data_{idx}.tfrec")
    ) as writer:
        if cfg.phase.startswith('train') or cfg.phase == 'validation':
            for df, (onset, wakeup) in zip(data, target):
                example = create_train_example(df, onset, wakeup)
                writer.write(example.SerializeToString())
        elif cfg.phase == 'test':
            for df in data:
                example = create_test_example(df)
                writer.write(example.SerializeToString())

def create_train_example(df: pd.DataFrame, onset, wakeup):
    feature = {
        "onset": int64_feature_list(list(onset)),
        "wakeup": int64_feature_list(list(wakeup)),
    }
    feature.update(dict([(col, float_feature_list(list(df[col].values))) for col in df.columns]))
    return tf.train.Example(features=tf.train.Features(feature=feature))
def create_test_example(df: pd.DataFrame):
    feature = {}
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