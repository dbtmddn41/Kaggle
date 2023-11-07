import numpy as np
import pandas as pd
import tensorflow as tf
import typing 
import polars as pl
from omegaconf import DictConfig
from pathlib import Path
import os, joblib

def get_dataset(cfg: DictConfig) -> None:
    # os.walk(Path(cfg.dir.processed_dir))
    global features
    # features = cfg.features
    features = ['enmo', 'anglez']

    filenames = tf.io.gfile.glob('/home/lyu/AI/CMI/data/preprocessed_data/'+"*0.tfrec")
    print(filenames)
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = (
    tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    .map(parse_tfrecord_fn , num_parallel_calls=AUTOTUNE)
    # .map(prepare_sample, num_parallel_calls=AUTOTUNE)
    # .shuffle(cfg.batch_size * 10)
    .batch(1)
    # .prefetch(AUTOTUNE)
    )
    return dataset

def parse_tfrecord_fn(example):
    feature_description = {
        "onset": tf.io.VarLenFeature(tf.int64),
        "wakeup": tf.io.VarLenFeature(tf.int64)
    }
    for feats in features:
        feature_description[feats] = tf.io.FixedLenFeature([], tf.float32)
    example = tf.io.parse_single_example(example, feature_description)
    # example["onset"] = tf.sparse.to_dense(example["onset"])
    return example

    
if __name__ == '__main__':
    print('============================')
    ds = get_dataset({})
    for d in ds.take(1):
        print(d)
        print('============================')