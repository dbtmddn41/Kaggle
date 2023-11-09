import numpy as np
import pandas as pd
import tensorflow as tf
import typing 
import polars as pl
from omegaconf import DictConfig
from pathlib import Path
import os, joblib
from scipy import stats
from functools import partial

def get_dataset(cfg: DictConfig, mode='train') -> None:
    def parse_tfrecord_fn(example):
        feature_description = {
            "onset": tf.io.VarLenFeature(tf.int64),
            "wakeup": tf.io.VarLenFeature(tf.int64),
        }
        for feats in cfg.features:
            feature_description[feats] = tf.io.VarLenFeature(tf.float32)
        example = tf.io.parse_single_example(example, feature_description)
        example["onset"] = tf.sparse.to_dense(example["onset"])
        example["wakeup"] = tf.sparse.to_dense(example["wakeup"])
        for feats in cfg.features:
            example[feats] = tf.sparse.to_dense(example[feats])
        return example
    
    def make_dataset(mode, example):
        if mode == 'train':
            seq_len = cfg.duration * 2
            start = rng.make_seeds(1)[0] % (seq_len - cfg.duration)
            start = tf.squeeze(start)
            end = start + cfg.duration
            for feats in cfg.features:
                if example[feats].shape[0] == seq_len:
                    example[feats] = tf.slice(example[feats], start, end)
            for label_name in ('onset', 'wakeup'):
                example[label_name] = tf.boolean_mask(example[label_name], (start <= example[label_name]) & (example[label_name] < end))
        elif mode == 'validation':
            seq_len, start = cfg.duration, 0
            end = seq_len
        
        feature = {}
        for feats in cfg.features:
            feature[feats] = example[feats]
        label = get_label(example, seq_len//2, start)
        if mode == 'train':
            label = gaussian_label(
                label, offset=cfg.label.offset, sigma=cfg.label.sigma
            )

        return feature, label
    # os.walk(Path(cfg.dir.processed_dir))

    filenames = tf.io.gfile.glob(cfg.dir.processed_dir+"/*1.tfrec")
    AUTOTUNE = tf.data.AUTOTUNE
    rng = tf.random.Generator.from_seed(123, alg='philox')
    dataset = (
    tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    .map(parse_tfrecord_fn , num_parallel_calls=AUTOTUNE)
    .map(partial(make_dataset, mode), num_parallel_calls=AUTOTUNE)
    # .shuffle(cfg.batch_size * 10)
    # .batch(1)
    # .prefetch(AUTOTUNE)
    )
    return dataset

def get_label(example, label_len: int, start: int):
    onset_label= tf.zeros((label_len,))
    wakeup_label = tf.zeros((label_len,))
    awake_label = tf.fill((label_len,), -1.)
    start = tf.squeeze(start)
    onset_label = tf.tensor_scatter_nd_update(onset_label, example['onset'][:, tf.newaxis]-start, tf.cast(tf.ones_like(example['onset']), tf.float32))
    awake_label = tf.tensor_scatter_nd_update(awake_label, example['onset'][:, tf.newaxis]-start-1, tf.cast(tf.zeros_like(example['onset']), tf.float32))
    wakeup_label = tf.tensor_scatter_nd_update(wakeup_label, example['wakeup'][:, tf.newaxis]-start, tf.cast(tf.ones_like(example['wakeup']), tf.float32))
    awake_label = tf.tensor_scatter_nd_update(awake_label, example['wakeup'][:, tf.newaxis]-start-1, tf.cast(tf.ones_like(example['wakeup']), tf.float32))
    
    mask = tf.concat([[-1], tf.squeeze(tf.where(awake_label > -1))], axis=0)
    for idx in tf.range(tf.shape(mask)[0]-1):
        indices = tf.range(mask[idx]+1, mask[idx+1])[:, tf.newaxis]
        updates = tf.repeat(awake_label[mask[idx+1]], tf.shape(indices)[0])
        awake_label = tf.tensor_scatter_nd_update(awake_label, indices, updates)
    indices = tf.range(mask[-1]+1, tf.shape(awake_label)[0])[:, tf.newaxis]
    updates = tf.repeat(tf.cast(tf.logical_not(tf.cast(awake_label[mask[-1]], tf.bool)), tf.float32), tf.shape(indices)[0])
    #라벨이 하나도 없으면 그냥 깨어있는걸로
    if tf.shape(mask)[0] == 1:
        updates = tf.repeat(1., tf.shape(indices)[0])
    awake_label = tf.tensor_scatter_nd_update(awake_label, indices, updates)
    return tf.stack([onset_label, wakeup_label, awake_label], axis=1)

def gaussian_label(label: tf.Tensor, offset: int, sigma: int) -> tf.Tensor:
    num_events = 2
    rv = stats.norm(0,sigma)
    x = np.arange(-offset, offset+1)
    gaussian_kernel = rv.pdf(x)
    gaussian = tf_convolve(label[:, :num_events], gaussian_kernel)

    res = tf.concat([gaussian, label[:, num_events:]], axis=1)
    return res


def tf_convolve(input, kernel):
    input = tf.transpose(input)[:, :, tf.newaxis]
    kernel = tf.cast(kernel[:, tf.newaxis, tf.newaxis], tf.float32)
    res = tf.nn.convolution(input, kernel, padding='SAME')
    return tf.transpose(tf.squeeze(res))
    
if __name__ == '__main__':
    print('============================')
    ds = get_dataset({})
    for d in ds.take(1):
        print(d)
        print('============================')