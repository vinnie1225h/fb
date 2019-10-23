from __future__ import absolute_import, division, print_function, unicode_literals
import functools

'''
# %tensorflow_version only exists in Colab.
try:
    % tensorflow_version 2.x
except Exception:
    pass
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow import feature_column

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os

import pandas as pd

# from google.colab import files
import io

print(tf.__version__)
print('eager execution:', tf.executing_eagerly())


# Define the model.
def define_model():
    ## Normalize numeric features.
    # desc = pd.read_csv(io.BytesIO(uploaded['train.csv']))[NUMERIC_FEATURES].describe()
    ##desc
    # MEAN = np.array(desc.T['mean'])
    # STD = np.array(desc.T['std'])

    # def normalize_numeric_data(data, mean, std):
    ## Center the data.
    # return (data - mean) / std

    # normalizer = functools.partial(normalize_numeric_data, mean = MEAN, std = STD)

    # numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn = normalizer, shape=[len(NUMERIC_FEATURES)])
    numeric_column = tf.feature_column.numeric_column('numeric', shape=[len(NUMERIC_FEATURES)])
    numeric_columns = [numeric_column]
    # numeric_columns

    # numeric_layer = keras.layers.DenseFeatures(numeric_columns)

    # Prepare categorical_columns.
    CATEGORIES = {'cheat': ['no', 'home', 'away', 'both']}
    categorical_columns = []
    for feature, vocab in CATEGORIES.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=vocab)
        categorical_columns.append(tf.feature_column.indicator_column(cat_col))

    # Create the column layer.
    preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)

    model = keras.Sequential([
        preprocessing_layer,
        keras.layers.Dense(7, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', input_shape=(7,)),
        keras.layers.Dense(5, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    return model


# Define a utility funciton create a tf dataset from a pandas Dataframe.
def df_to_ds(in_dataframe, shuffle=True, batch_size=32):
    frame = in_dataframe.copy()
    labels = frame.pop('result')
    ds = tf.data.Dataset.from_tensor_slices((dict(frame), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(frame))
    ds = ds.batch(batch_size)
    return ds


# The show batch utility.
def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


# Numeric feature packing utility.
# NUMERIC_FEATURES = ['advantage0', 'advantage1', 'advantage2', 'advantage3', 'advantage4', 'defense_home', 'defense_away', 'goals', 'players']
NUMERIC_FEATURES = ['advantage0', 'advantage1', 'advantage2', 'advantage3', 'advantage4', 'defense_home',
                    'defense_away', 'goals']


def pack_dataset(dataset):
    class PackNumericFeatures(object):
        def __init__(self, names):
            self.names = names

        def __call__(self, features, labels):
            numeric_freatures = [features.pop(name) for name in self.names]
            numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_freatures]
            numeric_features = tf.stack(numeric_features, axis=-1)
            features['numeric'] = numeric_features
            return features, labels

    return dataset.map(PackNumericFeatures(NUMERIC_FEATURES))


# Show the fitting history.
def plot_fitting_history(history, key='loss'):
    plt.figure(figsize=(8, 5))
    val = plt.plot(history.epoch, history.history['val_' + key], '--', label='FB' + ' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label='FB' + ' Train')
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()
    plt.xlim([0, max(history.epoch)])

    plt.show(block=True)  # In PyCharm we need this to draw the figure.


RECENT_BEST_MODEL_FILE = 'recent_best_weights.h5'

