import cv2
import numpy as np
import pandas as pd
import string

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, LSTM, Reshape, BatchNormalization, MaxPool2D, Bidirectional, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint


char_list = string.ascii_letters
filepath = "best_model_20200708.hdf5"


def modelBuilding(max_label_len):
    print("Creating Model.....")

    inputs = Input(shape=(32, 128, 1))
    conv_1 = keras.layers.Conv2D(
        64, 3, activation="relu", padding="same")(inputs)
    pool_1 = keras.layers.MaxPool2D(2, strides=2)(conv_1)

    conv_2 = keras.layers.Conv2D(
        128, 3, activation="relu", padding="same")(pool_1)
    pool_2 = keras.layers.MaxPool2D(2, strides=2)(conv_2)

    conv_3 = keras.layers.Conv2D(
        256, 3, activation="relu", padding="same")(pool_2)

    conv_4 = keras.layers.Conv2D(
        256, 3, activation="relu", padding="Same")(conv_3)
    # pooling with kernel (2,1)
    pool_4 = keras.layers.MaxPool2D(pool_size=(2, 1))(conv_4)
    conv_5 = keras.layers.Conv2D(
        512, 3, activation="relu", padding="same")(pool_4)
    # batchNorm
    batch_norm5 = BatchNormalization()(conv_5)

    conv_6 = keras.layers.Conv2D(
        512, 3, activation="relu", padding="Same")(batch_norm5)

    batch_norm6 = BatchNormalization()(conv_6)
    pool_6 = keras.layers.MaxPool2D(pool_size=(2, 1))(batch_norm6)

    conv_7 = keras.layers.Conv2D(512, 2, activation="relu")(pool_6)

    squeezed = Lambda(lambda x: k.squeeze(x, 1))(conv_7)

    # biderectional LSTM
    blstm_1 = Bidirectional(
        LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
    blsmt_2 = Bidirectional(
        LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

    outputs = Dense(len(char_list)+1, activation='softmax')(blsmt_2)

    act_model = Model(inputs, outputs)

    labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        return k.ctc_batch_cost(labels, y_pred, input_length, label_length)

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [outputs, labels, input_length, label_length])

    model = Model(inputs=[inputs, labels, input_length,
                          label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true,
                        y_pred: y_pred}, optimizer='adam')
    checkpoint = ModelCheckpoint(
        filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    act_model.save("CRNN_model_20200708.h5")

    return model, act_model, callbacks_list
