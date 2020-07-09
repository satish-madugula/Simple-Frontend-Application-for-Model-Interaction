import os
import sys
import argparse

import numpy as np
import pandas as pd
import cv2
import string

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as k
from preprocessing_data import CreatingDataset
from model import modelBuilding

char_list = string.ascii_letters


# # load model
# model = keras.models.load_model("CRNN_model_20200708.h5")
# # load weights
# model.load_weights("best_model_20200708.hdf5")


def get_statename(img, model):
    test_img = []
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # img = img[0:30, 0:]
    # # convert each image of shape (32, 128, 1)
    # img = cv2.resize(img, (128, 32))
    # img = np.expand_dims(img, axis=2)
    # Normalize each image
    # img = img / 255.
    test_img.append(img)
    test_img = np.array(test_img)
    prediction = model.predict(test_img)
    out = k.get_value(k.ctc_decode(prediction, input_length=np.ones(
        prediction.shape[0]) * prediction.shape[1], greedy=True)[0][0])
    # print(out)
    i = 0
    for x in out:
        text = ''
        for p in x:
            if int(p) != -1:
                text = text + char_list[int(p)]
        i += 1
    return text
