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


def get_statename(img, model):
    test_img = []
    test_img.append(img)
    test_img = np.array(test_img)
    prediction = model.predict(test_img)
    out = k.get_value(k.ctc_decode(prediction, input_length=np.ones(
        prediction.shape[0]) * prediction.shape[1], greedy=True)[0][0])
    # print(out)
    i = 0
    filecontent = ""
    for x in out:
        text = ''
        for p in x:
            if int(p) != -1:
                # char_list[int(p)], end='')
                text = text + char_list[int(p)]
        i += 1
    return text
