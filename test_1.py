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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", help="provide the input path of images which are need to be tested.")

    args = parser.parse_args()

    if args.path:
        path = args.path

    char_list = string.ascii_letters

    # load model
    model = keras.models.load_model("CRNN_model_20200708.h5")
    # load weights
    model.load_weights("best_model_20200708.hdf5")

    #test_img = []
    #test_filename = []

    for file in os.listdir(path):
        test_img = []
        # read input image and convert into gray scale image
        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        # img = img[0:30, 0:]
        # convert each image of shape (32, 128, 1)
        img = cv2.resize(img, (128, 32))
        img = np.expand_dims(img, axis=2)
        # Normalize each image
        #img = img / 255.
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
            # print("original_text =  ", tes_orig_txt[i])
            # # print("predicted text = ", end = '')
            for p in x:
                if int(p) != -1:
                    # char_list[int(p)], end='')
                    text = text + char_list[int(p)]
            filecontent = file, "," + text + "/n"
            print(filecontent)
            with open("result.txt", 'a+') as file:
                file.write(str(filecontent))
            print('\n')
            i += 1
