import os
import cv2
import numpy as np
import pandas as pd
import string

from tensorflow.keras.preprocessing.sequence import pad_sequences

from charTodigit import encodeChar

char_list = string.ascii_letters


def CreatingDataset(path, flag):
    path = path

    """
    lists for the training and validation datasets

    """
    training_img = []
    training_txt = []
    train_input_length = []
    train_label_length = []
    orig_txt = []

    if(flag == 'train'):
        valid_img = []
        valid_txt = []
        valid_input_length = []
        valid_label_length = []
        valid_orig_txt = []

    max_label_len = 0  # max label lenght for each image, initially set to 0

    print("preparing training and validation datasets....")
    #print("preparing the dataset includes preprocessing the images... ")
    i = 1

    for root, directory, filenames in os.walk(path):
        for dirname in directory:
            dirpath = os.path.join(path, dirname)
            for file in os.listdir(dirpath):
                img = cv2.imread(os.path.join(dirpath, file),
                                 cv2.IMREAD_GRAYSCALE)
                # this gives us approximately and only the state part of the Cropped ROI
                #img = img[0:30, 0:]
                # resize the image...to width = 128 and height = 32
                img = cv2.resize(img, (128, 32))
                img = np.expand_dims(img, axis=2)
                # normalize the image
                img = img / 255.
                text = dirname  # this gives the state name.
                # compute maximum length of the text
                if len(text) > max_label_len:
                    max_label_len = len(text)
                if(flag == 'train'):
                    if i % 10 == 0:
                        valid_orig_txt.append(text)
                        valid_label_length.append(len(text))

                        valid_input_length.append(31)
                        valid_img.append(img)

                        valid_txt.append(encodeChar(text))
                    else:
                        orig_txt.append(text)
                        train_label_length.append(len(text))

                        train_input_length.append(31)
                        training_img.append(img)

                        training_txt.append(encodeChar(text))
                else:
                    orig_txt.append(text)
                    train_label_length.append(len(text))

                    train_input_length.append(31)
                    training_img.append(img)

                    training_txt.append(encodeChar(text))

                i += 1

    # pad each output label to maximum text length

    train_padded_txt = pad_sequences(
        training_txt, maxlen=max_label_len, padding='post', value=len(char_list))
    if(flag == 'train'):
        valid_padded_txt = pad_sequences(
            valid_txt, maxlen=max_label_len, padding='post', value=len(char_list))

    print("preprocessing and data prepration completed")
    if(flag == 'train'):
        return training_img, train_padded_txt, train_input_length, train_label_length, valid_img, valid_padded_txt, valid_input_length, valid_label_length, max_label_len
    else:
        return training_img, train_padded_txt, train_input_length, train_label_length, max_label_len
