import streamlit as st
import charTodigit
import numpy as np
import os
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as k

from test_for_streamlit import get_statename
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as k

path = "C:\\Users\\Satish\\Desktop\\testImages"

# load model
model = keras.models.load_model("CRNN_model_20200708.h5")
# load weights
model.load_weights("best_model_20200708.hdf5")


def load_img(img):
    img1 = img  # img1 to dispay image of actual dimenstions
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert each image of shape (32, 128, 1)
    img = cv2.resize(img, (128, 32))
    img = np.expand_dims(img, axis=2)
    text = get_statename(img, model)
    # displaying image after processing the image ...
    st.image(img1, use_column_width=True)
    st.write("Predicted state name: ", text)


st.title("State Detection:")

img_lst = []

for img in os.listdir(path):
    img_lst.append(cv2.imread(os.path.join(path, img)))

count = len(img_lst)

#####################  the below code for image selection with slider  ##################
img_num = st.slider(label="slide to change image",
                    min_value=1, max_value=count, step=1, value=1)
load_img(img_lst[img_num-1])

#####################  the below code for image selection with sidebar  ##################
# img_num = st.sidebar.number_input(
#     label="Choose Image", min_value=1, max_value=count, step=1)
# load_img(img_lst[img_num-1])

#####################  the below code for button click event ##################
# if st.button("click here to get the predicted state name"):
#     text = get_statename(img_lst[img_num-1], model)
#     st.write("Predicted state name: ", text)
