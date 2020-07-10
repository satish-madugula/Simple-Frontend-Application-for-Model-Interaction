import streamlit as st
import charTodigit
import numpy as np
import os
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as k

from process_text import get_statename
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as k

#path = "C:\\Users\\Satish\\Desktop\\testImages"

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
    st.write("Predicted state name: <b>" +
             text + "</b>", unsafe_allow_html=True)


st.markdown(" ## State Detection:")
img = st.file_uploader(label="upload image", type=['JPG', "jpg"])
if img is not None:
    #image = Image.open(img)
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    #st.image(image, use_column_width=True)
    load_img(image)
