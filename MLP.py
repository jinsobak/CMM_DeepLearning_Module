import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import requests

url = "https://github.com/jinsobak/CMM_DeepLearning_Module/blob/main/MLP/datas/data_mv_sv_dv_ut_lt_hd_test.csv"
response = requests.get(url)

with open("filename.csv", "wb") as f:
    f.write(response.content)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])