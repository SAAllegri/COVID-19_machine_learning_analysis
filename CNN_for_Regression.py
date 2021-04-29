import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, AveragePooling2D
from tensorflow.keras.layers import LeakyReLU
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DoCNN:

    def __init__(self, X, y, test_size):
        self.N, self.D = X.shape
        X = X.reshape(self.N, self.D, 1)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size=test_size, shuffle=False)

    def predict(self):
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation="relu", input_shape=(self.D, 1)))
        model.add(Dropout(rate=0.10))
        model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation="relu"))
        model.add(Dropout(rate=0.10))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        model.fit(self.train_X, self.train_y, batch_size=12, epochs=200, verbose=0)

        return np.squeeze(model.predict(self.test_X))

