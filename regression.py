import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error as mse


class Regression:

    def normalize(self, y):

        return y / (np.max(y) - np.min(y))

    def linear_regression(self, x_train, y_train, x_test, degree, normalized=False):
        if normalized:
            y_train = self.normalize(y_train)
        model = make_pipeline(PolynomialFeatures(degree), linear_model.LinearRegression())
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)

        return prediction

    def ridge_regression(self, x_train, y_train, x_test, degree, normalized=False):
        if normalized:
            y_train = y_train / (np.max(y_train) - np.min(y_train))
        model = make_pipeline(PolynomialFeatures(degree), linear_model.Ridge())
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)

        return prediction
    
    def lasso_regression(self, x_train, y_train, x_test, degree, normalized=False):
        if normalized:
            y_train = y_train / (np.max(y_train) - np.min(y_train))
        model = make_pipeline(PolynomialFeatures(degree), linear_model.Lasso(alpha=0.1))
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)

        return prediction

   


