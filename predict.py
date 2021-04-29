import numpy as np
import matplotlib.pyplot as plt
from regression import Regression
from sklearn.metrics import mean_squared_error as mse
from cross_validation import CrossValidation


class Predict:

    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        N = X.shape[0]
        N_train = int(np.round(0.95 * N))
        self.train_x = X[:N_train]
        self.train_y = y[:N_train]
        self.test_x = X[N_train:]
        self.test_y = y[N_train:]
        self.reg = Regression()

    def degree_optimum(self):
        degree = np.arange(1, 5)
        degree_optimal = 1
        min_error = 1e50
        for i in degree:
            cross = CrossValidation(self.model, self.train_x, self.train_y, degree=i)
            error = cross.cross_val_error()
            if error < min_error:
                min_error = error
                degree_optimal = i

        return degree_optimal
    
    def rmse(self, pred, label):

        return np.sqrt(np.sum((pred - label) ** 2) / len(pred))
    
    def prediction(self, degree):
        if self.model == 'linear_regression':
            predicted_y = np.round(self.reg.linear_regression(self.train_x, self.train_y, self.test_x, degree=degree, normalized=False))
        elif self.model == 'ridge_regression':
            predicted_y = np.round(self.reg.ridge_regression(self.train_x, self.train_y, self.test_x, degree=degree, normalized=False))
        elif self.model == 'lasso_regression':
            predicted_y = np.round(self.reg.lasso_regression(self.train_x, self.train_y, self.test_x, degree=degree, normalized=False))

        return predicted_y
    
    