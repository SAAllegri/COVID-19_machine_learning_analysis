import numpy as np
import matplotlib.pyplot as plt
from regression import Regression
from sklearn.metrics import mean_squared_error as mse


class CrossValidation:

    def __init__(self, model, X, y, degree):
        self.X = X
        self.y = y
        self.degree = degree
        self.reg = Regression()
        self.model = model
        
    def rmse(self, pred, label):

        return np.sqrt(np.sum((pred - label) ** 2) / len(pred))
    
    def cross_val_error(self, kfold=10):
        N, D = self.X.shape
        slice_size = int(N / kfold)
        error = np.zeros(kfold)
        for i in range(kfold):
            X_train = np.delete(self.X, slice(i * (slice_size), (i+1) * (slice_size)), axis=0)
            y_train = np.delete(self.y, slice(i * (slice_size), (i+1) * (slice_size)))

            X_test = self.X[i * (slice_size):(i + 1) * (slice_size)]
            y_ground = self.y[i * (slice_size):(i+1) * (slice_size)]   # Ground Label for test data
            if self.model == 'linear_regression':
                y_test = self.reg.linear_regression(X_train, y_train, X_test, degree=self.degree, normalized=False)
            elif self.model == 'ridge_regression':
                y_test = self.reg.ridge_regression(X_train, y_train, X_test, degree=self.degree, normalized=False)
            elif self.model == 'lasso_regression':
                y_test = self.reg.lasso_regression(X_train, y_train, X_test, degree=self.degree, normalized=False)

            error[i] = self.rmse(y_ground, y_test)

        return np.mean(error)
    
    