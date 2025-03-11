import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.1, epoch=1000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.weight = None
        self.bias = None

    def sigmiod(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_test, y_pred):
        return -np.mean(y_test * np.log(y_pred) + (1 - y_test) * (1 - np.log(y_pred)))

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epoch):
            linear_model = np.dot(x, self.weight) + self.bias
            y_predicted = self.sigmiod(linear_model)

            dw = (1 / n_samples) * np.dot(x.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, x):
        linear_model = np.dot(x, self.weight) + self.bias
        y_predict = self.sigmiod(linear_model)
        return (y_predict >= 0.5).astype(int)

