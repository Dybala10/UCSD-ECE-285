"""
Linear Regression model
"""

import numpy as np


class Linear(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # Initialize in train
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the linear regression update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights
        # train using gradient descent
        # reshape y_train to (N, self.n_class)
        y_train_encoded = np.zeros((N, self.n_class))
        y_train_encoded[np.arange(N), y_train] = 1

        # train using gradient descent
        for i in range(self.epochs):
            # compute the scores
            scores = np.dot(X_train, self.w.T)

            # compute the gradients
            dW = np.dot(X_train.T, (scores - y_train_encoded)) / N + (self.weight_decay * self.w.T)

            # update the weights
            self.w -= self.lr * dW.T
        return self.w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        y_pred = np.zeros(X_test.shape[1])
        scores = X_test.dot(self.w.T)
        y_pred = np.argmax(scores, axis = 1)
        return y_pred