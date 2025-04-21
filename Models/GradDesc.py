import numpy as np

np.random.seed(42)


class GradientDesc():
    """ Gradient descending realization """

    def __init__(self, model: np.array, lr: float, num_epoch: int):
        """
        alpha0 - initial model parameters
        lr - learning rate
        num_epoch - number of learning epochs
        """
        self.alpha0 = model
        self.lr = lr
        self.num_epoch = num_epoch

    def gradient(self, y_true: int, y_pred: float, x: np.array) -> np.array:
        """
        Gradient count

        y_true - true_value for x
        y_pred - logit for x from our model prediction
        x - list of object features
        """
        grad = x*(1-y_true)*y_pred - y_true*(1-y_pred)
        return grad

    def update(self, alpha: np.array, gradient: np.array, lr: float) -> np.array:
        """
        Weight update function
        alpha: current model parameters
        gradient: counted gradient
        lr: learning rate
        """
        alpha_new = alpha - lr*gradient
        return alpha_new

    def train(self, x_train: np.array, y_train: np.array):
        """
        Model training function

        x_train - object matrix: [obj, features]
        y_train - list of true values

        """
        def sigmoid(val):
            return 1 / (1 + np.exp(-val))

        alpha = self.alpha0.copy()
        for epo in range(self.num_epoch):
            for i, x in enumerate(x_train):
                x = np.append(x, 1)
                y_pred = sigmoid(x.dot(alpha))
                grad = self.gradient(y_train[i], y_pred, x)
                alpha = self.update(alpha, grad, self.lr)
        return alpha