import numpy as np
import pandas as pd


def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def one_hot(y):
    one_hot_y = np.zeros((y.shape[0], y.max()+1))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y.T

def deriv_relu(x):
    return x > 0

def loading_data(df):
    df = np.array(df)
    df = df[:].T
    X = df[1:]/255.
    y = df[0]
    return X, y

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


class Dense:
    def __init__(self, input_shape, output_shape, activation='relu'):
        self.weight = np.random.randn(output_shape, input_shape)
        self.bias = np.random.randn(output_shape, 1)
        self.activation = activation

    def forward(self, input_val):
        self.input = input_val
        return np.dot(self.weight, self.input) + self.bias

    def back_prop(self, output, learning_rate):
        weights_gradient = np.dot(output, self.input.T)
        input_gradient = np.dot(self.weight.T, output)
        self.weight -= learning_rate * weights_gradient
        self.bias -= learning_rate * output

    def __str__(self):
        return f"weight = {self.weight.shape} bias = {self.bias.shape} activation = {self.activation}"

class NN:
    def __init__(self):
        self.network = []

    def train(self, X, y):
        for e in range(0, 10):
            loss = 0
            for i in range(0, 10):
                out = self.predict(X)

                loss += mse(y, out)

                grad = mse_prime(y, out)

                for layer in reversed(self.network):
                    grad = layer.back_prop(grad, 0.1)

            loss /= len(X)
            print(f"Error: {loss}")




    def add(self, layer):
        self.network.append(layer)
        return

    def gradient_descent(self):
        pass

    def update_params(self):
        pass

    def initialize_params(self):
        pass

    def get_accuracy(self):
        pass

    def predict(self, input_val):
        output = input_val
        for layer in self.network:
            output = layer.forward(output)
        return output


if __name__ == "__main__":
    path = "/Users/a./Desktop/Neural-Net-from-scratch/train.csv"
    df = pd.read_csv(path, nrows=100)
    X, y = loading_data(df)
    y = one_hot(y)
    print(y.shape)
    model = NN()
    model.add(Dense(784, 128))
    model.add(Dense(128, 64))
    model.add(Dense(64, 10))
    model.train(X, y)