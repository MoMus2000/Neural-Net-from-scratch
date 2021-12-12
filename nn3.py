import numpy as np
import pandas as pd


class Activation:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def back_prop(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

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
        return input_gradient

    def __str__(self):
        return f"weight = {self.weight.shape} bias = {self.bias.shape} activation = {self.activation}"

class NN:
    def __init__(self):
        self.network = []

    def train(self, X, Y):
        for e in range(0, 10):
            loss = 0
            for x, y in zip(X, Y):
                out = self.predict(x)

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


from keras.datasets import mnist
from keras.utils import np_utils

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 20)


if __name__ == "__main__":
    path = "/Users/a./Desktop/Neural-Net-from-scratch/train.csv"
    # df = pd.read_csv(path, nrows=1000)
    # X, y = loading_data(df)
    # y = one_hot(y)
    model = NN()
    model.add(Dense(28*28, 128))
    model.add(Tanh())
    model.add(Dense(128, 64))
    model.add(Tanh())
    model.add(Dense(64, 10))
    model.train(x_train, y_train)