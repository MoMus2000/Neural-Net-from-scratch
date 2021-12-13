import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

EPOCHS = 100

def to_categorical(y, num_classes, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

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
        for e in range(0, EPOCHS):
            print(f"Epoch {e} of {EPOCHS}")
            loss = 0
            for x, y in zip(X, Y):
                out = self.predict(x)

                loss += mse(y, out)

                grad = mse_prime(y, out)

                for layer in reversed(self.network):
                    grad = layer.back_prop(grad, 0.05)

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

def loading_data(df):
    X = df.drop(labels=['label'], axis=1)
    X = np.array(X)
    X = X.reshape(X.shape[0] ,28*28, 1)
    y = np.array(df['label'])
    y = to_categorical(y, 10)
    y = y.reshape(y.shape[0], 10, 1)
    return X, y


if __name__ == "__main__":
    # path = "/Users/a./Desktop/Neural-Net-from-scratch/train.csv"
    # df = pd.read_csv(path)
    # X, y = loading_data(df)
    # model = NN()
    # model.add(Dense(28*28, 128))
    # model.add(Tanh())
    # model.add(Dense(128, 64))
    # model.add(Tanh())
    # model.add(Dense(64, 10))
    # model.train(X, y)
    # X = np.reshape([[0, 1], [0, 1], [1, 0], [1, 1], [0,0]], (5, 2, 1))
    # Y = np.reshape([[0], [0], [0], [1], [0]], (5, 1, 1))
    X, Y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
    # plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, s=25, edgecolor="k")

    X = np.reshape(X, (len(X), 2, 1))
    Y = np.reshape(Y, (len(X), 1, 1))
    model = NN()
    model.add(Dense(2, 64))
    model.add(Tanh())
    model.add(Dense(64, 32))
    model.add(Tanh())
    model.add(Dense(32, 16))
    model.add(Tanh())
    model.add(Dense(16, 1))
    model.add(Tanh())
    model.train(X, Y)

    results = []
    x_axis = []
    y_axis = []
    acc = 0
    corr = 0
    for x,y in zip(X, Y):
        result = model.predict(x)[0][0]
        results.append(result)
        x_axis.append(x[0][0])
        y_axis.append(y[0][0])
        if result > 0.5 and y[0][0] == 1:
            corr += 1
        elif result < 0.5 and y[0][0] == 0:
            corr+=1


    print(f"Binary Classification Accuracy : {(corr/len(X)) * 100}")
    plt.scatter(x_axis, y_axis, marker="o", s=25, edgecolor="k")
    plt.scatter(x_axis, results, marker="x", s=25, edgecolor="k")
    plt.show()

