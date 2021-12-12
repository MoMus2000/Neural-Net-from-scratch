import numpy as np
import pandas as pd


"""
Creating a neural network from scratch
Steps: 
1. Loading the data
2. Forward propagation
3. Backpropagation
4. Gradient Descent
"""

path = "/Users/a./Downloads/train.csv"

df = pd.read_csv(path)
m, n = df.shape

def initialize_params():
    W1 = np.random.rand(10, 784) - 0.5 # 784 num of nodes in input
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def relu(x):
    return np.maximum(x, 0)

def relu_deriv(x):
    return x > 0

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def one_hot(Y):
    one_hot_y = np.zeros((Y.shape[0], Y.max() + 1))
    one_hot_y[np.arange(Y.size), Y] = 1
    return one_hot_y.T

def loading_data(df):
    df = np.array(df)
    df = df[:m].T
    X = df[1:]/255.
    y = df[0]
    return X, y

def forward_prop(X, W1, b1, W2, b2):
    Z1 = W1.dot(X) + b1
    a1 = relu(Z1)
    Z2 = W2.dot(a1) + b2
    a2 = softmax(Z2)
    return Z1, a1, Z2, a2

def backprop(Z1, a1, Z2, a2, X, y, W1, W2):
    y = one_hot(y)
    dz2 = a2 - y
    dw2 = 1/m * dz2.dot(a1.T)
    db2 = 1/m*np.sum(dz2)

    dz1 = W2.T.dot(dz2)* relu_deriv(Z1)
    dw1 = 1/m * dz1.dot(X.T)
    db1 = 1/m*sum(dz1)

    return dw1, db1, dw2, db2

def update_params(W1, b1, W2, b2, DW1, DW2, DB1, DB2, learning_rate):
    W1 = W1 - learning_rate*DW1
    b1 = b1 - learning_rate*DB1

    W2 = W2 - learning_rate*DW2
    b2 = b2 - learning_rate*DB2

    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(iterations=100):
    W1, b1, W2, b2 = initialize_params()
    X, y = loading_data(df)
    for i in range(0, iterations):
        print(f"training ... {int(i/iterations * 100)}",end="\r")
        Z1, a1, Z2, a2 = forward_prop(X, W1, b1, W2, b2)
        dw1, db1, dw2, db2 = backprop(Z1, a1, Z2, a2 , X, y, W1, W2)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dw1, dw2, db1, db2, 0.1)
        if i % 10 == 0:
            preds = get_predictions(a2)
            print(f"{get_accuracy(preds, y)} accuracy detected on train ....", end="\n")







if __name__ == "__main__":
    gradient_descent()