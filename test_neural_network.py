import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load data
def load_data():
    train_data = np.array(pd.read_csv("dataset/mnist_train.csv", nrows=1000))
    test_data = np.array(pd.read_csv("dataset/mnist_test.csv", nrows=1000))

    x_train = train_data[:, 1:] / 255
    x_train = x_train.T
    y_train = train_data[:, 0]
    y_train = y_train.reshape(y_train.shape[0], 1)

    x_test = test_data[:, 1:] / 255
    x_test = x_test.T
    y_test = test_data[:, 0]
    y_test = y_test.reshape(y_test.shape[0], 1)

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

w1 = np.load("weights_and_bias/w1.npy")
b1 = np.load("weights_and_bias/b1.npy")
w2 = np.load("weights_and_bias/w2.npy")
b2 = np.load("weights_and_bias/b2.npy")



def ReLU(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def get_predictions(a2):
    prediction = np.argmax(a2, axis=0)
    prediction = prediction.reshape(prediction.shape[0], 1)
    return prediction

z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x_test)

prediction = get_predictions(a2)



#visualize the 10 first images
for i in range(0, 20):
    plt.imshow(x_test.T[i].reshape(28, 28), cmap="Greys")
    plt.title(f"prediction: {prediction[i]}    actual number: {y_test[i]}")
    plt.show()