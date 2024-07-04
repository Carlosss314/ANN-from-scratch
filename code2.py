import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

#load data
def load_data(number_of_images):
    train_data = np.array(pd.read_csv("dataset/mnist_train.csv", nrows=number_of_images))
    test_data = np.array(pd.read_csv("dataset/mnist_test.csv", nrows=1000))

    x_train = train_data[:, 1:] / 255
    x_train = x_train.T
    y_train = train_data[:, 0]
    y_train = y_train.reshape(y_train.shape[0], 1)

    x_test = test_data[:, 1:] / 255
    x_test = x_test.T
    y_test = test_data[:, 0]
    y_test = y_test.reshape(y_test.shape[0], 1)

    #visualize the 10 first images
    # for i in range(1, 11):
    #     plt.imshow(train_data[:, 1:][i].reshape(28, 28), cmap="Greys")
    #     plt.title(y_train[i])
    #     plt.show()

    return x_train, y_train, x_test, y_test



#deep learning
def init_params():
    w1 = np.random.randn(20, 784)
    b1 = np.random.randn(20, 1)
    w2 = np.random.randn(10, 20)
    b2 = np.random.randn(10, 1)
    return w1, b1, w2, b2

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

def one_hot_encoder(y):
    return LabelBinarizer().fit_transform(y).T

def deriv_ReLU(z):
    return z > 0

def backward_prop(z1, a1, a2, w2, x, y):
    m = y.size
    one_hot_y = one_hot_encoder(y)
    dz2 = a2 - one_hot_y
    dw2 = 1/m * dz2.dot(a1.T)
    db2 = 1/m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * deriv_ReLU(z1)
    dw1 = 1/m * dz1.dot(x.T)
    db1 = 1/m * np.sum(dz1)
    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2


def get_predictions(a2):
    prediction = np.argmax(a2, axis=0)
    prediction = prediction.reshape(prediction.shape[0], 1)
    return prediction

def cost(y, y_dataset):
    m = y.size
    return 1/2*m * sum((y-y_dataset)**2)

def accuracy(predictions, y):
    return np.sum(predictions == y) / y.size


def neural_net(x, y, alpha, iterations):
    w1, b1, w2, b2 = init_params()

    #lists to keep track of cost and accuracy
    train_cost = []
    train_accuracy = []
    test_cost = []
    test_accuracy = []

    for i in tqdm(range(iterations)):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backward_prop(z1, a1, a2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        #add cost and accuracy into lists
        if (i%(iterations/100) == 0):
            train_cost.append(cost(get_predictions(a2), y_train))
            train_accuracy.append(accuracy(get_predictions(a2), y_train))
            test_z1, test_a1, test_z2, test_a2 = forward_prop(w1, b1, w2, b2, x_test)
            test_cost.append(cost(get_predictions(test_a2), y_test))
            test_accuracy.append(accuracy(get_predictions(test_a2), y_test))

    #display cost and accuracy for train_set and test_set
    fig, ax_left = plt.subplots(1, 2)
    fig.set_size_inches(14, 6)
    #cost
    ax_left[0].set_title("COST")
    ax_left[0].plot(train_cost, label="train cost")
    ax_right = ax_left[0].twinx()
    ax_right.plot(test_cost, color="orange", label="test cost")
    ax_left[0].legend()
    ax_right.legend()
    #accuracy
    ax_left[1].set_title("ACCURACY")
    ax_left[1].plot(train_accuracy, label="train accuracy")
    ax_left[1].plot(test_accuracy, label="test accuracy")
    ax_left[1].legend()
    plt.show()

    #print accuracy
    print(train_accuracy[-1])
    print(test_accuracy[-1])

    return w1, b1, w2, b2



x_train, y_train, x_test, y_test = load_data(20000)

w1, b1, w2, b2 = neural_net(x_train, y_train, 0.01, 30000)

#save weights and bias in files
# np.save("weights_and_bias/w1", w1)
# np.save("weights_and_bias/b1", b1)
# np.save("weights_and_bias/w2", w2)
# np.save("weights_and_bias/b2", b2)