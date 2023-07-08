from numpy import *
from sigmoid import *


def predict(x, theta1, theta2):
    x = insert(x, 0, 1, axis=1)
    a2 = sigmoid(dot(x, transpose(theta1)))
    h = sigmoid(dot(insert(a2, 0, 1, axis=1), transpose(theta2)))
    return argmax(h, axis=1) + 1
def predict_noArgmax(x, theta1, theta2):
    x = insert(x, 0, 1, axis=1)
    a2 = sigmoid(dot(x, transpose(theta1)))
    return sigmoid(dot(insert(a2, 0, 1, axis=1), transpose(theta2)))