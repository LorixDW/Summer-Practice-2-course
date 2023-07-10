import math
from scipy import *

from numpy import *


def genetrate(size, mu=0, sigma=1):
    U1 = random.rand(size)
    U2 = random.rand(size)
    R = sqrt(-2 * log(U1))
    Theta = 2 * pi * U2
    X = R * cos(Theta)
    return X * sigma + mu


def densityFunc(x, mu=0, sigma=1):
    return (1 / (sigma * sqrt(2 * pi))) * exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def distributinFunc(x, mu=0, sigma=1):
    z = (x - mu)/sigma
    return 0.5 * (1 + math.erf(z / sqrt(2)))


def sampleAvg(X):
    return sum(X) / len(X)


def sampleVarience(X):
    avg = sampleAvg(X)
    return (1 / (len(X) - 1)) * sum((X - avg) ** 2)


def sampleStandartDeviation(X):
    return sqrt(sampleVarience(X))


def likelihood(params, data):
    mu, sigma = params[0], params[1]
    return -sum(log(densityFunc(data, mu, sigma)))


def paramsMLE(X):
    return optimize.minimize(likelihood, [0, 1], method='Nelder-Mead', args=(X)).x