from numpy import *

from sigmoid import *


def pack_teta(t1, t2):
    return concatenate((t1.flatten(), t2.flatten()))


def unpack_teta(t_pack):
    t1 = reshape(t_pack[:25 * 401], (25, 401))
    t2 = reshape(t_pack[25 * 401:], (10, 26))
    return t1, t2


def gradient(t_pack, x, y, m, l):
    t1, t2 = unpack_teta(t_pack)

    a1 = insert(x, 0, 1, axis=1)
    z2 = dot(a1, t1.T)
    a2 = sigmoid(z2)
    a2 = insert(a2, 0, 1, axis=1)
    z3 = dot(a2, t2.T)
    h = sigmoid(z3)

    delta3 = h - y
    delta2 = dot(delta3, t2[:, 1:]) * sigmoid_gradient(z2)
    Delta1 = dot(delta2.T, a1) / m
    Delta2 = dot(delta3.T, a2) / m

    Delta1[:, 1:] += (l / m) * t1[:, 1:]
    Delta2[:, 1:] += (l / m) * t2[:, 1:]
    return pack_teta(Delta1, Delta2)