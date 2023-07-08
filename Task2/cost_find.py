from numpy import *
from predict import *
from functions import *

def cost(h, y, m, t1, t2, l):
    J = sum(- (y * log(h)) - ((1 - y) * log(1 - h))) / m
    t = sum(t1 ** 2) + sum(t2 ** 2)
    return J + t * (l / (2 * m))


def cost_alt(t_pack, x, y, m, l):
    t1, t2 = unpack_teta(t_pack)
    h = predict_noArgmax(x, t1, t2)
    return cost(h, y, m, t1, t2, l)


