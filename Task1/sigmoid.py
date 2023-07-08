import math

from numpy import *
def sigmoid(Z):
    # Если Z - numpy массив, то выражение применяется к каждому его элемкнту (маппинг)
    return 1 / (1 + math.e ** -Z)
