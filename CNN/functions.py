import numpy as np


class Activation:
    def __init__(self, fwd_act, bkw_act):
        self.forward_activation = fwd_act
        self.backward_activation = bkw_act

    def activation(self, x):
        return self.forward_activation(x)

    def backprop_activation(self, dx, x):
        return self.backward_activation(dx, x)


def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs


def _back_idx(dx, x):
    return dx


def _back_relu(dx, x):
    dx[x <= 0] = 0
    return dx


def _idx(x):
    return x


def _relu(x):
    x[x <= 0] = 0
    return x


def _softmax(x):
    out = np.exp(x)
    return out / np.sum(out)


def categoricalCrossEntropy(probs, label):
    return -np.sum(label * np.log(probs))


idx = Activation(_idx, _back_idx)
relu = Activation(_relu, _back_relu)
softmax = Activation(_softmax, None)
