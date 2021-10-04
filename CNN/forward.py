"""
Description: forward operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018
"""
import numpy as np


#####################################################
################ Forward Operations #################
#####################################################
def convolution(image, filt, bias, s=1):  # TODO: refactor: merge convolution & maxpool in the same function
    """
    Confolves `filt` over `image` using stride `s`
    """
    (n_f, n_c_f, f, _) = filt.shape  # filter dimensions
    n_c, in_dim, _ = image.shape  # image dimensions
    out_dim = int((in_dim - f) / s) + 1  # calculate output dimensions
    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"
    out = np.zeros((n_f, out_dim, out_dim))

    # convolve the filter over every part of the image, adding the bias at each step.
    param_dict = {"f": f, "filt": filt, "s_x": s, "s_y": s, "image": image, "in_dim_w": in_dim, "in_dim_h": in_dim, "n_c": n_f, "out": out, "bias": bias}
    out = apply(param_dict, one_conv)

    return out


def maxpool(image, f=2, s=2, filt=None, bias=None):  # I assume f==size and s==stride  # TODO: refactor: merge convolution & maxpool in the same function
    """
    Downsample `image` using kernel size `f` and stride `s`
    """
    n_c, h_prev, w_prev = image.shape
    h = int((h_prev - f) / s) + 1
    w = int((w_prev - f) / s) + 1

    downsampled = np.zeros((n_c, h, w))
    param_dict = {"f": f, "filt": None, "s_x": s, "s_y": s, "image": image, "in_dim_w": w_prev, "in_dim_h": h_prev,
                  "n_c": n_c, "out": downsampled, "bias": None}
    downsampled = apply(param_dict, one_pool)
    return downsampled


def apply(param_dict, function):
    f = param_dict["f"]
    for curr_c in range(param_dict["n_c"]):
        curr_y = out_y = 0
        while curr_y + f <= param_dict["in_dim_h"]:
            curr_x = out_x = 0
            while curr_x + f <= param_dict["in_dim_w"]:
                param_dict["out"] = function(curr_x, curr_y, curr_c, out_x, out_y, param_dict)
                curr_x += param_dict["s_x"]
                out_x += 1
            curr_y += param_dict["s_y"]
            out_y += 1
    return param_dict["out"]


def one_conv(curr_x, curr_y, curr_c, out_x, out_y, param_dict):
    f = param_dict["f"]

    param_dict["out"][curr_c, out_y, out_x] = np.sum(param_dict["filt"][curr_c] * param_dict["image"][:, curr_y:curr_y + f, curr_x:curr_x + f]) + param_dict["bias"][curr_c]
    return param_dict["out"]


def one_pool(curr_x, curr_y, curr_c, out_x, out_y, param_dict):  # filter and bias are not used.
    f = param_dict["f"]
    param_dict["out"][curr_c, out_y, out_x] = np.max(param_dict["image"][curr_c, curr_y:curr_y + f, curr_x:curr_x + f])
    return param_dict["out"]


def softmax(x):
    out = np.exp(x)
    return out / np.sum(out)


def categoricalCrossEntropy(probs, label):
    return -np.sum(label * np.log(probs))
