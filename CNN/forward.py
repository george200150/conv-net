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
def convolution(image, filt, bias, s=1):
    """
    Confolves `filt` over `image` using stride `s`
    """
    (n_f, n_c_f, f, _) = filt.shape  # filter dimensions
    n_c, in_dim, _ = image.shape  # image dimensions

    out_dim = int((in_dim - f) / s) + 1  # calculate output dimensions

    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"

    out = np.zeros((n_f, out_dim, out_dim))

    # convolve the filter over every part of the image, adding the bias at each step.
    out = odd_out(bias, f, filt, image, in_dim, n_f, out, s, one_conv)  # TODO: swapping signature decreases performance - idk y
    # out = odd_out(f, filt, s, image, in_dim, n_c, out, bias, one_conv)

    return out


def odd_out(bias, f, filt, image, in_dim, n_f, out, s, function):  # TODO: swapping signature decreases performance - idk y
# def odd_out(f, filt, s, image, in_dim, n_f, out, bias, function):
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                out = function(f, filt, image, curr_x, curr_y, curr_f, out, out_x, out_y, bias)
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return out


def maxpool(image, f=2, s=2):  # I assume f==size and s==stride
    """
    Downsample `image` using kernel size `f` and stride `s`
    """
    n_c, h_prev, w_prev = image.shape

    h = int((h_prev - f) / s) + 1
    w = int((w_prev - f) / s) + 1

    downsampled = np.zeros((n_c, h, w))
    downsampled = apply(f, None, s, s, image, w_prev, h_prev, n_c, downsampled, None, one_pool)
    return downsampled


def apply(f, filt, stride_x, stride_y, image, in_dim_w, in_dim_h, n_c, out, bias, function):
    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= in_dim_h:
            curr_x = out_x = 0
            while curr_x + f <= in_dim_w:
                out = function(f, filt, image, curr_x, curr_y, curr_c, out, out_x, out_y, bias)
                curr_x += stride_x
                out_x += 1
            curr_y += stride_y
            out_y += 1
    return out


def one_conv(f, filt, image, curr_x, curr_y, curr_f, out, out_x, out_y, bias):  # TODO: for easier i/o formatting, use dictionaries in the future
    out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:, curr_y:curr_y + f, curr_x:curr_x + f]) + \
                                bias[curr_f]
    return out


def one_pool(f, filt, image, curr_x, curr_y, curr_c, out, out_x, out_y, bias):  # filter and bias are not used.
    out[curr_c, out_y, out_x] = np.max(image[curr_c, curr_y:curr_y + f, curr_x:curr_x + f])
    return out


def softmax(x):
    out = np.exp(x)
    return out / np.sum(out)


def categoricalCrossEntropy(probs, label):
    return -np.sum(label * np.log(probs))
