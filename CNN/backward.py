"""
Description: backpropagation operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018
"""

import numpy as np

from CNN.utils import *


#####################################################
############### Backward Operations #################
#####################################################

def convolutionBackward(dconv_prev, conv_in, filt, s):
    """
    Backpropagation through a convolutional layer.
    """
    (n_f, n_c, f, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape
    # initialize derivatives
    dout = np.zeros(conv_in.shape)
    dfilt = np.zeros(filt.shape)
    dbias = np.zeros((n_f, 1))
    dout, dfilt, dbias = conv_bkwd(conv_in, dbias, dconv_prev, dfilt, dout, f, filt, n_f, orig_dim, s)
    return dout, dfilt, dbias

def maxpoolBackward(dpool, orig, f, s):
    """
    Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the original maxpooling during the forward step.
    """
    (n_c, orig_dim, _) = orig.shape

    dout = np.zeros(orig.shape)

    dout = pool_bkwd(dout, dpool, f, n_c, orig, orig_dim, s)
    return dout


def conv_bkwd(conv_in, dbias, dconv_prev, dfilt, dout, f, filt, n_f, orig_dim, s):
    for curr_f in range(n_f):
        # loop through all filters
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                dfilt, dout = conv_bkwd_xtract(conv_in, curr_f, curr_x, curr_y, dconv_prev, dfilt, dout, f, filt, out_x, out_y)
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        # loss gradient of the bias
        dbias[curr_f] = np.sum(dconv_prev[curr_f])
    return dout, dfilt, dbias

def pool_bkwd(dout, dpool, f, n_c, orig, orig_dim, s):
    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                dout = poo_bkwd_xtract(curr_c, curr_x, curr_y, dout, dpool, f, orig, out_x, out_y)
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return dout


def conv_bkwd_xtract(conv_in, curr_f, curr_x, curr_y, dconv_prev, dfilt, dout, f, filt, out_x, out_y):  # TODO: for easier i/o formatting, use dictionaries in the future
    # loss gradient of filter (used to update the filter)
    dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y + f, curr_x:curr_x + f]
    # loss gradient of the input to the convolution operation (conv1 in the case of this network)
    dout[:, curr_y:curr_y + f, curr_x:curr_x + f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f]
    return dfilt, dout

def poo_bkwd_xtract(curr_c, curr_x, curr_y, dout, dpool, f, orig, out_x, out_y):
    # obtain index of largest value in input for current window
    (a, b) = nanargmax(orig[curr_c, curr_y:curr_y + f, curr_x:curr_x + f])
    dout[curr_c, curr_y + a, curr_x + b] = dpool[curr_c, out_y, out_x]
    return dout
