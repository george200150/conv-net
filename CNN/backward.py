"""
Description: backpropagation operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018
"""

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
    param_dict = {"conv_in": conv_in, "dbias": dbias, "dconv_prev": dconv_prev, "dfilt": dfilt, "dout": dout,
                  "f": f, "filt": filt, "n_c": n_f, "orig_dim": orig_dim, "s": s}  # TODO: refactor keys... ("n_c": n_f)
    dict_out = conv_bkwd(param_dict, conv_bkwd_xtract)
    dout, dfilt, dbias = dict_out["dout"], dict_out["dfilt"], dict_out["dbias"]
    return dout, dfilt, dbias


def maxpoolBackward(dpool, orig, f, s):
    """
    Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the original maxpooling during the forward step.
    """
    (n_c, orig_dim, _) = orig.shape

    dout = np.zeros(orig.shape)

    dict_in = {"dout": dout, "dconv_prev": dpool, "f": f, "n_c": n_c, "conv_in": orig, "orig_dim": orig_dim, "s": s, "dfilt": None, "dbias": None}
    dict_out = pool_bkwd(dict_in, poo_bkwd_xtract)
    dout, dfilt, dbias = dict_out["dout"], dict_out["dfilt"], dict_out["dbias"]
    return dout, dfilt, dbias


def conv_bkwd(param_dict, bkwd_op):
    f = param_dict["f"]
    orig_dim = param_dict["orig_dim"]
    s = param_dict["s"]

    for curr_c in range(param_dict["n_c"]):
        # loop through all filters
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                param_dict["dout"], param_dict["dfilt"] = bkwd_op(curr_c, curr_x, curr_y, f, out_x, out_y, param_dict)
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        # loss gradient of the bias
        if param_dict["dbias"] is not None:
            param_dict["dbias"][curr_c] = np.sum(param_dict["dconv_prev"][curr_c])
    # dict_out = {"dout": param_dict["dout"], "dfilt": param_dict["dfilt"], "dbias": param_dict["dbias"]}
    return param_dict


def pool_bkwd(param_dict, bkwd_op):
    f = param_dict["f"]
    orig_dim = param_dict["orig_dim"]
    s = param_dict["s"]

    for curr_c in range(param_dict["n_c"]):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                param_dict["dout"], param_dict["dfilt"] = bkwd_op(curr_c, curr_x, curr_y, f, out_x, out_y, param_dict)
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        if param_dict["dbias"] is not None:
            param_dict["dbias"][curr_c] = np.sum(param_dict["dconv_prev"][curr_c])
    # dict_out = {"dout": param_dict["dout"], "dfilt": param_dict["dfilt"], "dbias": param_dict["dbias"]}
    return param_dict


def conv_bkwd_xtract(curr_c, curr_x, curr_y, f, out_x, out_y, param_dict):
    dconv_prev = param_dict["dconv_prev"]
    conv_in = param_dict["conv_in"]
    dfilt = param_dict["dfilt"]
    filt = param_dict["filt"]
    dout = param_dict["dout"]

    # loss gradient of filter (used to update the filter)
    dfilt[curr_c] += dconv_prev[curr_c, out_y, out_x] * conv_in[:, curr_y:curr_y + f, curr_x:curr_x + f]
    # loss gradient of the input to the convolution operation (conv1 in the case of this network)
    dout[:, curr_y:curr_y + f, curr_x:curr_x + f] += dconv_prev[curr_c, out_y, out_x] * filt[curr_c]
    return dout, dfilt


def poo_bkwd_xtract(curr_c, curr_x, curr_y, f, out_x, out_y, param_dict):
    dconv_prev = param_dict["dconv_prev"]
    conv_in = param_dict["conv_in"]
    dout = param_dict["dout"]
    # obtain index of largest value in input for current window
    (a, b) = nanargmax(conv_in[curr_c, curr_y:curr_y + f, curr_x:curr_x + f])
    dout[curr_c, curr_y + a, curr_x + b] = dconv_prev[curr_c, out_y, out_x]
    return dout, None
