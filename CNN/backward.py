import numpy as np
from CNN.functions import nanargmax


# TODO: refactor: merge convolutionBackward & maxpoolBackward in the same function
def convolutionBackward(dconv_prev, input_image_matrix, kernel_weights_matrix, stride):
    """
    Backpropagation through a convolutional layer.
    """
    (no_channels_out, no_channels_in_from_kernel, kernel_size, _) = kernel_weights_matrix.shape
    (no_channels_in_from_image, source_h, source_w) = input_image_matrix.shape
    # initialize derivatives
    dout = np.zeros(input_image_matrix.shape)
    dfilt = np.zeros(kernel_weights_matrix.shape)
    dbias = np.zeros((no_channels_out, 1))
    param_dict = {"conv_in": input_image_matrix, "dbias": dbias, "dconv_prev": dconv_prev, "dfilt": dfilt, "dout": dout,
                  "f": kernel_size, "filt": kernel_weights_matrix, "n_c": no_channels_out, "orig_dim_h": source_h,
                  "orig_dim_w": source_w, "s": stride}
    dict_out = apply_bkwd(param_dict, do_one_conv_bkwd)
    dout, dfilt, dbias = dict_out["dout"], dict_out["dfilt"], dict_out["dbias"]
    return dout, dfilt, dbias


# TODO: refactor: merge convolutionBackward & maxpoolBackward in the same function
def maxpoolBackward(dpool, input_image_matrix, kernel_size, stride):
    """
    Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the
    original maxpooling during the forward step.
    """
    (no_channels, source_h, source_w) = input_image_matrix.shape

    dout = np.zeros(input_image_matrix.shape)

    dict_in = {"dout": dout, "dconv_prev": dpool, "f": kernel_size, "n_c": no_channels, "conv_in": input_image_matrix,
               "orig_dim_h": source_h, "orig_dim_w": source_w, "s": stride, "dfilt": None, "dbias": None}
    dict_out = apply_bkwd(dict_in, do_one_pool_bkwd)
    dout, dfilt, dbias = dict_out["dout"], dict_out["dfilt"], dict_out["dbias"]
    return dout, dfilt, dbias


def apply_bkwd(param_dict, bkwd_op):
    """
    Parametrized gradient computation execution body that applies a gradient extraction function over the whole input.
    """
    kernel_size = param_dict["f"]
    orig_dim_h = param_dict["orig_dim_h"]
    orig_dim_w = param_dict["orig_dim_w"]
    stride = param_dict["s"]

    for curr_c in range(param_dict["n_c"]):
        # loop through all filters
        curr_y = out_y = 0
        while curr_y + kernel_size <= orig_dim_h:
            curr_x = out_x = 0
            while curr_x + kernel_size <= orig_dim_w:
                param_dict["dout"], param_dict["dfilt"] = bkwd_op(curr_x, curr_y, curr_c, kernel_size, out_x, out_y,
                                                                  param_dict)
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1
        # loss gradient of the bias
        if param_dict["dbias"] is not None:
            param_dict["dbias"][curr_c] = np.sum(param_dict["dconv_prev"][curr_c])
    return param_dict


def do_one_conv_bkwd(curr_x, curr_y, curr_c, kernel_size, out_x, out_y, param_dict):
    """
    Parametrized function that extracts the gradient of one convolution considering a certain window in the input.
    """
    dconv_prev = param_dict["dconv_prev"]
    input_image_matrix = param_dict["conv_in"]
    dfilt = param_dict["dfilt"]
    kernel_weights_matrix = param_dict["filt"]
    dout = param_dict["dout"]
    # loss gradient of filter (used to update the filter)
    dfilt[curr_c] += dconv_prev[curr_c, out_y, out_x] * \
        input_image_matrix[:, curr_y:curr_y + kernel_size, curr_x:curr_x + kernel_size]
    # loss gradient of the input to the convolution operation (conv1 in the case of this network)
    dout[:, curr_y:curr_y + kernel_size, curr_x:curr_x + kernel_size] += dconv_prev[curr_c, out_y, out_x] * \
        kernel_weights_matrix[curr_c]
    return dout, dfilt


def do_one_pool_bkwd(curr_x, curr_y, curr_c, kernel_size, out_x, out_y, param_dict):
    """
    Parametrized function that extracts the gradient of one max pooling considering a certain window in the input.
    """
    dconv_prev = param_dict["dconv_prev"]
    input_image_matrix = param_dict["conv_in"]
    dout = param_dict["dout"]
    # obtain index of largest value in input for current window
    (a, b) = nanargmax(input_image_matrix[curr_c, curr_y:curr_y + kernel_size, curr_x:curr_x + kernel_size])
    dout[curr_c, curr_y + a, curr_x + b] = dconv_prev[curr_c, out_y, out_x]
    return dout, None
