import numpy as np


def convolution(input_image_matrix, kernel_weights_matrix, bias, stride=1):
    """
    Convolve `kernel_weights_matrix` over `input_image_matrix` using stride `stride`
    """
    (no_channels_out, no_channels_in_from_kernel, kernel_size, _) = kernel_weights_matrix.shape  # filter dimensions
    no_channels_in_from_image, initial_h, initial_w = input_image_matrix.shape  # image dimensions
    output_h = int((initial_h - kernel_size) / stride) + 1  # calculate output dimensions
    output_w = int((initial_w - kernel_size) / stride) + 1
    assert(no_channels_in_from_image == no_channels_in_from_kernel,
           "Filter and Image dimension mismatch. Expected {} but instead got {}."
           .format(no_channels_in_from_image, no_channels_in_from_kernel))
    output_matrix = np.zeros((no_channels_out, output_h, output_w))

    # convolve the filter over every part of the image, adding the bias at each step.
    param_dict = {"f": kernel_size, "filt": kernel_weights_matrix, "s_x": stride, "s_y": stride,
                  "image": input_image_matrix, "in_dim_w": initial_w, "in_dim_h": initial_h, "n_c": no_channels_out,
                  "out": output_matrix, "bias": bias}
    output_matrix = apply(param_dict, do_one_conv)

    return output_matrix


def maxpool(input_image_matrix, kernel_size=2, stride=2):
    """
    Downsample `image` using kernel size `f` and stride `s`
    """
    no_channels, initial_h, initial_w = input_image_matrix.shape
    output_h = int((initial_h - kernel_size) / stride) + 1
    output_w = int((initial_w - kernel_size) / stride) + 1

    output_matrix = np.zeros((no_channels, output_h, output_w))
    param_dict = {"f": kernel_size, "filt": None, "s_x": stride, "s_y": stride, "image": input_image_matrix,
                  "in_dim_w": initial_w,
                  "in_dim_h": initial_h, "n_c": no_channels, "out": output_matrix, "bias": None}
    output_matrix = apply(param_dict, do_one_pool)
    return output_matrix


def apply(param_dict, function):
    """
    Parametrized convolution execution body that applies a function (sliding window) over the whole input matrix.
    """
    kernel_size = param_dict["f"]
    for curr_c in range(param_dict["n_c"]):
        curr_y = out_y = 0
        while curr_y + kernel_size <= param_dict["in_dim_h"]:
            curr_x = out_x = 0
            while curr_x + kernel_size <= param_dict["in_dim_w"]:
                param_dict["out"] = function(curr_x, curr_y, curr_c, out_x, out_y, param_dict)
                curr_x += param_dict["s_x"]
                out_x += 1
            curr_y += param_dict["s_y"]
            out_y += 1
    return param_dict["out"]


def do_one_conv(curr_x, curr_y, curr_channels, out_x, out_y, param_dict):
    """
    Parametrized function that executes one convolution over a certain window in the input matrix.
    """
    kernel_size = param_dict["f"]

    param_dict["out"][curr_channels, out_y, out_x] = \
        np.sum(param_dict["filt"][curr_channels] *
               param_dict["image"][:, curr_y:curr_y + kernel_size, curr_x:curr_x + kernel_size]) + \
        param_dict["bias"][curr_channels]
    return param_dict["out"]


def do_one_pool(curr_x, curr_y, curr_channels, out_x, out_y, param_dict):
    """
    Parametrized function that executes one max pooling over a certain window in the input matrix.
    """
    kernel_size = param_dict["f"]
    param_dict["out"][curr_channels, out_y, out_x] = np.max(
        param_dict["image"][curr_channels, curr_y:curr_y + kernel_size, curr_x:curr_x + kernel_size])
    return param_dict["out"]
