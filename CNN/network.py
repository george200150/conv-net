from collections import OrderedDict

import numpy as np

from CNN.backward import convolutionBackward, maxpoolBackward
from CNN.callbacks import DumpModelPickleCallback
from CNN.forward import convolution, maxpool
from CNN.functions import categoricalCrossEntropy, softmax, relu, idx
from CNN.optimizer import AdamOptimizer
from CNN.utils import initializeFilter, initializeWeight


class SequentialModel:
    def __init__(self):
        self.layers = OrderedDict()
        self.optimizer = None

    def no_layers(self):
        return len(self.layers.keys())

    def add(self, layer):
        layer_hash = layer.name() + str(len(self.layers) + 1)
        self.layers[layer_hash] = layer

    def params(self):
        no_layers = len(self.layers)
        parameter_list = [[] for _ in range(no_layers * 2)]
        for indx, layer in enumerate(self.layers.values()):
            w, b = layer.params()
            parameter_list[indx] = w
            parameter_list[indx + no_layers] = b
        return parameter_list

    def load_model(self, parameter_list):
        self.set_params(parameter_list)

    def full_forward(self, x):
        outputs = [x]  # first, insert the input image itself
        for layer in self.layers.values():
            x = layer.forward(x)
            outputs.append(x)
        return outputs

    def full_backprop(self, probs, label, feed_results):
        layers = self.layers.values()
        no_layers = len(layers)
        prev_layer = None
        for layer in layers:  # tell backprop what activations to use
            if prev_layer is not None:
                layer.set_backward_activation(prev_layer.activation)
            prev_layer = layer

        dout = probs - label  # derivative of loss w.r.t. final layer output
        grads_weights = [0 for _ in range(no_layers)]
        grads_biases = [0 for _ in range(no_layers)]
        for indx, layer in enumerate(reversed(layers)):
            res_index = no_layers - 1 - indx
            dout, grads_weights[res_index], grads_biases[res_index] = layer.backprop(dout, feed_results[res_index])
        return grads_weights, grads_biases

    def set_params(self, parameter_list):
        no_layers = len(self.layers)
        for indx, layer in enumerate(self.layers.values()):
            layer.set_weights(parameter_list[indx])
            layer.set_biases(parameter_list[indx + no_layers])

    def set_optimizer(self, opt):
        self.optimizer = opt

    def train(self, dataloader):
        return self.optimizer.train(self, dataloader)


class Layer:
    def name(self):
        raise NotImplementedError("abstract layer")

    def __init__(self, out_channels=None, in_channels=None):
        self.output_dimension = out_channels
        self.input_dimension = in_channels
        self.weights = None
        self.biases = None
        self.activation = idx
        self.back_activation = idx

    def params(self):
        return self.weights, self.biases

    def set_weights(self, w):
        self.weights = w

    def set_biases(self, b):
        self.biases = b

    def set_activation(self, fct):
        self.activation = fct

    def set_backward_activation(self, fct):
        self.back_activation = fct

    def forward(self, x):
        raise NotImplementedError("inference")

    def backprop(self, dx, x):
        raise NotImplementedError("gradient computation")


class Conv2D(Layer):
    def name(self):
        return "Conv2D"

    def __init__(self, out_channels, in_channels, kernel, stride):
        super(Conv2D, self).__init__(out_channels=out_channels, in_channels=in_channels)
        self.kernel_dimension = kernel
        self.stride = stride
        w_shape = (out_channels, in_channels, kernel[0], kernel[1])
        self.weights = initializeFilter(w_shape)
        self.biases = np.zeros((self.weights.shape[0], 1))

    def forward(self, x):
        x = convolution(x, self.weights, self.biases, stride=self.stride)
        x = self.activation.activation(x)
        return x

    def backprop(self, dx, x):
        # backpropagate previous gradient through second convolutional layer.
        dx, d_weights, d_bias = convolutionBackward(dx, x, self.weights, stride=1)

        dx = self.back_activation.backprop_activation(dx, x)
        return dx, d_weights, d_bias


class MaxPool(Layer):
    def name(self):
        return "MaxPool"

    def __init__(self, kernel):
        super(MaxPool, self).__init__()
        self.kernel_dimension = kernel

    def forward(self, x):
        x = maxpool(x, self.kernel_dimension[0], self.kernel_dimension[1])
        x = self.activation.activation(x)
        return x

    def backprop(self, dx, x):
        # backprop through the max-pooling layer(only neurons with highest activation in window get updated)
        dx, _, _ = maxpoolBackward(dx, x, self.kernel_dimension[0], self.kernel_dimension[1])

        dx = self.back_activation.backprop_activation(dx, x)
        return dx, None, None


class Flatten(Layer):
    def name(self):
        return "Flatten"

    def __init__(self):
        super(Flatten, self).__init__()
        self.original_shape = None

    def forward(self, x):
        (nf2, dim2_x, dim2_y) = x.shape
        self.original_shape = x.shape
        flat_x = x.reshape((nf2 * dim2_x * dim2_y, 1))
        flat_x = self.activation.activation(flat_x)
        return flat_x

    def backprop(self, dx, x):
        dx = dx.reshape(self.original_shape)  # reshape fully connected into dimensions of pooling layer
        dx = self.activation.backprop_activation(dx, x)
        return dx, None, None


class Dense(Layer):
    def name(self):
        return "Dense"

    def __init__(self, out_channels, in_channels):
        super(Dense, self).__init__(out_channels=out_channels, in_channels=in_channels)
        self.output_dimension = out_channels
        self.input_dimension = in_channels
        w_shape = (out_channels, in_channels)
        self.weights = initializeWeight(w_shape)
        self.biases = np.zeros((self.weights.shape[0], 1))

    def forward(self, x):
        x = self.weights.dot(x) + self.biases
        x = self.activation.activation(x)
        return x

    def backprop(self, dx, x):
        d_weight = dx.dot(x.T)  # loss gradient of final dense layer weights
        d_biases = np.sum(dx, axis=1).reshape(self.biases.shape)  # loss gradient of final dense layer biases
        dx = self.weights.T.dot(dx)  # loss gradient of first dense layer outputs

        dx = self.back_activation.backprop_activation(dx, x)
        return dx, d_weight, d_biases


def build_model(num_classes=10, img_depth=1, img_dim_x=28, img_dim_y=28, f=5, num_filt1=8, num_filt2=8, save_path='params.pkl'):
    model = SequentialModel()
    # Initializing all the parameters

    # INITIALIZE CONV & FC LAYERS WEIGHTS (co, ci, kh, kw) & BIASES
    conv_1 = Conv2D(out_channels=num_filt1, in_channels=img_depth, kernel=(f, f), stride=1)
    conv_1.set_activation(relu)
    model.add(conv_1)

    conv_2 = Conv2D(out_channels=num_filt2, in_channels=num_filt1, kernel=(f, f), stride=1)
    conv_2.set_activation(relu)
    model.add(conv_2)

    pooled = MaxPool(kernel=(2, 2))
    pooled.set_activation(idx)
    model.add(pooled)

    flatten = Flatten()
    flatten.set_activation(idx)
    model.add(flatten)

    dense3 = Dense(out_channels=128, in_channels=800)
    dense3.set_activation(relu)
    model.add(dense3)

    dense4 = Dense(out_channels=num_classes, in_channels=128)
    dense4.set_activation(softmax)
    model.add(dense4)

    optimizer = AdamOptimizer(num_classes=num_classes, img_depth=img_depth, img_dim_x=img_dim_x, img_dim_y=img_dim_y)
    optimizer.set_loss(categoricalCrossEntropy)
    optimizer.addCallbacks({DumpModelPickleCallback.get_name(): DumpModelPickleCallback(save_path=save_path)})
    optimizer.setFrequency(1)  # execute callbacks every n-th epoch

    model.set_optimizer(optimizer)

    return model


def build_model_DIODE(num_classes=2, img_depth=3, img_dim_x=512, img_dim_y=384, f=5, num_filt1=8, num_filt2=16, batch_size=32, save_path='params_DIODE.pkl'):
    num_filt3 = 32
    num_filt4 = 64
    num_filt5 = 128
    num_filt6 = 256

    model = SequentialModel()
    # Initializing all the parameters

    # INITIALIZE CONV & FC LAYERS WEIGHTS (co, ci, kh, kw) & BIASES
    # TODO: stride problems - must set stride in the same manner as back_activation
    conv_1 = Conv2D(out_channels=num_filt1, in_channels=img_depth, kernel=(f -2 , f - 2), stride=1)  # 3 x 3
    conv_1.set_activation(relu)
    model.add(conv_1)

    pooled_1 = MaxPool(kernel=(2, 2))
    pooled_1.set_activation(idx)
    model.add(pooled_1)

    conv_2 = Conv2D(out_channels=num_filt2, in_channels=num_filt1, kernel=(f - 2, f - 2), stride=1)  # 3 x 3
    conv_2.set_activation(relu)
    model.add(conv_2)

    pooled_2 = MaxPool(kernel=(2, 2))
    pooled_2.set_activation(idx)
    model.add(pooled_2)

    conv_3 = Conv2D(out_channels=num_filt3, in_channels=num_filt2, kernel=(f - 2, f - 2), stride=1)  # 3 x 3
    conv_3.set_activation(relu)
    model.add(conv_3)

    pooled_3 = MaxPool(kernel=(2, 2))
    pooled_3.set_activation(idx)
    model.add(pooled_3)

    # conv_4 = Conv2D(out_channels=num_filt4, in_channels=num_filt3, kernel=(f - 2, f - 2), stride=1)  # 3 x 3
    # conv_4.set_activation(relu)
    # model.add(conv_4)
    #
    # pooled_4 = MaxPool(kernel=(2, 2))
    # pooled_4.set_activation(idx)
    # model.add(pooled_4)

    # conv_5 = Conv2D(out_channels=num_filt5, in_channels=num_filt4, kernel=(f - 2, f - 2), stride=1)  # 3 x 3
    # conv_5.set_activation(relu)
    # model.add(conv_5)
    # 
    # pooled_5 = MaxPool(kernel=(2, 2))
    # pooled_5.set_activation(idx)
    # model.add(pooled_5)
    # 
    # conv_6 = Conv2D(out_channels=num_filt6, in_channels=num_filt5, kernel=(f - 2, f - 2), stride=1)  # 3 x 3
    # conv_6.set_activation(relu)
    # model.add(conv_6)
    # 
    # pooled_6 = MaxPool(kernel=(2, 2))
    # pooled_6.set_activation(idx)
    # model.add(pooled_6)

    flatten = Flatten()
    flatten.set_activation(idx)
    model.add(flatten)

    # dense5 = Dense(out_channels=512, in_channels=8960)
    dense5 = Dense(out_channels=64, in_channels=768)  # shrunk images
    dense5.set_activation(relu)
    model.add(dense5)

    # dense6 = Dense(out_channels=num_classes, in_channels=512)
    dense6 = Dense(out_channels=num_classes, in_channels=64)
    dense6.set_activation(softmax)
    model.add(dense6)

    optimizer = AdamOptimizer(num_classes=num_classes, img_depth=img_depth, img_dim_x=img_dim_x, img_dim_y=img_dim_y, batch_size=batch_size)
    optimizer.set_loss(categoricalCrossEntropy)
    optimizer.addCallbacks({DumpModelPickleCallback.get_name(): DumpModelPickleCallback(save_path=save_path)})
    optimizer.setFrequency(1)  # execute callbacks every n-th epoch
    model.set_optimizer(optimizer)
    return model


def train(model, dataloader):
    cost = model.train(dataloader)
    return cost
