from collections import OrderedDict

import numpy as np

from CNN.backward import convolutionBackward, maxpoolBackward
from CNN.callbacks import DumpModelPickleCallback
from CNN.dataloader import DataLoader
from CNN.forward import convolution, maxpool
from CNN.functions import categoricalCrossEntropy, softmax, relu, idx
from CNN.optimizer import AdamOptimizer
from CNN.utils import initializeFilter, initializeWeight


#####################################################
##################### Training ######################
#####################################################
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

    def full_forward(self, x):
        outputs = [x]  # first, insert the input image itself
        for indx, layer in enumerate(self.layers.values()):
            x = layer.forward(x)
            outputs.append(x)
        return x, outputs  # TODO: could refactor this, as outputs contains x already

    def full_backprop(self, probs, label, feed_results):
        layers = self.layers.values()
        no_layers = len(layers)
        prev_layer = None
        for layer in layers:  # tell backprop what activations to use
            if prev_layer is not None:
                layer.set_backward_activation(prev_layer.activation)
            prev_layer = layer

        dout = probs - label  # derivative of loss w.r.t. final dense layer output
        grads_weights = [0 for _ in range(no_layers)]
        grads_biases = [0 for _ in range(no_layers)]
        for indx, layer in enumerate(reversed(layers)):
            dout, grads_weights[no_layers - 1 - indx], \
                grads_biases[no_layers - 1 - indx] = layer.backprop(dout, feed_results[no_layers - indx - 1])
        return grads_weights, grads_biases

    def add_grads(self, grads_w, grads_b):
        for indx, layer in enumerate(self.layers.values()):
            w, b = layer.params()
            if w is not None and grads_w[indx] is not None:
                layer.set_weights(w + grads_w[indx])
                layer.set_biases(b + grads_b[indx])

    def set_params(self, parameter_list):
        no_layers = len(self.layers)
        for indx, layer in enumerate(self.layers.values()):
            layer.set_weights(parameter_list[indx])
            layer.set_biases(parameter_list[indx + no_layers])

    def set_optimizer(self, opt):
        self.optimizer = opt

    def train(self, train_data):
        return self.optimizer.train(self, train_data)  # TODO: must see what parameters are needed


class Layer:  # TODO: refactor Layer not to contain kernels (it has no sense for dense)
    def name(self):
        raise NotImplementedError("abstract layer")

    def __init__(self, out_dim=None, in_dim=None, kernel=None):  # TODO: kernel_filter_size, kernel_stride
        self.output_dimension = out_dim
        self.input_dimension = in_dim
        self.kernel_dimension = kernel
        self.weights = None
        self.biases = None
        self.activation = None
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

    def __init__(self, out_dim, in_dim, kernel):
        super(Conv2D, self).__init__(out_dim=out_dim, in_dim=in_dim, kernel=kernel)
        w_shape = (out_dim, in_dim, kernel[0], kernel[1])  # TODO: kernel_filter_size, kernel_stride
        self.weights = initializeFilter(w_shape)
        self.biases = np.zeros((self.weights.shape[0], 1))

    def forward(self, x):
        x = convolution(x, self.weights, self.biases, stride=1)
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
        super(MaxPool, self).__init__(kernel=kernel)

    def forward(self, x):
        x = maxpool(x, self.kernel_dimension[0], self.kernel_dimension[1])
        # TODO: activation ?
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
        (nf2, dim2, _) = x.shape
        self.original_shape = x.shape
        flat_x = x.reshape((nf2 * dim2 * dim2, 1))
        # TODO: activation ?
        return flat_x

    def backprop(self, dx, x):
        dx = dx.reshape(self.original_shape)  # reshape fully connected into dimensions of pooling layer
        dx = self.activation.backprop_activation(dx, x)
        return dx, None, None


# TODO: Layers are now implemented for SINGLE-THREADED execution exclusively (last input's size is stored inside them)
class Dense(Layer):
    def name(self):
        return "Dense"

    def __init__(self, out_dim, in_dim):
        super(Dense, self).__init__(out_dim=out_dim, in_dim=in_dim)
        self.output_dimension = out_dim
        self.input_dimension = in_dim
        w_shape = (out_dim, in_dim)
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


def build_model(num_classes=10, img_dim=28, img_depth=1, f=5, num_filt1=8, num_filt2=8, save_path='params.pkl'):
    model = SequentialModel()
    # Initializing all the parameters

    # INITIALIZE CONV & FC LAYERS WEIGHTS (co, ci, kh, kw) & BIASES
    conv_1 = Conv2D(out_dim=num_filt1, in_dim=img_depth, kernel=(f, f))
    conv_1.set_activation(relu)
    model.add(conv_1)

    conv_2 = Conv2D(out_dim=num_filt2, in_dim=num_filt1, kernel=(f, f))
    conv_2.set_activation(relu)
    model.add(conv_2)

    pooled = MaxPool(kernel=(2, 2))
    pooled.set_activation(idx)
    model.add(pooled)

    flatten = Flatten()
    flatten.set_activation(idx)
    model.add(flatten)

    dense3 = Dense(out_dim=128, in_dim=800)
    dense3.set_activation(relu)
    model.add(dense3)

    dense4 = Dense(out_dim=num_classes, in_dim=128)
    dense4.set_activation(softmax)
    model.add(dense4)

    optimizer = AdamOptimizer(num_classes=num_classes, img_dim=img_dim)
    optimizer.set_loss(categoricalCrossEntropy)
    optimizer.addCallbacks({DumpModelPickleCallback.get_name(): DumpModelPickleCallback(save_path=save_path)})
    optimizer.setFrequency(1)  # execute callbacks every n-th epoch

    model.set_optimizer(optimizer)

    return model


def train(model, img_dim):  # TODO: refactor this to look more appropiate
    # TODO: check how Pytorch uses DataLoader with network
    dataloader = DataLoader(img_dim)
    X, y_dash = dataloader.load_data()

    train_data = np.hstack((X, y_dash))
    np.random.shuffle(train_data)

    cost = model.train(train_data)
    return cost
