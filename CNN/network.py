from collections import OrderedDict

import numpy as np
import pickle
from tqdm import tqdm

from CNN.backward import convolutionBackward, maxpoolBackward
from CNN.forward import convolution, maxpool
from CNN.functions import categoricalCrossEntropy, softmax, relu, idx
from CNN.utils import initializeFilter, initializeWeight, extract_data, extract_labels


def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, model, cost):
    """
    update the parameters through Adam gradient descnet.
    """
    global grads
    X = batch[:, 0:-1]  # get batch inputs
    X = X.reshape(len(batch), n_c, dim, dim)  # TODO: change with (dim_x, dim_y)
    Y = batch[:, -1]  # get batch labels

    cost_ = 0
    batch_size = len(batch)

    # initialize gradients and momentum, RMS params
    # dvs = None
    # for _ in range(3):
    #     weights = [np.zeros(w_b.shape) for w_b in params]  # FULL PYTHON LIST
    #     if dvs is None:
    #         dvs = [weights]
    #     else:
    #         dvs.append(weights)
    dvs = None  # TODO: refactor this - find out what the parameters do - too tired for this now.
    params = model.params()
    for _ in range(3):
        weights = []
        for w_b in params:  # FULL PYTHON LIST
            if w_b is not None:
                weights.append(np.zeros(w_b.shape))
            else:
                weights.append(None)
        if dvs is None:
            dvs = [weights]
        else:
            dvs.append(weights)

    # full forward run
    for i in range(batch_size):
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)  # convert label to one-hot

        # Collect Gradients for training example
        # grads, loss = conv(x, y, params, 1, 2, 2)
        probs, feed_results = model.full_forward(x)
        loss = categoricalCrossEntropy(probs, y)  # categorical cross-entropy loss
        # print(loss)
        grads_w, grads_b = model.full_backprop(probs, y, feed_results)

        # model.add_grads(grads_w, grads_b)
        grads = []
        grads.extend(grads_w)
        grads.extend(grads_b)
        for xx in range(model.no_layers() * 2):
            if grads[xx] is not None:
                dvs[0][xx] += grads[xx]
            else:
                dvs[0][xx] = None

        cost_ += loss

    # backprop
    for my_i in range(8):
        if dvs[0][my_i] is None or dvs[1][my_i] is None or dvs[2][my_i] is None:
            continue
        dvs[1][my_i] = beta1 * dvs[1][my_i] + (1 - beta1) * dvs[0][my_i] / batch_size  # momentum update
        dvs[2][my_i] = beta2 * dvs[2][my_i] + (1 - beta2) * (dvs[0][my_i] / batch_size) ** 2  # RMSProp update
        # combine momentum and RMSProp to perform update with Adam
        params[my_i] -= lr * dvs[1][my_i] / np.sqrt(dvs[2][my_i] + 1e-7)

    cost_ = cost_ / batch_size
    cost.append(cost_)

    model.set_params(params)
    return model.params(), cost


#####################################################
##################### Training ######################
#####################################################

class AdamOptimizer:
    def __init__(self, lr=0.01, beta1=0.95, beta2=0.99, batch_size=32, num_epochs=2):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.loss = None
        self.callbacks = []
        self.frequency = 1

    def set_loss(self, loss_fct):
        self.loss = loss_fct

    def addCallbacks(self, callback_list):
        self.callbacks.append(callback_list)

    def setFrequency(self, freq):
        self.frequency = freq


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
        self.optimizer.train(self, train_data)  # TODO: must see what parameters are needed
        raise NotImplementedError("train")


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


def train(img_dim=28, img_depth=1, f=5, num_filt1=8, num_filt2=8, lr=0.01, beta1=0.95, beta2=0.99, batch_size=32,
          num_epochs=2, num_classes=10, save_path="params.pkl"):
    # training data
    # m = 50000
    global params
    m = 500
    X = extract_data('train-images-idx3-ubyte.gz', m, img_dim)
    y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m, 1)

    # TODO: this is a normalization step (can encapsulate it in a function)
    X -= int(np.mean(X))
    X /= int(np.std(X))

    train_data = np.hstack((X, y_dash))

    np.random.shuffle(train_data)

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

    dense4 = Dense(out_dim=10, in_dim=128)
    dense4.set_activation(softmax)
    model.add(dense4)

    # params = model.params()
    # optimizer = AdamOptimizer(lr=0.01, beta1=0.95, beta2=0.99, batch_size=32, num_epochs=2)
    # optimizer.set_loss(categoricalCrossEntropy)
    # optimizer.addCallbacks([DumpModelCallback(save_path='params.pkl')])  # object-oriented style
    # optimizer.setFrequency(1)  # every x-th epoch

    # model.set_optimizer(adamGD)

    # cost = model.train(train_data)

    cost = []

    print("LR:" + str(lr) + ", Batch Size:" + str(batch_size))

    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm(batches)
        for x, batch in enumerate(t):
            # params, cost = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
            params, cost = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, model, cost)
            # TODO: this should be refactored to be in a better place

            t.set_description("Cost: %.2f" % (cost[-1]))

    to_save = [params, cost]

    # TODO: save callback - we need a dumper strategy
    with open(save_path, 'wb') as file:
        pickle.dump(to_save, file)

    return cost
