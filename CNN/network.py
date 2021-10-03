"""
Description: methods to set up and train the network's parameters.

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
"""
from CNN.forward import *
from CNN.backward import *
from CNN.utils import *

import numpy as np
import pickle
from tqdm import tqdm


#####################################################
############### Building The Network ################
#####################################################

def conv(image, label, params, conv_s, pool_f, pool_s):
    # TODO: extract in model.forward(): !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # layer.activation = relu
    # layer.activation = softmax
    ################################################
    ############## Forward Operation ###############
    ################################################
    # TODO: params[i] and params[i + no_layers]  (depends on weight and bias index)
    conv1 = convolution(image, params[0], params[4], conv_s)  # convolution operation
    # TODO: create ReLU separated function
    conv1[conv1 <= 0] = 0  # pass through ReLU non-linearity

    conv2 = convolution(conv1, params[1], params[5], conv_s)  # second convolution operation
    # TODO: create ReLU separated function
    conv2[conv2 <= 0] = 0  # pass through ReLU non-linearity

    pooled = maxpool(conv2, pool_f, pool_s)  # maxpooling operation

    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1))  # flatten pooled layer

    z = params[2].dot(fc) + params[6]  # first dense layer
    # TODO: create ReLU separated function
    z[z <= 0] = 0  # pass through ReLU non-linearity

    out = params[3].dot(z) + params[7]  # second dense layer

    # softmax is an activation function...
    probs = softmax(out)  # predict class probabilities with the softmax activation function

    ################################################
    #################### Loss ######################
    ################################################

    # TODO: loss as callback !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    loss = categoricalCrossEntropy(probs, label)  # categorical cross-entropy loss

    ################################################
    ############# Backward Operation ###############
    ################################################
    dout = probs - label  # derivative of loss w.r.t. final dense layer output
    dw4 = dout.dot(z.T)  # loss gradient of final dense layer weights
    db4 = np.sum(dout, axis=1).reshape(params[7].shape)  # loss gradient of final dense layer biases

    dz = params[3].T.dot(dout)  # loss gradient of first dense layer outputs
    dz[z <= 0] = 0  # backpropagate through ReLU
    dw3 = dz.dot(fc.T)
    db3 = np.sum(dz, axis=1).reshape(params[6].shape)

    dfc = params[2].T.dot(dz)  # loss gradients of fully-connected layer (pooling layer)
    dpool = dfc.reshape(pooled.shape)  # reshape fully connected into dimensions of pooling layer

    dconv2 = maxpoolBackward(dpool, conv2, pool_f,
                             pool_s)  # backprop through the max-pooling layer(only neurons with highest activation in window get updated)
    dconv2[conv2 <= 0] = 0  # backpropagate through ReLU

    dconv1, df2, db2 = convolutionBackward(dconv2, conv1, params[1],
                                           conv_s)  # backpropagate previous gradient through second convolutional layer.
    dconv1[conv1 <= 0] = 0  # backpropagate through ReLU

    dimage, df1, db1 = convolutionBackward(dconv1, image, params[0],
                                           conv_s)  # backpropagate previous gradient through first convolutional layer.

    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]

    return grads, loss


#####################################################
################### Optimization ####################
#####################################################

def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
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
    dvs = None
    for _ in range(3):
        weights = [np.zeros(w_b.shape) for w_b in params]  # FULL PYTHON LIST
        if dvs is None:
            dvs = [weights]
        else:
            dvs.append(weights)

    # full forward run
    for i in range(batch_size):
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)  # convert label to one-hot

        # Collect Gradients for training example
        grads, loss = conv(x, y, params, 1, 2, 2)
        # [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads
        # TODO: [df1,df2,dw3,cw4] are weights; [db1,db2,db3,db4] are biases

        for xx in range(8):  # [0, 4, 1, 5, 2, 6, 3, 7]:
            dvs[0][xx] += grads[xx]

        cost_ += loss

    # backprop
    for my_i in range(8):
        dvs[1][my_i] = beta1 * dvs[1][my_i] + (1 - beta1) * dvs[0][my_i] / batch_size  # momentum update
        dvs[2][my_i] = beta2 * dvs[2][my_i] + (1 - beta2) * (dvs[0][my_i] / batch_size) ** 2  # RMSProp update
        # combine momentum and RMSProp to perform update with Adam
        params[my_i] -= lr * dvs[1][my_i] / np.sqrt(dvs[2][my_i] + 1e-7)

    cost_ = cost_ / batch_size
    cost.append(cost_)

    return params, cost


#####################################################
##################### Training ######################
#####################################################

def train(num_classes=10, lr=0.01, beta1=0.95, beta2=0.99, img_dim=28, img_depth=1, f=5, num_filt1=8, num_filt2=8,
          batch_size=32, num_epochs=2, save_path='params.pkl'):  # TODO: batch size is a hyperparam for model
    # training data
    # m = 50000
    m = 500
    X = extract_data('train-images-idx3-ubyte.gz', m, img_dim)
    y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m, 1)

    # TODO: this is a normalization step (can encapsulate it in a function)
    X -= int(np.mean(X))
    X /= int(np.std(X))

    train_data = np.hstack((X, y_dash))

    np.random.shuffle(train_data)

    params = [0., 0., 0., 0., 0., 0., 0., 0.]
    # TODO: create model
    # model = SequentialModel()
    # Initializing all the parameters

    # INITIALIZE CONV & FC LAYERS WEIGHTS (co, ci, kh, kw) & BIASES

    # TODO: 1 conv2d
    # conv_1 = Conv2D(in_dim=img_depth, out_dim=num_filt1, kernel=(f, f))
    # model.add(conv_1)
    params[0] = (num_filt1, img_depth, f, f)  # link input_image_shape to conv2d_layer1_shape
    params[0] = initializeFilter(params[0])  # TODO: isinstance checks => choose initialization politics
    params[4] = np.zeros((params[0].shape[0], 1))

    # TODO: 2 conv2d
    # conv_2 = Conv2D(in_dim=num_filt1, out_dim=num_filt2, kernel=(f, f))
    # model.add(conv_2)
    params[1] = (num_filt2, num_filt1, f, f)  # link conv2d_layer1_shape to conv2d_layer2_shape
    params[1] = initializeFilter(params[1])
    params[5] = np.zeros((params[1].shape[0], 1))

    # TODO: 3 dense
    # dense3 = Dense(in_dim=800, out_dim=128)
    # model.add(dense3)
    params[2] = (128, 800)
    params[2] = initializeWeight(params[2])
    params[6] = np.zeros((params[2].shape[0], 1))

    # TODO: 4 dense
    # dense4 = Dense(in_dim=128, out_dim=10)
    # model.add(dense4)
    params[3] = (10, 128)
    params[3] = initializeWeight(params[3])
    params[7] = np.zeros((params[3].shape[0], 1))

    # params = model.params()

    cost = []

    print("LR:" + str(lr) + ", Batch Size:" + str(batch_size))

    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm(batches)
        for x, batch in enumerate(t):
            params, cost = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
            t.set_description("Cost: %.2f" % (cost[-1]))

    to_save = [params, cost]

    # TODO: save callback - we need a dumper strategy
    with open(save_path, 'wb') as file:
        pickle.dump(to_save, file)

    return cost
