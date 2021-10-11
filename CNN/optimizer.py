from tqdm import tqdm

import numpy as np


class AdamOptimizer:
    def __init__(self, num_classes=10, img_depth=1, img_dim_x=28, img_dim_y=28, lr=0.01, beta1=0.95, beta2=0.99,
                 batch_size=32, num_epochs=5):
        self.num_classes = num_classes
        self.img_dim_x = img_dim_x
        self.img_dim_y = img_dim_y
        self.img_depth = img_depth
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.loss = None
        self.callbacks = {}
        self.frequency = 1

    def set_loss(self, loss_fct):
        self.loss = loss_fct

    def addCallbacks(self, callback_dict):
        self.callbacks.update(callback_dict)

    def setFrequency(self, freq):
        self.frequency = freq

    def train(self, model, dataloader):
        # train_data = dataloader.load_data(m, train_sample_path, train_label_path)
        # np.random.shuffle(train_data)  # TODO: shuffle meta for loading data instead of loading all data to shuffle

        cost = []

        print("LR:" + str(self.lr) + ", Batch Size:" + str(self.batch_size))

        for indx, epoch in enumerate(range(self.num_epochs)):
            # np.random.shuffle(train_data)  # TODO: need data shuffle every epoch ???

            batches = range(dataloader.no_batches)  # TODO: create getter
            t = tqdm(batches)

            for _ in t:
                batch = dataloader.load_batch()
                params, cost = self.adamGD(batch, self.num_classes, self.img_dim_x, self.img_dim_y, self.img_depth, model, cost)

                t.set_description("Cost: %.2f" % (cost[-1]))

            dataloader.reset_iterator()
            if indx % self.frequency == 0:  # TODO: should include more callbacks (evaluation metrics)
                print("Saving model...")
                to_save = [model.params(), cost]
                self.callbacks["SaveModel"](to_save)

        return cost

    def adamGD(self, batch, num_classes, dim_x, dim_y, n_c, model, cost):
        lr = self.lr
        beta1 = self.beta1
        beta2 = self.beta2
        """
        update the parameters through Adam gradient descnet.
        """
        global grads
        X = batch[:, 0:-1]  # get batch inputs
        X = X.reshape(len(batch), n_c, dim_x, dim_y)
        Y = batch[:, -1]  # get batch labels

        cost_ = 0
        batch_size = len(batch)

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
            feed_results = model.full_forward(x)
            probs = feed_results[-1]
            loss_value = self.loss(probs, y)  # categorical cross-entropy loss value
            grads_w, grads_b = model.full_backprop(probs, y, feed_results)

            grads = []
            grads.extend(grads_w)
            grads.extend(grads_b)
            for xx in range(model.no_layers() * 2):
                if grads[xx] is not None:
                    dvs[0][xx] += grads[xx]
                else:
                    dvs[0][xx] = None

            cost_ += loss_value

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
