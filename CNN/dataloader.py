import numpy as np

from CNN.utils import extract_data, extract_labels


class DataLoader:  # TODO: refactor this to look more appropiate
    def __init__(self, img_dim):
        self.img_dim = img_dim

    def load_data(self):
        # m = 50000
        m = 500
        X = extract_data('train-images-idx3-ubyte.gz', m, self.img_dim)
        y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m, 1)

        X -= int(np.mean(X))
        X /= int(np.std(X))

        return X, y_dash
