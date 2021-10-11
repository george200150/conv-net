import json
import math
import os.path as osp
import random
from itertools import chain

import numpy as np
from PIL import Image

from CNN.utils import extract_data, extract_labels


class DataLoader:
    def __init__(self, img_dim_x, img_dim_y, batch_size):
        self.img_dim_x = img_dim_x
        self.img_dim_y = img_dim_y
        self.batch_size = batch_size
        self.batch_index = 0

    def reset_iterator(self):
        self.batch_index = 0

    def load_batch(self):
        raise NotImplementedError("Get batch abstract.")

    def load_data(self):
        raise NotImplementedError("No data load politics.")


class MNISTDataLoader(DataLoader):
    def __init__(self, img_dim_x, img_dim_y, sample_path, label_path, batch_size, num_images):
        super(MNISTDataLoader, self).__init__(img_dim_x, img_dim_y, batch_size)
        self.sample_path = sample_path
        self.label_path = label_path

        # load whole data then select batches from the lists
        self.imgs = extract_data(self.sample_path, num_images, self.img_dim_x, self.img_dim_y)
        self.y_dash = extract_labels(self.label_path, num_images).reshape(num_images, 1)
        self.no_batches = int(math.ceil(len(self.imgs) / batch_size))

    def load_batch(self):
        return_values = []
        for indx in range(self.batch_size * self.batch_index, self.batch_size * (self.batch_index + 1)):
            if indx >= len(self.imgs):
                break
            return_values.append([self.imgs[indx], self.y_dash[indx]])

        X = []
        y_dash = []
        for x, y in return_values:
            X.append(x)
            y_dash.append(y)  # label must be incapsulated in list (it's a list in here)

        X = np.asarray(X)
        y_dash = np.asarray(y_dash)

        self.imgs -= int(np.mean(self.imgs))
        self.imgs /= int(np.std(self.imgs))
        data = np.hstack((X, y_dash))

        self.batch_index += 1
        return data

    def load_data(self):
        self.imgs -= int(np.mean(self.imgs))
        self.imgs /= int(np.std(self.imgs))
        data = np.hstack((self.imgs, self.y_dash))
        return data


_VALID_SPLITS = ('train', 'val', 'test')
_VALID_SCENE_TYPES = ('indoors', 'outdoor')


def check_and_tuplize_tokens(tokens, valid_tokens):
    if not isinstance(tokens, (tuple, list)):
        tokens = (tokens,)
    for split in tokens:
        assert split in valid_tokens
    return tokens


def enumerate_paths(src):
    """flatten out a nested dictionary into an iterable
    DIODE metadata is a nested dictionary;
    One could easily query a particular scene and scan, but sequentially
    enumerating files in a nested dictionary is troublesome. This function
    recursively traces out and aggregates the leaves of a tree.
    """
    if isinstance(src, list):
        return src
    elif isinstance(src, dict):
        acc = []
        for k, v in src.items():
            _sub_paths = enumerate_paths(v)
            _sub_paths = list(map(lambda x: osp.join(k, x), _sub_paths))
            acc.append(_sub_paths)
        return list(chain.from_iterable(acc))
    else:
        raise ValueError('do not accept data type {}'.format(type(src)))


diode_cls_mapping = {"indoors": 0, "outdoor": 1}
diode_cls_unmapping = {0: "indoors", 1: "outdoor"}


class DIODEDataLoader(DataLoader):
    def __init__(self, img_dim_x, img_dim_y, meta_fname, data_root, splits, scene_types, num_images, batch_size):
        super(DIODEDataLoader, self).__init__(img_dim_x, img_dim_y, batch_size)
        self.data_root = data_root
        self.splits = check_and_tuplize_tokens(
            splits, _VALID_SPLITS
        )
        self.scene_types = check_and_tuplize_tokens(
            scene_types, _VALID_SCENE_TYPES
        )
        with open(meta_fname, 'r') as f:
            self.meta = json.load(f)

        imgs = []
        for split in self.splits:
            for scene_type in self.scene_types:
                _curr = enumerate_paths(self.meta[split][scene_type])
                _curr = map(lambda x: osp.join(split, scene_type, x), _curr)
                imgs.extend(list(_curr))
        self.imgs = imgs

        # -----------------------------------------------------------------------------
        num_images = min(num_images, len(self.imgs))
        self.imgs = random.sample(self.imgs, num_images)  # only use num_images images

        self.classes = list(set([x.split("\\")[3] for x in self.imgs]))
        # -----------------------------------------------------------------------------
        self.no_batches = int(math.ceil(len(self.imgs) / batch_size))

    def load_batch(self):
        return_values = []
        for indx in range(self.batch_size * self.batch_index, self.batch_size * (self.batch_index + 1)):
            if indx > len(self.imgs):
                break
            im_fname = osp.join(self.data_root, '{}.png'.format(self.imgs[indx]))
            image_path = osp.join(self.data_root, im_fname)
            im = Image.open(image_path)
            cls = image_path.split("\\")[1]  # get indoors / outdoors

            # RGB images with a resolution of 1024 × 768.
            newsize = (self.img_dim_x, self.img_dim_y)
            # newsize = (28, 28)
            im = im.resize(newsize)

            im = np.array(im).astype(np.float32)
            im = np.moveaxis(im, -1, 0)  # channels first
            im = im.reshape(3 * self.img_dim_x * self.img_dim_y)
            return_values.append([im, cls])

        X = []
        y_dash = []
        for x, y in return_values:
            X.append(x)
            y_dash.append([diode_cls_mapping[y]])  # label must be incapsulated in list

        X = np.asarray(X)
        y_dash = np.asarray(y_dash)

        mean_R = 105.47346543030518
        mean_G = 94.42803436155408
        mean_B = 86.98747079847048

        mean = (mean_R + mean_G + mean_B) / 3

        stdev_R = 57.78172984460825
        stdev_G = 58.87019203306219
        stdev_B = 65.32571556352487

        stdev = stdev_R + stdev_G + stdev_B

        X -= mean  # normalize data (on every channel? merge mean and std together?)
        X /= stdev
        data = np.hstack((X, y_dash))

        self.batch_index += 1
        return data

    def load_data(self):
        return_values = []
        for im in self.imgs:
            # if len(return_values) == m:
            #     break
            im_fname = osp.join(self.data_root, '{}.png'.format(im))
            image_path = osp.join(self.data_root, im_fname)
            im = Image.open(image_path)
            cls = image_path.split("\\")[1]  # get indoors / outdoors

            # RGB images with a resolution of 1024 × 768.
            newsize = (self.img_dim_x, self.img_dim_y)
            # newsize = (28, 28)
            im = im.resize(newsize)

            im = np.array(im).astype(np.float32)
            im = np.moveaxis(im, -1, 0)  # channels first
            im = im.reshape(3 * self.img_dim_x * self.img_dim_y)
            return_values.append([im, cls])

        X = []
        y_dash = []
        for x, y in return_values:
            X.append(x)
            y_dash.append([diode_cls_mapping[y]])  # label must be incapsulated in list

        X = np.asarray(X)
        y_dash = np.asarray(y_dash)

        X -= int(np.mean(X))
        X /= int(np.std(X))
        data = np.hstack((X, y_dash))
        return data

