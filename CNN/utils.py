import numpy as np
import gzip


def extract_data(filename, num_images, IMAGE_HEIGHT, IMAGE_WIDTH):
    """
    Extract images by reading the file bytestream. Reshape the read values into a 3D matrix of dimensions [m, h, w],
    where `m` is the number of training examples.
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_HEIGHT * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_HEIGHT * IMAGE_WIDTH)
        return data


def extract_labels(filename, num_images):
    """
    Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def initializeFilter(size, scale=1.0):
    stddev = scale / np.sqrt(np.prod(size))
    return np.random.normal(loc=0, scale=stddev, size=size)


def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01


def safe_division(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return -1
