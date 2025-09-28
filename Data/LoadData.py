import numpy as np
import os

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic = np.frombuffer(f.read(4), dtype='>i4')[0]  # Diese Zeile einschalten
        num_images = np.frombuffer(f.read(4), dtype='>i4')[0]
        rows = np.frombuffer(f.read(4), dtype='>i4')[0]
        cols = np.frombuffer(f.read(4), dtype='>i4')[0]
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images


def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def getData():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    train_images_path = os.path.join(current_dir, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(current_dir, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(current_dir, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(current_dir, 't10k-labels.idx1-ubyte')

    X_train = load_mnist_images(train_images_path)
    y_train = load_mnist_labels(train_labels_path)
    X_test = load_mnist_images(test_images_path)
    y_test = load_mnist_labels(test_labels_path)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    return X_train_flat, y_train, X_test_flat, y_test