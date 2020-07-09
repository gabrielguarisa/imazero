import numpy as np
from .dataset import Dataset
from .preprocessing import grayscale, LabelBinarizer
from mnist import get_mnist, get_fashion_mnist
import wisardpkg as wp
import pandas as pd
from sklearn.model_selection import train_test_split
from cifar10_web import cifar10


def make_mnist(dataset_name, binarization_name, folder):
    getter_func = get_fashion_mnist if dataset_name == "fashion" else get_mnist
    train_images, train_labels, test_images, test_labels = getter_func()

    return Dataset.binarize(
        np.copy(train_images),
        np.copy(train_labels),
        np.copy(test_images),
        np.copy(test_labels),
        dataset_name,
        binarization_name,
        num_classes=10,
        entry_size=784,
    ).save(folder)


def make_cifar10(binarization_name, folder):
    train_images, train_labels, test_images, test_labels = cifar10()

    train_images = train_images.reshape([-1, 3, 32, 32])
    test_images = test_images.reshape([-1, 3, 32, 32])

    # move the channel dimension to the last
    train_images = np.rollaxis(train_images, 1, 4)
    test_images = np.rollaxis(test_images, 1, 4)

    train_images_gray = np.array(grayscale(train_images) * 255, dtype="int")
    test_images_gray = np.array(grayscale(test_images) * 255, dtype="int")

    lb = LabelBinarizer()
    lb.fit(list(range(10)))
    new_train_labels = lb.inverse_transform(train_labels)
    new_test_labels = lb.inverse_transform(test_labels)

    return Dataset.binarize(
        train_images_gray,
        new_train_labels,
        test_images_gray,
        new_test_labels,
        "cifar10",
        binarization_name,
        num_classes=10,
        entry_size=1024,
        shape=(32, 32),
    ).save(folder)
