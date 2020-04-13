import numpy as np
from .dataset import Dataset
from .preprocessing import grayscale, LabelBinarizer
from mnist import get_mnist, get_fashion_mnist
import wisardpkg as wp
import pandas as pd
from sklearn.model_selection import train_test_split

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
