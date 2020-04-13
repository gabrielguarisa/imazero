import os
from .dataset import Dataset
from .makers import make_mnist
import wisardpkg as wp


def get_dataset(dataset_name, binarization_name, folder="datasets/data/"):
    if folder[-1] != "/":
        folder = "{}/".format(folder)

    filename = "{}{}/{}_info.ds".format(folder, dataset_name, binarization_name)
    if os.path.isfile(filename):
        print("Loading:", filename)
        return Dataset.load(dataset_name, binarization_name, folder)
    elif dataset_name == "mnist" or dataset_name == "fashion":
        return make_mnist(dataset_name, binarization_name, folder)
    else:
        raise Exception("Dataset not found!")


def get_shape(dataset_name):
    shapes = {
        "mnist": (28, 28),
        "fashion": (28, 28),
    }

    return shapes[dataset_name]
