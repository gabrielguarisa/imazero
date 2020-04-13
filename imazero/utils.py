import pickle
from os import makedirs, path
import random
import numpy as np


def mkdir(folder):
    if folder[-1] != "/":
        folder = "{}/".format(folder)
    makedirs(folder, exist_ok=True)
    return folder


def valid_tuple_sizes(entry_size, max_value=64):
    valid_values = []
    for tuple_size in range(3, max_value):
        if entry_size % tuple_size == 0:
            valid_values.append(tuple_size)
    return valid_values


def get_images_by_label(X, y):
    images = {}
    for i in range(len(y)):
        if y[i] not in images:
            images[y[i]] = []
        images[y[i]].append(X[i])
    return images


def file_exists(filename):
    return path.exists(filename)


def save_data(data, filename):
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(filename):
    with open(filename, "rb") as handle:
        return pickle.load(handle)


def argsort(arr, reverse=True, shuffled=True):
    value_addr_map = {}
    output = []
    for i in range(len(arr)):
        if arr[i] not in value_addr_map:
            value_addr_map[arr[i]] = []

        value_addr_map[arr[i]].append(i)

    keys = list(value_addr_map.keys())
    keys.sort()

    for key in keys:
        values = value_addr_map[key]
        if shuffled:
            random.shuffle(values)

        output = [*output, *values]

    if reverse:
        return output[::-1]

    return output


def random_mapping(tuple_size, entry_size, complete_addressing=True):
    indexes = np.arange(entry_size)
    num_rams = entry_size // tuple_size
    remainder = tuple_size - (entry_size % tuple_size)

    if remainder > 0:
        num_rams += 1
        if complete_addressing:
            indexes = np.concatenate(
                (indexes, np.random.randint(entry_size, size=remainder))
            )
        else:
            indexes = np.concatenate((indexes, np.full(remainder, -1)))

    np.random.shuffle(indexes)
    return np.reshape(indexes, (num_rams, tuple_size))
