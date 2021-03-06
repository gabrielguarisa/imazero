import pickle
from os import makedirs, path
import random
import numpy as np
from math import log2


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
    remainder = entry_size % tuple_size

    if remainder > 0:
        num_rams += 1
        if complete_addressing:
            indexes = np.concatenate(
                (indexes, np.random.randint(entry_size, size=tuple_size - remainder))
            )
        else:
            indexes = np.concatenate((indexes, np.full(tuple_size - remainder, -1)))

    np.random.shuffle(indexes)
    return np.reshape(indexes, (num_rams, tuple_size))


def get_bin(x, n=0):
    """
  Get the binary representation of x.

  Parameters
  ----------
  x : int
  n : int
      Minimum number of digits. If x needs less digits in binary, the rest
      is filled with zeros.

  Returns
  -------
  str
  """
    return format(x, "b").zfill(n)


def get_int(x):
    """
  Get the int value of x.

  Parameters
  ----------
  x : str
  Returns
  -------
  int
  """
    return int(x, 2)


def change_char(str, i, char):
    return str[:i] + char + str[i + 1 :]


def binary_entropy_func(p):
    if p == 0 or p == 1:
        return 0.0
    return -p * log2(p) - (1 - p) * log2(1 - p)


def get_mental_images(X, y, normalized=True):
    images = get_images_by_label(X, y)
    mental_images = {}
    for key, value in images.items():
        mental_images[key] = np.sum(value, axis=0)

    if normalized:
        for key, _ in images.items():
            mental_images[key] = mental_images[key] / np.max(mental_images[key])

    return mental_images


def get_entropy_images(X, y):
    mi = get_mental_images(X, y)
    entropy_images = {}

    for key, value in mi.items():
        entropy_images[key] = np.array(
            [binary_entropy_func(value[i]) for i in range(len(value))]
        )

    return entropy_images
