import numpy as np


def mean_threshold(image):
    return np.where(image < np.mean(image), 0, 1).ravel()
