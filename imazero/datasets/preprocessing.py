import numpy as np
import math
from skimage.filters import threshold_sauvola, threshold_yen, threshold_otsu, threshold_local
from scipy.ndimage import interpolation
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import wisardpkg as wp


def grayscale(data, dtype="float32"):
    r, g, b = (
        np.asarray(0.3, dtype=dtype),
        np.asarray(0.59, dtype=dtype),
        np.asarray(0.11, dtype=dtype),
    )
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    return rst


def distance(x_1, x_2, y_1, y_2):
    return math.sqrt(pow((x_2 - x_1), 2) + pow((y_2 - y_1), 2))


def border_distance(x, y, mult=1.0):
    array = np.zeros((x, y))
    x_center = (x - 1) / 2
    y_center = (y - 1) / 2

    max_distance = distance(x_center, 0, y_center, 0)

    for i in range(x):
        for j in range(y):
            array[i][j] = (max_distance - distance(x_center, i, y_center, j)) * mult

    return array.astype(int)


def moments(image):
    # A trick in numPy to create a mesh grid
    c0, c1 = np.mgrid[: image.shape[0], : image.shape[1]]
    totalImage = np.sum(image)  # sum of pixels
    m0 = np.sum(c0 * image) / totalImage  # mu_x
    m1 = np.sum(c1 * image) / totalImage  # mu_y
    m00 = np.sum((c0 - m0) ** 2 * image) / totalImage  # var(x)
    m11 = np.sum((c1 - m1) ** 2 * image) / totalImage  # var(y)
    m01 = np.sum((c0 - m0) * (c1 - m1) * image) / totalImage  # covariance(x,y)
    # Notice that these are \mu_x, \mu_y respectively
    mu_vector = np.array([m0, m1])
    # Do you see a similarity between the covariance matrix
    covariance_matrix = np.array([[m00, m01], [m01, m11]])
    return mu_vector, covariance_matrix


def deskew(image):
    c, v = moments(image)
    alpha = v[0, 1] / v[0, 0]
    alpha = v[0, 1] / v[0, 0]
    affine = np.array([[1, 0], [alpha, 1]])
    ocenter = np.array(image.shape) / 2.0
    offset = c - np.dot(affine, ocenter)
    return interpolation.affine_transform(image, affine, offset=offset)


class MeanThresholding(object):
    def __init__(self, shape=None):
        self.shape = shape

    def transform(self, images):
        binary_images = []
        for i in range(len(images)):
            binary_images.append(np.where(images[i] > np.mean(images[i]), 1, 0).ravel())

        return binary_images


class Sauvola(object):
    def __init__(self, window_size=5, shape=None):
        self.window_size = window_size
        self.shape = shape

    def transform(self, images):
        binary_images = []

        for i in range(len(images)):
            images[i] = np.array(images[i]).reshape(self.shape)
            thresh_sauvola = threshold_sauvola(images[i], window_size=self.window_size)
            binary_images.append((images[i] > thresh_sauvola).ravel())

        return binary_images


class LocalThresholding(object):
    def __init__(self, block_size=23, shape=None):
        self.block_size = block_size
        self.shape = shape

    def transform(self, images):
        binary_images = []

        for i in range(len(images)):
            image = np.array(images[i]).reshape(self.shape)
            thresh_sauvola = threshold_local(image, block_size=self.block_size)
            binary_img = np.where(image > thresh_sauvola, 1, 0)
            binary_images.append(binary_img.ravel())

        return binary_images


class Yen(object):
    def __init__(self, shape=None):
        self.shape = shape

    def transform(self, images):
        binary_images = []

        for i in range(len(images)):
            images[i] = np.array(images[i]).reshape(self.shape)
            thresh_yen = threshold_yen(images[i])
            binary_images.append((images[i] > thresh_yen).ravel())

        return binary_images


class Otsu(object):
    def __init__(self, shape=None):
        self.shape = shape

    def transform(self, images):
        binary_images = []

        for i in range(len(images)):
            images[i] = np.array(images[i]).reshape(self.shape)
            thresh_otsu = threshold_otsu(images[i])
            binary_images.append((images[i] > thresh_otsu).ravel())

        return binary_images


class MeanThresholdingWithDeskew(object):
    def transform(self, images):
        binary_images = []

        for i in range(len(images)):
            new_image = deskew(np.array(images[i])).ravel()
            binary_images.append(np.where(new_image > np.mean(new_image), 1, 0))

        return binary_images


class Thermometer(object):
    def __init__(self, size=32, shape=None):
        self.shape = shape
        self.size = size

    def transform(self, images):
        therm = wp.DynamicThermometer(
            [self.size] * len(images[0]),
            minimum=np.min(images, axis=0),
            maximum=np.max(images, axis=0),
        )
        binary_images = []

        for i in range(len(images)):
            images[i] = np.array(images[i]).reshape(self.shape)
            binary_images.append(therm.transform(images[i]))

        return binary_images


def get_preprocessing(binarization):
    if binarization == "mt":
        return MeanThresholding()
    elif binarization == "sv":
        return Sauvola()
    elif binarization == "yn":
        return Yen()
    elif binarization == "ot":
        return Otsu()
    elif binarization == "mtwd":
        return MeanThresholdingWithDeskew()
    elif binarization == "lt":
        return LocalThresholding()
    elif binarization == "th":
        return Thermometer()
    else:
        raise Exception("Binarization function not found!")
