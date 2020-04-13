from abc import ABC, abstractmethod
import numpy as np


class TemplateFinder(ABC):
    def __init__(self, name):
        self.name = name
        super().__init__()

    def __str__(self):
        return self.name

    def get_name(self):
        return self.name

    def set_shape(self, shape):
        pass

    @abstractmethod
    def search(self, arr):
        pass


class PatternFinder(TemplateFinder):
    def __init__(self, shape, patterns, name="PatternFinder"):
        self._shape = shape
        self._patterns = patterns
        self._window_shape = self._patterns[0].shape
        super().__init__(name)

    def set_shape(self, shape):
        self._shape = shape

    def search(self, arr):
        addresses = []
        image = np.array(arr).reshape(self._shape)

        for r in range(self._shape[0] - self._window_shape[0]):
            for c in range(self._shape[1] - self._window_shape[1]):
                window = image[
                    r : r + self._window_shape[0], c : c + self._window_shape[1]
                ]
                for pattern in self._patterns:
                    if np.array_equal(window, pattern):
                        temp_addr = []
                        for i in range(r, r + self._window_shape[0]):
                            for j in range(c, c + self._window_shape[1]):
                                temp_addr.append(i * self._shape[0] + j)
                        addresses.append(temp_addr)

        return addresses


class OnesFinder(TemplateFinder):
    def __init__(self, name="OnesFinder"):
        super().__init__(name)

    def search(self, arr):
        addresses = [[]]

        for i in range(len(arr)):
            if arr[i] == 1:
                addresses[0].append(i)

        return addresses


def get_line_finder(shape=None):
    return PatternFinder(
        shape,
        [
            np.array([[0, 1], [0, 1]]),
            np.array([[1, 0], [1, 0]]),
            np.array([[1, 1], [0, 0]]),
            np.array([[0, 0], [1, 1]]),
        ],
        name="LineFinder",
    )


def get_diagonal_finder(shape=None):
    return PatternFinder(
        shape,
        [
            np.array([[0, 1], [0, 1]]),
            np.array([[1, 0], [1, 0]]),
            np.array([[1, 1], [0, 0]]),
            np.array([[0, 0], [1, 1]]),
            np.array([[1, 0], [0, 1]]),
            np.array([[0, 1], [1, 0]]),
        ],
        name="DiagonalFinder",
    )
