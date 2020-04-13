from abc import ABC, abstractmethod
import numpy as np


class TemplateFinder(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def search(self, arr):
        pass


class PatternFinder(TemplateFinder):
    def __init__(self, shape, patterns):
        self._shape = shape
        self._patterns = patterns
        self._window_shape = self._patterns[0].shape
        super().__init__()

    def __str__(self):
        return "PatternFinder"

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
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "OnesFinder"

    def search(self, arr):
        addresses = [[]]

        for i in range(len(arr)):
            if arr[i] == 1:
                addresses[0].append(i)

        return addresses


def get_line_finder(shape):
    return PatternFinder(
        shape,
        [
            np.array([[0, 1], [0, 1]]),
            np.array([[1, 0], [1, 0]]),
            np.array([[1, 1], [0, 0]]),
            np.array([[0, 0], [1, 1]]),
        ],
    )


def get_diagonal_finder(shape):
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
    )
