from ctypes import *
import numpy as np
from imazero._lib import flib, WrapperBase


class WiSARD(WrapperBase):
    def __init__(self, mapping):
        destroy = flib.wisard_destroy
        destroy.argtypes = [c_void_p]
        destroy.restype = None

        super().__init__(destroy)

        self.mapping = self.list_to_ndarray(mapping, dtype=np.uint32)

        if self.mapping.ndim != 2:
            raise Exception("Mapping needs to be a 2D array!")

        self.wisard_train = flib.wisard_train
        self.wisard_train.argtypes = [c_void_p, c_void_p, c_void_p, c_int]
        self.wisard_train.restype = None

        self.wisard_classify = flib.wisard_classify
        self.wisard_classify.argtypes = [c_void_p, c_void_p, c_void_p, c_int]
        self.wisard_classify.restype = None

    def fit(self, X, y):
        X = self.list_to_ndarray(X, dtype=np.uint8)
        y = self.list_to_ndarray(y, dtype=np.uint8)

        if X.ndim != 2:
            raise Exception("X needs to be a 2D array!")

        if self.ptr == None:
            wisard_create = flib.wisard_create
            wisard_create.argtypes = [c_void_p, c_int, c_uint, c_uint, c_uint]
            wisard_create.restype = c_void_p
            num_rows, num_cols = self.mapping.shape
            self.ptr = wisard_create(
                self.mapping.ctypes.data, num_rows, num_cols, len(set(y)), len(X[0])
            )
            self.mapping = None

        self.wisard_train(self.ptr, X.ctypes.data, y.ctypes.data, y.size)
        return self

    def predict(self, X):
        X = self.list_to_ndarray(X, dtype=np.uint8)
        if X.ndim != 2:
            raise Exception("X needs to be a 2D array!")

        if self.ptr == None:
            raise Exception("Run fit first!")

        output = np.zeros(len(X), dtype=np.uint32)
        self.wisard_classify(self.ptr, X.ctypes.data, output.ctypes.data, len(X))
        return output

    def score(self, X, y):
        y_pred = self.predict(X)
        total = 0
        for i in range(len(y)):
            total += 1 if y[i] == y_pred[i] else 0

        return total / len(y)


class RandomWiSARD(WiSARD):
    def __init__(
        self, tuple_size, entry_size, address_replication=0, complete_addressing=True
    ):
        mapping = self._random_mapping(tuple_size, entry_size, complete_addressing)
        for _ in range(address_replication):
            mapping = np.concatenate(
                (
                    mapping,
                    self._random_mapping(tuple_size, entry_size, complete_addressing),
                )
            )
        super().__init__(mapping)

    def _random_mapping(self, tuple_size, entry_size, complete_addressing=True):
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
