from ctypes import *
import numpy as np
from imazero._lib import flib, WrapperBase
import wisardpkg as wp
from imazero.utils import random_mapping


class WiSARD(WrapperBase):
    def __init__(self, mapping):
        destroy = flib.wisard_destroy
        destroy.argtypes = [c_void_p]
        destroy.restype = None

        super().__init__(destroy)

        self.mapping = self.list_to_ndarray(mapping, dtype=np.uint32)

        self.__len_memories = len(self.mapping)

        if self.mapping.ndim != 2:
            raise Exception("Mapping needs to be a 2D array!")

        self.wisard_train = flib.wisard_train
        self.wisard_train.argtypes = [c_void_p, c_void_p, c_void_p, c_int]
        self.wisard_train.restype = None

        self.wisard_classify = flib.wisard_classify
        self.wisard_classify.argtypes = [c_void_p, c_void_p, c_void_p, c_int]
        self.wisard_classify.restype = None

        self.wisard_azhar_measures = flib.wisard_azhar_measures
        self.wisard_azhar_measures.argtypes = [
            c_void_p,
            c_void_p,
            c_void_p,
            c_void_p,
            c_int,
        ]
        self.wisard_azhar_measures.restype = None

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

    def azhar_measures(self, X, y):
        X = self.list_to_ndarray(X, dtype=np.uint8)
        y = self.list_to_ndarray(y, dtype=np.uint8)

        if X.ndim != 2:
            raise Exception("X needs to be a 2D array!")

        if self.ptr == None:
            raise Exception("Run fit first!")

        output = np.zeros(self.__len_memories * 3, dtype=np.double)
        self.wisard_azhar_measures(
            self.ptr, X.ctypes.data, y.ctypes.data, output.ctypes.data, len(X)
        )
        return np.reshape(output, (self.__len_memories, 3))

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
        mapping = random_mapping(tuple_size, entry_size, complete_addressing)
        for _ in range(address_replication):
            mapping = np.concatenate(
                (mapping, random_mapping(tuple_size, entry_size, complete_addressing))
            )
        super().__init__(mapping)


class PolimappingWiSARD:
    def __init__(self, mapping):
        _mapping = {}

        for i in range(len(mapping)):
            _mapping[str(i)] = mapping[i]

        self.ptr = wp.Wisard(len(mapping[0][0]), mapping=_mapping)

    def fit(self, X, y):
        self.ptr.train(wp.DataSet(X, np.array(y, dtype=str)))
        return self

    def predict(self, X):
        return np.array(self.ptr.classify(wp.DataSet(X)), dtype=int)

    def score(self, X, y):
        y_pred = self.predict(X)
        total = 0
        for i in range(len(y)):
            total += 1 if y[i] == y_pred[i] else 0

        return total / len(y)


class RandomPolimappingWiSARD(PolimappingWiSARD):
    def __init__(
        self,
        tuple_size,
        ndim,
        entry_size,
        address_replication=0,
        complete_addressing=True,
    ):
        all_mappings = {}
        for i in range(ndim):
            mapping = random_mapping(tuple_size, entry_size, complete_addressing)
            for _ in range(address_replication):
                mapping = np.concatenate(
                    (
                        mapping,
                        random_mapping(tuple_size, entry_size, complete_addressing),
                    )
                )
            all_mappings[i] = mapping

        super().__init__(all_mappings)
