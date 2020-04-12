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

    def fit(self, X, y):
        X = self.list_to_ndarray(X, dtype=np.uint8)
        y = self.list_to_ndarray(y, dtype=np.uint8)

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
