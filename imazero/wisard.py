from ctypes import *
import numpy as np
from imazero._lib import flib, WrapperBase


class WiSARD(WrapperBase):
    def __init__(self, mapping):
        if type(mapping) == list or mapping.dtype.name != "uint32":
            mapping = np.array(mapping, dtype=np.uint32)

        if mapping.ndim != 2:
            raise Exception("Mapping needs to be a 2D array!")

        num_rows, num_cols = mapping.shape

        wisard_create = flib.wisard_create
        wisard_create.argtypes = [c_void_p, c_int, c_uint]
        wisard_create.restype = c_void_p
        self.ptr = wisard_create(mapping.ctypes.data, num_rows, num_cols)

        self.destroy = flib.wisard_destroy
        self.destroy.argtypes = [c_void_p]
        self.destroy.restype = None
