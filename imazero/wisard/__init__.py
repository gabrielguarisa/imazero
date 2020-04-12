from ctypes import *
import numpy as np
from imazero._lib import flib

class WrapperBase:
    INVALID_ARGUMENTS = "Arguments types invalids!"

    def __del__(self):
        if self.ptr:
            self.destroy(self.ptr)
            self.ptr = None

    def validate(self):
        if getattr(self, "ptr", None) is None:
            raise RuntimeError("class pointer is null!")
        if getattr(self, "destroy", None) is None:
            raise RuntimeError("destructor not found!")


class Memory(WrapperBase):
    def __init__(self, mapping):
        if type(mapping) == list or mapping.dtype.name != "uint32":
            mapping = np.array(mapping, dtype=np.uint32)

        native_memory_create = flib.memory_create
        native_memory_create.argtypes = [c_void_p, c_int]
        native_memory_create.restype = c_void_p
        self.ptr = native_memory_create(mapping.ctypes.data, mapping.size)

        self.destroy = flib.memory_destroy
        self.destroy.argtypes = [c_void_p]
        self.destroy.restype = None
