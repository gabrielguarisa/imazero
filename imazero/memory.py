from ctypes import *
import numpy as np
from imazero._lib import flib, WrapperBase


class Memory(WrapperBase):
    def __init__(self, mapping, ndim):
        self.__ndim = ndim
        if type(mapping) == list or mapping.dtype.name != "uint32":
            mapping = np.array(mapping, dtype=np.uint32)

        memory_create = flib.memory_create
        memory_create.argtypes = [c_void_p, c_int, c_uint]
        memory_create.restype = c_void_p
        self.ptr = memory_create(mapping.ctypes.data, mapping.size, ndim)

        self.destroy = flib.memory_destroy
        self.destroy.argtypes = [c_void_p]
        self.destroy.restype = None

        self.memory_write = flib.memory_write
        self.memory_write.argtypes = [c_void_p, c_void_p, c_int]
        self.memory_write.restype = None

        self.memory_read = flib.memory_read
        self.memory_read.argtypes = [c_void_p, c_void_p, c_void_p]
        self.memory_read.restype = None

    def write(self, image, dim):
        if type(image) == list or image.dtype.name != "uint8":
            image = np.array(image, dtype=np.uint8)

        self.memory_write(self.ptr, image.ctypes.data, dim)

    def read(self, image):
        if type(image) == list or image.dtype.name != "uint8":
            image = np.array(image, dtype=np.uint8)

        output = np.zeros(self.__ndim, dtype=np.uint32)
        self.memory_read(self.ptr, image.ctypes.data, output.ctypes.data)
        return output