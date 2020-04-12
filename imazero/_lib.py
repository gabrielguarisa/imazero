from ctypes import CDLL
import numpy as np


def get_dll():
    import os, fnmatch

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    file_dll = os.path.join(__location__, "imazero.so")
    if not os._exists(file_dll):
        for root, dirs, files in os.walk(__location__):
            for name in files:
                if fnmatch.fnmatch(name, "imazero*.so"):
                    file_dll = os.path.join(root, name)
                    return file_dll
    return file_dll


flib = CDLL(get_dll())


class WrapperBase:
    def __init__(self, destroy, ptr=None):
        self.destroy = destroy
        self.ptr = ptr
        self.INVALID_ARGUMENTS = "Arguments types invalids!"

    def __del__(self):
        if self.ptr:
            self.destroy(self.ptr)
            self.ptr = None

    def validate(self):
        if getattr(self, "ptr", None) is None:
            raise RuntimeError("class pointer is null!")
        if getattr(self, "destroy", None) is None:
            raise RuntimeError("destructor not found!")

    def list_to_ndarray(self, arr, dtype):
        if type(arr) == list or arr.dtype != dtype:
            arr = np.array(arr, dtype=dtype)

        return arr
