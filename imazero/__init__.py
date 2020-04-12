from ctypes import *
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

__zerar = flib.zerar
__zerar.restype = c_void_p
__zerar.argtypes = [c_void_p, c_int]

def zerar(m):
   if m.dtype.name != 'uint8':
       raise RuntimeError("m must have type uint8")
  
   return __zerar(m.ctypes.data, m.size)
