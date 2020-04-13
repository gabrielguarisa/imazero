
#!/usr/bin/env python
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

__package_name__ = 'imazero'
__src__ = 'wrapper/imazero.cpp'

ext_modules = [
    Extension(
        __package_name__,
        [__src__],
        include_dirs=["src/"],
        language='c++'
    ),
]

setup(
    name=__package_name__,
    ext_package=__package_name__,
    packages=[__package_name__],
    author='Gabriel Guarisa',
    description='A library of wisard with some models based on wisard',
    version='0.0.2',
    ext_modules=ext_modules,
    zip_safe=False,
    keywords = ['wisard', 'weithgless', 'neural', 'net'],
    classifiers=[
        "License :: OSI Approved :: MIT License"
    ],
)