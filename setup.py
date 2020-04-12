
#!/usr/bin/env python
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()
#     f.close()

__package_name__ = 'imazero'
__src__ = 'wrapper/imazero.cpp'

extensions = [Extension("wisard",
        [__src__],
        include_dirs=[],
        language='c++' ),
]
setup(
    name=__package_name__,
    ext_package=__package_name__,
    packages=[__package_name__],
    version="0.0.1",
    author='Gabriel Guarisa',
    description='A library of wisard with some models based on wisard',
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    ext_modules=extensions,
    zip_safe=False,
    keywords = ['wisard', 'weithgless', 'neural', 'net'],
    classifiers=[
        "License :: OSI Approved :: MIT License"
    ],
)
