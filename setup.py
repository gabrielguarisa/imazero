
#!/usr/bin/env python
from setuptools import setup, Extension, find_packages
# from os import path

# here = path.abspath(path.dirname(__file__))

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
    packages=[__package_name__, *find_packages()],
    author='Gabriel Guarisa',
    description='A library of wisard with some models based on wisard',
    version='0.0.2',
    ext_modules=ext_modules,
    zip_safe=False,
    install_requires=['numpy', 'pybind11', 'wisardpkg==2.0.0a6', 'sklearn', 'pandas', 'scikit-image', 'scipy', 'get-mnist==0.2.2'],
    keywords = ['wisard', 'weithgless', 'neural', 'net'],
    classifiers=[
        "License :: OSI Approved :: MIT License"
    ],
)