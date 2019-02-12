#!/usr/bin/env python3

import os
from distutils.core import setup

setup(
    name="echonn",
    version='0.0.1',
    description="Time Series Forecasting of Chaotic Dynamical Systems",
    author='Lucas Wilson',
    author_email='lkwilson96@gmail.com',
    url='https://github.com/larkwt96/honors-thesis',
    install_requires=['numpy', 'pygame', 'torch', 'torchvision', 'matplotlib'],
    packages=['echonn'],
    package_dir={'echonn': 'src/echonn'},
    include_package_data=True,
)
