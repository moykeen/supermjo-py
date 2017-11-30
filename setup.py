# -*- coding: utf-8 -*-

import sys, os
from setuptools import find_packages, setup


with open(os.path.join(os.path.dirname(__file__), 'README.rst'), 'rb') as readme:
    README = readme.read()


setup(
    name="supermjo-py",
    version="0.1.0",
    description='Python interface to Super-Mjograph',
    long_description=README.decode('utf-8'),
    url="https://github.com/moykeen/supermjo-py",
    author="Makoto Tanahashi",
    author_email="makoto.mjo@gmail.com",
    license='MIT',
    classifiers=[
        "Operating System :: MacOS :: MacOS X",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
    keywords="graph plot visualization",
    packages=find_packages(),
    install_requires=["py-applescript", "pyobjc", "numpy", "pandas"]
)
