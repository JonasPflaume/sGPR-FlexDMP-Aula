import io
import os
import sys
from setuptools import find_packages, setup, Command

# Package Metadata
NAME = "trajencoder"
DESCRIPTION = "trajectory encoder with B-spline and DMP"
URL = ""
EMAIL = ""
AUTHOR = "Jiayun Li"
REQUIRES_PYTHON = ">=3.8.6"
VERSION = "0.0.0"

REQUIRED = [
    "numpy==1.19.5", "scipy", "matplotlib"
]

setup(
    name=NAME,
    #version=about["__version__"],
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=["trajencoder", "trajencoder.dmp", "trajencoder.bspline", "trajencoder.flexdmp", "trajencoder.vae"],
    install_requires=REQUIRED,
    include_package_data=True,
    license=""
)
