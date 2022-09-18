import io
import os
import sys
from setuptools import find_packages, setup, Command

# Package Metadata
NAME = "aula"
DESCRIPTION = "augmented lagrangian constrained optimizer"
URL = ""
EMAIL = ""
AUTHOR = "Jiayun Li"
REQUIRES_PYTHON = ">=3.8.6"
VERSION = "0.0.0"

REQUIRED = [
    "numpy==1.19.5"
]

setup(
    name=NAME,
    #version=about["__version__"],
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=["aula", "dnewton", "optimization_algorithms"],
    install_requires=REQUIRED,
    include_package_data=True,
    license=""
)
