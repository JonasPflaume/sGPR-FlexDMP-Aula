import io
import os
import sys
from setuptools import find_packages, setup, Command

# Package Metadata
NAME = "gprplanner"
DESCRIPTION = "active gaussian process regression"
URL = ""
EMAIL = ""
AUTHOR = "Jiayun Li"
REQUIRES_PYTHON = ">=3.6.9"
VERSION = "0.0.0"

REQUIRED = [
    "numpy==1.19.5", "scipy", "matplotlib", "ruckig"
]

setup(
    name=NAME,
    #version=about["__version__"],
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=["gprplanner", "gprplanner.ruckigtest", "gprplanner.gprplanner_gui"],
    install_requires=REQUIRED,
    include_package_data=True,
    license=""
)
