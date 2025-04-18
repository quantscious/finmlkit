# setup.py
from setuptools import setup, find_packages

setup(
    name='finmlkit',
    version='0.1',
    packages=find_packages(),  # will find the inner finmlkit package
    install_requires=[
        # add dependencies if needed
    ]
)