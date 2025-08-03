# setup.py
from setuptools import setup, find_packages
from finmlkit.__version__ import __version__

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="finmlkit",
    version=__version__,
    description="Financial ML toolkit",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Dániel Terbe",
    url="https://github.com/quantscious/finmlkit",
    license="MIT",
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=">=3.9",
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)