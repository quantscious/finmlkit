# setup.py
from setuptools import setup, find_packages
import os


def get_version():
    """Read version from _version.py without importing."""
    here = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(here, 'finmlkit', '_version.py')

    with open(version_file, 'r', encoding='utf-8') as f:
        content = f.read()
        for line in content.splitlines():
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")

    raise RuntimeError('Cannot find version string')

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="finmlkit",
    version=get_version(),
    description="Financial ML toolkit",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Dániel Terbe",
    author_email="dainel@terbe.dev",
    url="https://github.com/quantscious/finmlkit",
    license="MIT",
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=6.0",
            "flake8",
            "black",
        ]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)