from importlib.metadata import files

from setuptools import find_packages, setup
from setuptools.command.install import install

setup(
    name='diffenix',
    packages=find_packages(include=['diffenix']),
    version='0.0.0.1',
    install_requires=[],
    description='A library to model the gas evolution in debris discs',
    author='Paul Huet',
    readme="README.md",
    license_files="LICENSE",
    setup_requires=['pytest-runner'],
)
