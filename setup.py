from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description=long_description,
    author='Moritz Steigerwald',
    license='MIT',
)