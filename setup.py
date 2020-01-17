from setuptools import find_packages, setup

setup(
    name="embodied-rl",
    packages=find_packages(),
    version="0.0.1",
    install_requires=["gym", "matplotlib", "pybullet"],
)
