from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="embodied-rl",
        packages=find_packages(),
        version="0.0.1",
        install_requires=[
            "gym",
            "matplotlib",
            "pybullet",
            "torchvision~=0.3.0",
            "tensorflow",
            "tensorboardx",
            "ai2thor",
            "networkx",
            "pillow==6.2.1",
            "setuptools",
            "torch==1.1.0",
        ],
    )
