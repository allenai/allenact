from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="embodied-rl",
        packages=find_packages(),
        version="0.0.1",
        install_requires=[
            "gym",
            "matplotlib",
            "torchvision~=0.3.0",
            "tensorflow",
            "tensorboardx",
            "torch~=1.1.0",
            "ai2thor",
            "networkx",
            "pillow==6.2.1",
            "setuptools",
            "setproctitle",
            "moviepy",
            "filelock",
        ],
    )
