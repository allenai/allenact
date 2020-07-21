from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        dependency_links=[
            "git+https://github.com/unnat/babyai.git@ff645fe00ea8412a29bd5e2d6f79ae1595d229a7#egg=babyai"
        ],
        name="embodied-rl",
        packages=find_packages(),
        version="0.0.1",
        install_requires=[
            "gym",
            "matplotlib",
            "torchvision~=0.5.0",
            "tensorflow",
            "tensorboardx",
            "torch~=1.4.0",
            "ai2thor==2.3.9",
            "networkx",
            "pillow==6.2.1",
            "setuptools",
            "setproctitle",
            "moviepy",
            "filelock",
            "numpy-quaternion",
            "pandas",
            "seaborn",
            "statsmodels",
            "gym-minigrid",
            "gin-config",
            "localreg",
        ],
    )
