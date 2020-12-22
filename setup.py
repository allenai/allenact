from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        dependency_links=[
            "git+https://github.com/unnat/babyai.git@ff645fe00ea8412a29bd5e2d6f79ae1595d229a7#egg=babyai"
        ],
        name="allenact",
        packages=find_packages(),
        version="0.1.0",
        install_requires=[
            "gym>=0.17.0,<0.18.0",
            "matplotlib>=3.3.1",
            "torchvision>=0.7.0",
            "tensorboardx==2.1",
            "tensorboard==2.2.1",
            "torch<1.7.0,>=1.6.0",
            "ai2thor<2.6.0,>=2.5.1",
            "networkx",
            "pillow<7.0.0",
            "setuptools",
            "setproctitle",
            "moviepy>=1.0.3",
            "filelock",
            "numpy>=1.19.1",
            "numpy-quaternion",
            "pandas>=1.1.3",
            "gym-minigrid",
            "gin-config",
            "colour",
            "patsy",
            "pyquaternion",
            "scipy>=1.5.2",
            "tqdm",
        ],
    )
