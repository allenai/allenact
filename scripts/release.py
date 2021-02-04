import os
from pathlib import Path
from subprocess import getoutput


def make_package(name, verbose=False):
    orig_dir = os.getcwd()
    base_dir = os.path.join(os.path.abspath(os.path.dirname(Path(__file__))), "..")
    os.chdir(base_dir)

    with open(".VERSION", "r") as f:
        __version__ = f.readline()

    output = getoutput(f"python3 {name}/setup.py sdist")
    if verbose:
        print(output)

    os.chdir(os.path.join(base_dir, "dist"))

    output = getoutput(f"tar zxvf {name}-{__version__}.tar.gz")
    if verbose:
        print(output)

    output = getoutput(
        f"cp {name}-{__version__}/{name}/setup.py {name}-{__version__}/setup.py"
    )
    if verbose:
        print(output)

    output = getoutput(f"tar zcvf {name}-{__version__}.tar.gz {name}-{__version__}/")
    if verbose:
        print(output)

    output = getoutput(f"rm -r {name}-{__version__}")
    if verbose:
        print(output)

    os.chdir(orig_dir)


verbose = False
make_package("allenact", verbose)
make_package("allenact_plugins", verbose)
