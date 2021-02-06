import os
import sys
from pathlib import Path
from subprocess import getoutput


def make_package(name, verbose=False):
    """Prepares sdist for allenact or allenact_plugins."""

    orig_dir = os.getcwd()
    base_dir = os.path.join(os.path.abspath(os.path.dirname(Path(__file__))), "..")
    os.chdir(base_dir)

    with open(".VERSION", "r") as f:
        __version__ = f.readline().strip()

    # generate sdist via setuptools
    output = getoutput(f"{sys.executable} {name}/setup.py sdist")
    if verbose:
        print(output)

    os.chdir(os.path.join(base_dir, "dist"))

    # uncompress the tar.gz sdist
    output = getoutput(f"tar zxvf {name}-{__version__}.tar.gz")
    if verbose:
        print(output)

    # copy setup.py to the top level of the package (required by pip install)
    output = getoutput(
        f"cp {name}-{__version__}/{name}/setup.py {name}-{__version__}/setup.py"
    )
    if verbose:
        print(output)

    # recompress tar.gz
    output = getoutput(f"tar zcvf {name}-{__version__}.tar.gz {name}-{__version__}/")
    if verbose:
        print(output)

    # remove temporary directory
    output = getoutput(f"rm -r {name}-{__version__}")
    if verbose:
        print(output)

    os.chdir(orig_dir)


if __name__ == "__main__":
    verbose = False
    make_package("allenact", verbose)
    make_package("allenact_plugins", verbose)
