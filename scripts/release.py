import os
from pathlib import Path
from subprocess import getoutput


def split_req_file(fname):
    pypi = []
    ext = []
    cline = None
    pline = None
    eline = None

    def ensure_section(sec, cursec):
        if cursec != cline:
            if len(sec) > 0:
                sec.append("")
            sec.append(cline)
        return cline

    with open(fname, "r") as f:
        for line in f.readlines():
            line = line[:-1]
            if len(line) == 0:
                continue
            if line[0] == "[":
                cline = line
            else:
                if "@ git+https://github.com/" in line:
                    eline = ensure_section(ext, eline)
                    ext.append(line)
                else:
                    pline = ensure_section(pypi, pline)
                    pypi.append(line)

    return [line + "\n" for line in pypi], [line + "\n" for line in ext]


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

    reqs, deplinks = split_req_file(
        f"{name}-{__version__}/{name}.egg-info/requires.txt"
    )

    with open(f"{name}-{__version__}/{name}.egg-info/dependency_links.txt", "w") as f:
        f.writelines(deplinks)

    with open(f"{name}-{__version__}/{name}.egg-info/requires.txt", "w") as f:
        f.writelines(reqs)

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
