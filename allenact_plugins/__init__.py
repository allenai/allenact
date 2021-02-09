import os

from pathlib import Path

__version__ = None

try:
    from allenact_plugins._version import __version__
except ModuleNotFoundError:
    try:
        with open(os.path.join(os.path.dirname(Path(__file__)), "..", ".VERSION")) as f:
            __version__ = f.read().strip()
        pass
    except ModuleNotFoundError:
        pass
