import os
from pathlib import Path

ABS_PATH_OF_TOP_LEVEL_DIR = os.path.abspath(os.path.dirname(Path(__file__)))
ABS_PATH_OF_DOCS_DIR = os.path.join(ABS_PATH_OF_TOP_LEVEL_DIR, "docs")
