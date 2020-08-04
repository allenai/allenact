# FAQ

## How do I generate documentation?

Documentation is generated using [mkdoc](https://www.mkdocs.org/) and
[pydoc-markdown](https://pypi.org/project/pydoc-markdown/). 

To generate html documentation (which is placed under the `site` directory) run
```bash
bash scripts/build_docs.sh
```
from within the `embodied-ai` root directory.

To serve the documentation on your local machine (with live 
reloading of modified documentation) run
```bash
python scripts/build_docs.py
mkdocs serve
```
Alternatively, the `site` directory (once built) can be served as a static webpage on your local machine 
without installing any dependencies by running `python -m http.server 8000` from within the `site` directory.