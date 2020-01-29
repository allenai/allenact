# Generating documentation

Documentation is generated using [mkdoc](https://www.mkdocs.org/) and
[pydoc-markdown](https://pypi.org/project/pydoc-markdown/). 

To generate html documentation under the `site` directory run
```bash
bash scripts/build_docs.sh
```
from within the `embodied-rl` root directory run.

To serve the documentation on your local machine (with live 
reloading of modified documentation) run
```bash
python scripts/build_docs.py
mkdocs serve
```