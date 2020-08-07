# FAQ

## How do I generate documentation?

Documentation is generated using [mkdoc](https://www.mkdocs.org/) and
[pydoc-markdown](https://pypi.org/project/pydoc-markdown/). 

If you have made no changes to the documentation and only wish to build documentation on your local machine, run the following from within the `embodied-ai` root directory. Note: This will generate HTML documentation within the `site` folder
```bash
mkdocs build
```

If you have made no changes to the documentation and only wish to serve documentation on your local machine (with live reloading of modified documentation), run the following from within the `embodied-ai` root directory.
```bash
mkdocs serve
```


If you have made changes to the documentation, you will need to run a documentation builder script before you serve it on your local machine.
```bash
bash scripts/build_docs.sh
mkdocs serve
```
Alternatively, the `site` directory (once built) can be served as a static webpage on your local machine 
without installing any dependencies by running `python -m http.server 8000` from within the `site` directory.

