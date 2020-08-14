# FAQ

## How do I generate documentation?

Documentation is generated using [mkdoc](https://www.mkdocs.org/) and
[pydoc-markdown](https://pypi.org/project/pydoc-markdown/). 


### Building documentation locally

If you have made no changes to the documentation and only wish to build documentation on your local machine, run the following from within the `allenact` root directory. Note: This will generate HTML documentation within the `site` folder
```bash
mkdocs build
```

### Serving documentation locally

If you have made no changes to the documentation and only wish to serve documentation on your local machine (with live reloading of modified documentation), run the following from within the `allenact` root directory.
```bash
mkdocs serve
```

Then navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### Modifying and serving documentation locally

If you have made changes to the documentation, you will need to run a documentation builder script before you serve it on your local machine.
```bash
bash scripts/build_docs.sh
mkdocs serve
```

Then navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

Alternatively, the `site` directory (once built) can be served as a static webpage on your local machine 
without installing any dependencies by running `python -m http.server 8000` from within the `site` directory.

