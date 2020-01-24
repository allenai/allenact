# Generating docs

From the `embodied-rl` root directory run
```bash
sphinx-apidoc -f -o docs/source/core . # builds .rst files
sphinx-build -b html docs/source/ docs/build/ # Creates the html files
```
to generate html documentation under the `docs/build` director.