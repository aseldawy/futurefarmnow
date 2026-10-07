# Running the WSGI server locally to test soil sampling

This guide shows how to start the Python (WSGI/Flask) server on your own machine
and test the **soil sample** endpoint (`/soil/sample.json`). That endpoint reads
soil rasters from the [POLARIS](http://hydrology.cee.duke.edu/POLARIS/) dataset,
so you only need a few POLARIS tiles to try it out.

## 1. Set up the Python environment (one time)

### Use Python 3.11

The packages in `wsgi/requirements.txt` are pinned to versions that ship
prebuilt wheels for Python **3.11** (3.12 and 3.13 should also work). Newer
Python versions, such as 3.14, have no wheels for those versions. In that case
pip tries to compile numpy, scipy and scikit-learn from source, and the install
fails with errors like `Dependency "OpenBLAS" not found`. Check your version:

```shell
python3 --version
```

If it is not 3.11–3.13, do not use the system `python3`. Install Python 3.11
next to it with one of these tools (no root needed, and the system Python is
not touched):

- **[uv](https://docs.astral.sh/uv/)** (recommended):
  `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **pyenv**: `pyenv install 3.11`

### Install the GDAL system library

The GDAL Python package compiles against the system GDAL library. On
Ubuntu/Debian:

```shell
sudo apt install libgdal-dev gdal-bin
```

On macOS, use `brew install gdal`.

### Create the virtual environment

From the repository root, create the environment with **one** of these
options:

```shell
# Option A: uv (downloads Python 3.11 automatically if needed)
uv venv --python 3.11 --seed ffnenv

# Option B: pyenv
"$(pyenv root)/versions/$(pyenv latest 3.11)/bin/python" -m venv ffnenv

# Option C: your system python3 is already 3.11-3.13
#           (on Ubuntu/Debian first run: sudo apt install python3-venv python3-dev)
python3 -m venv ffnenv
```

Then activate it and install the packages:

```shell
source ffnenv/bin/activate      # run this again in every new terminal
python --version                # must print 3.11.x (or 3.12/3.13)
pip install --upgrade pip

# Install everything except GDAL, preferring prebuilt wheels
grep -v '^gdal==' wsgi/requirements.txt > /tmp/ffn-requirements.txt
pip install --prefer-binary -r /tmp/ffn-requirements.txt

# Install the GDAL Python package that matches your system GDAL version
pip install "gdal==$(gdal-config --version)"
```

Notes:

- On Python 3.11 every heavy package (numpy, scipy, scikit-learn, rasterio,
  ...) installs from a prebuilt wheel. Only small pure-Python packages such as
  `spint` are built locally, which is quick. If pip starts compiling numpy or
  scipy, or reports errors about OpenBLAS, a Fortran compiler or `meson`, your
  environment is using the wrong Python version. Delete it and start this step
  over.
- GDAL is installed on its own because the Python package must have the same
  version as the system library, and the version pinned in `requirements.txt`
  may not match yours. This is the one package that is compiled locally, which
  is why `libgdal-dev` is needed.
- If a previous attempt failed, delete the environment and start over:
  `deactivate; rm -rf ffnenv`.
- The `ffnenv/` folder is already in `.gitignore`. To leave the environment,
  run `deactivate`.

If you already use conda, you can run
`conda env create -f wsgi/environment.yml && conda activate ffnenv` instead.

## 2. Create the config file (one time)

The server reads its settings from `wsgi/conf.py`, which is not checked in.
Copy the example:

```shell
cp wsgi/conf.py.example wsgi/conf.py
```

The default `SOIL_DATA_DIR = "data/POLARIS"` is a **relative path**, so it is
resolved against the directory you start the server from. Always start the
server from the repository root (see step 4), or change it to an absolute path.

## 3. Add sample POLARIS data

### Folder layout

Place the data under `data/POLARIS/` at the repository root (`data/` is in
`.gitignore`, so it will not be committed). The layout is:

```text
data/POLARIS/
└── <layer>/                      # e.g. clay, alpha, ph, om, ...
    └── <from>_<to>_compressed/   # depth range in cm, e.g. 0_5_compressed
        ├── lat3334_lon-118-117.tif
        ├── ...more tiles...
        └── _index.csv            # optional, created by gridex.py
```

Rules the code relies on:

- `<layer>` must be one of the names in `SOIL_LAYERS` in `wsgi/conf.py`:
  `alpha, bd, clay, hb, ksat, lambda, n, om, ph, sand, silt, theta_r, theta_s`.
- Each depth folder **must** be named `<from>_<to>_compressed`. Folders that do
  not end in `_compressed` are ignored.
- POLARIS depths are `0_5`, `5_15`, `15_30`, `30_60`, `60_100`, and `100_200`.
  A request like `soildepth=0-15` uses every depth folder that overlaps the
  range and averages them, weighted by thickness.
- The tile file names do not matter; any `.tif` in the folder is used.

### Downloading a few tiles

You can find sample files to download for testing on the following link:

https://drive.google.com/drive/u/0/folders/1CIOUtHuuX4nGa90EGzORLO5Fq_n_He10

Place all the downloaded data under your `data/` directory.

### Optional: build the spatial index

When a depth folder contains many tiles, build an index so that each query only
opens the tiles it touches:

```shell
python wsgi/gridex.py data/POLARIS
```

This writes an `_index.csv` file into every folder that has `.tif` files. If a
folder has no index, the code just reads all of its tiles, which is fine for a
handful of test tiles. **Rerun this command after adding tiles**, because a
stale index will hide the new files.

## 4. Start the server in development mode

From the **repository root**, with the `ffnenv` environment active:

```shell
flask --debug --app wsgi/server.py run
```

The server starts on <http://127.0.0.1:5000>. With `--debug` it restarts
automatically when you edit a Python file. It also serves the files in
`public_html/` at `/public_html/...`, but only in debug mode. Press `Ctrl+C` to
stop it.

## 5. Test the soil sample function

### From the browser

Open <http://127.0.0.1:5000/public_html/soil_sample.html>.

### From the command line

Send a GeoJSON geometry in the request body. Query parameters:

| Parameter    | Meaning                                         | Example |
|--------------|-------------------------------------------------|---------|
| `soildepth`  | Depth range in cm, `from-to`                    | `0-5`   |
| `layer`      | Soil layer; repeat it to request several layers | `clay`  |
| `num_points` | Number of sample points to choose               | `7`     |

```shell
curl -X POST "http://127.0.0.1:5000/soil/sample.json?soildepth=0-5&layer=alpha&layer=clay&num_points=7" \
     -H "Content-Type: application/geo+json" \
     -d '{"type":"MultiPolygon","coordinates":[[[[-117.09285814562507,33.82248877684290],[-117.09538513346878,33.82400211388484],[-117.09783597921066,33.82807099268817],[-117.09774080090969,33.82882290268782],[-117.09183021871685,33.83022678523190],[-117.08819916096476,33.83170205102514],[-117.08712364400770,33.83004118685605],[-117.08410648683598,33.82495865906554],[-117.08734255445930,33.82398783761156],[-117.09285814562507,33.82248877684290]]]]}'
```

A successful response looks like this (values shortened):

```json
{
  "query": { "type": "MultiPolygon", "coordinates": [...] },
  "results": [ { "id": 0, "x": -117.0921, "y": 33.8261 }, ... ],
  "statistics": {
    "layers": {
      "clay":  { "actual": { "min": ..., "max": ..., "mean": ... }, "sample": { ... } },
      "alpha": { "actual": { ... }, "sample": { ... } }
    }
  }
}
```

- `results` lists the chosen sample points.
- `actual` gives statistics over every pixel in the polygon.
- `sample` gives the same statistics over the chosen points only, so you can
  check how well the sample represents the field.

If any requested layer has no data inside the polygon, the server responds with
HTTP **404** and a JSON body that has a `message`, an empty `results` list and
empty `statistics`.

Each request also writes two debug files, `output.csv` (the raw pixel values
that were extracted) and `results.csv`, to the directory you started the server
from. You can delete them at any time.

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `ModuleNotFoundError: No module named 'conf'` | `wsgi/conf.py` is missing (step 2). |
| HTTP 404 with `"message": "No soil data found ..."` | At least one requested layer has no data for the polygon and depth. Either the layer was not downloaded, no tile covers the polygon, the depth folder is not named `*_compressed`, the server was not started from the repository root, or `_index.csv` is out of date (see the next row). |
| `Matched files: []` in the log although the tile covers the polygon | `_index.csv` is out of date or was built by an older `gridex.py` that swapped latitude and longitude. Delete the old index files with `find data/POLARIS -name _index.csv -delete` and rerun `python wsgi/gridex.py data/POLARIS`. |
| Fewer points than `num_points` | The polygon is too small. POLARIS pixels are about 30 m, so a small field holds only a few pixels. |

The server prints `[SAMPLE_DEBUG]` lines to the terminal for every request.
Check them first when something looks wrong.
