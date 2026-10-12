# Deploying the WSGI server behind Apache with mod_wsgi

This guide deploys the Python (Flask) part of FutureFarmNow on a Linux server.
It runs inside Apache through **mod_wsgi**, using a Python **virtual
environment** (no conda). The commands are for Ubuntu/Debian. For Red Hat
Enterprise Linux, Rocky Linux, AlmaLinux or Fedora, also read
[Red Hat-family servers](#red-hat-family-servers-rhel-rocky-alma-fedora),
which lists the commands and paths that differ.

The Java backend and the Next.js client are deployed separately; see the
[README](../README.md). This guide only covers the endpoints that the Python
code serves:

| Endpoint | Module |
|----------|--------|
| `/soil/sample.json` | `wsgi/soil_sample.py` |
| `/soil/singlepolygon.json` | `wsgi/soil_stats.py` |
| `/ndvi/singlepolygon.json` | `wsgi/ndvi_timeseries.py` |
| `/etmap`, `/etmap/...` | `wsgi/app/et_map/` |
| `/vectors.json` | `wsgi/server.py` |

To run the server on your own machine for testing instead, see
[wsgi-dev-soil-sample.md](wsgi-dev-soil-sample.md).

## How the pieces fit together

```text
browser ──> Apache ──(mod_rewrite)──> /wsgi/<endpoint>
                                         │
                                         ▼
                         mod_wsgi daemon process (user www-data)
                         Python 3.11 from /var/www/ffn.example.com/ffnenv
                         runs wsgi/wsgi.py  ->  Flask app in wsgi/server.py
```

Two rules shape the whole setup:

1. **mod_wsgi must be built for the same Python as the virtual environment.**
   mod_wsgi embeds the Python interpreter inside Apache, and a module built for
   one Python version cannot load packages installed for another. The Ubuntu
   package `libapache2-mod-wsgi-py3` is built for the system Python, which may
   not be a version this project supports. So this guide builds mod_wsgi with
   `pip` **inside the virtual environment**, which guarantees a match.
2. **The Python version must be 3.11, 3.12 or 3.13.** The packages in
   `wsgi/requirements.txt` are pinned to versions that only install there. We
   use [uv](https://docs.astral.sh/uv/) to get Python 3.11 regardless of what
   the server ships with.

Throughout this guide:

- `/var/www/ffn.example.com` is the deployment directory. Replace it
  everywhere with your own path.
- `www-data` is the user Apache runs as on Ubuntu/Debian.

## 1. Install system packages

```shell
sudo apt install apache2 apache2-dev libgdal-dev gdal-bin curl
sudo a2enmod rewrite
```

On Red Hat-family servers, use this instead (GDAL comes from EPEL):

```shell
sudo dnf install epel-release        # on RHEL itself, see the Red Hat section below
sudo dnf install httpd httpd-devel gdal gdal-devel gcc curl
```

- `apache2-dev` (`httpd-devel` on Red Hat) provides `apxs`, which
  `pip install mod_wsgi` needs to compile the Apache module.
- `libgdal-dev` is needed to compile the GDAL Python package.

If `libapache2-mod-wsgi-py3` is installed, disable it so it does not conflict
with the module built in step 4:

```shell
sudo a2dismod wsgi
```

## 2. Lay out the deployment directory

The final layout looks like this:

```text
/var/www/ffn.example.com/
├── ffnenv/          # Python virtual environment (step 3)
├── wsgi/            # copy of the repository's wsgi/ folder
│   ├── wsgi.py      # entry point that mod_wsgi loads
│   ├── server.py
│   └── conf.py      # created from conf.py.example (step 5)
├── data/            # POLARIS, NDVI, vectors.json, ...
└── public_html/     # static client files (optional, see README)
```

Create the directory and copy the code from your local checkout:

```shell
sudo mkdir -p /var/www/ffn.example.com
sudo chown $USER:www-data /var/www/ffn.example.com

# From your local checkout:
rsync -av --exclude=__pycache__ --exclude=conf.py wsgi/ remote_host:/var/www/ffn.example.com/wsgi/
```

`--exclude=conf.py` keeps your local development config from overwriting the
server's config on later deployments.

Put the data under `/var/www/ffn.example.com/data`. You can use a symbolic link
if the data lives on a bigger disk. The folder layout is the same as in
development; see
[wsgi-dev-soil-sample.md](wsgi-dev-soil-sample.md#3-add-sample-polaris-data).
After adding tiles, build the spatial index:

```shell
/var/www/ffn.example.com/ffnenv/bin/python /var/www/ffn.example.com/wsgi/gridex.py /var/www/ffn.example.com/data/POLARIS
```

(Run this after step 3, once the environment exists.)

## 3. Create the virtual environment

### Install Python 3.11 in a shared location

By default uv installs Python under your home directory (`~/.local/share/uv`),
which Apache's `www-data` user usually cannot read. Install it under `/opt`
instead:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh     # then open a new terminal

sudo mkdir -p /opt/uv-python
sudo chown $USER /opt/uv-python
UV_PYTHON_INSTALL_DIR=/opt/uv-python uv python install 3.11
```

This Python includes the shared `libpython3.11.so` that mod_wsgi links against.

### Create the environment and install the packages

```shell
cd /var/www/ffn.example.com
UV_PYTHON_INSTALL_DIR=/opt/uv-python uv venv --python 3.11 --seed ffnenv
source ffnenv/bin/activate

# Stop here if the environment does not have the right Python version
python -c 'import sys; v = sys.version_info[:2]; print("Python %d.%d" % v); assert (3, 11) <= v <= (3, 13), "Wrong Python version"'

pip install --upgrade pip
grep -v '^gdal==' wsgi/requirements.txt > /tmp/ffn-requirements.txt
pip install --prefer-binary -r /tmp/ffn-requirements.txt
pip install "gdal==$(gdal-config --version)"
```

The reasons for each step, and fixes for common pip errors, are in
[wsgi-dev-soil-sample.md](wsgi-dev-soil-sample.md#1-set-up-the-python-environment-one-time).

## 4. Build mod_wsgi and load it into Apache

mod_wsgi is compiled from C source and needs the **Python header files**
(`Python.h`) of the Python the environment uses:

- **uv Python (step 3):** the headers are included; nothing to install.
- **System Python:** install its development package first, for example
  `sudo apt install python3.12-dev` on Ubuntu or
  `sudo dnf install python3.12-devel` on Red Hat. Match the version shown by
  `python --version` inside the environment.

Without the headers, the build fails with
`fatal error: Python.h: No such file or directory`.

With the environment still active:

```shell
pip install mod_wsgi
mod_wsgi-express module-config
```

The second command prints two lines similar to these:

```apache
LoadModule wsgi_module "/var/www/ffn.example.com/ffnenv/lib/python3.11/site-packages/mod_wsgi/server/mod_wsgi-py311.cpython-311-x86_64-linux-gnu.so"
WSGIPythonHome "/var/www/ffn.example.com/ffnenv"
```

Save **only the `LoadModule` line** in Apache's module configuration. The
`WSGIPythonHome` line is not needed, because the site configuration in step 6
sets `python-home` on the daemon process instead. Where the line goes depends on
the Linux distribution. If you are not sure which one you have, run
`ls -d /etc/apache2 /etc/httpd`: the folder that exists tells you.

**Ubuntu/Debian** (`/etc/apache2` exists):

```shell
mod_wsgi-express module-config | grep LoadModule | sudo tee /etc/apache2/mods-available/wsgi_venv.load
sudo a2enmod wsgi_venv
```

**Red Hat, Rocky, Alma, Fedora** (`/etc/httpd` exists). Every `.conf` file in
`/etc/httpd/conf.modules.d/` is loaded automatically, so there is no enable
command:

```shell
mod_wsgi-express module-config | grep LoadModule | sudo tee /etc/httpd/conf.modules.d/10-wsgi-venv.conf
# If the distribution's mod_wsgi package is installed, remove it so that only one module is loaded
sudo dnf remove python3-mod_wsgi
```

Run these commands with the virtual environment active, so that
`mod_wsgi-express` is the one inside `ffnenv`. To check the result, run
`sudo apachectl -M | grep wsgi`, which should print `wsgi_module (shared)`.

Rebuild mod_wsgi with `pip install --force-reinstall mod_wsgi` whenever you
recreate the virtual environment or change its Python version.

## 5. Configure the application

```shell
cd /var/www/ffn.example.com
cp wsgi/conf.py.example wsgi/conf.py
```

Edit `wsgi/conf.py` and use **absolute paths**:

```python
NDVI_DATA_DIR = "/var/www/ffn.example.com/data/NDVI"
SOIL_DATA_DIR = "/var/www/ffn.example.com/data/POLARIS"
```

The example file uses relative paths, which resolve against the process's
working directory. Under Apache that directory is whatever `home=` says in
step 6, so absolute paths are less fragile.

### File permissions

The daemon process runs as `www-data`. It needs:

- **Read** access to `ffnenv/`, `wsgi/`, `data/` and `/opt/uv-python`.
- **Write** access to the working directory (`/var/www/ffn.example.com`). Each
  soil sample request writes `output.csv` and `results.csv` there.
- **Write** access to `wsgi/etmap.db` and `wsgi/results/` if you use ET Map.
  These are the defaults in `wsgi/app/et_map/etmap_modules/config.py`.

```shell
cd /var/www/ffn.example.com
sudo chgrp -R www-data . /opt/uv-python
sudo chmod -R g+rX . /opt/uv-python
sudo chmod g+w . wsgi
[ -f wsgi/etmap.db ] && sudo chmod g+w wsgi/etmap.db
```

If `data` is a symbolic link, also grant read access on its target, and make
sure every parent directory of that target is at least `o+x` or `g+x` for
`www-data`.

## 6. Configure the Apache site

Create the site file, or add the WSGI-related lines to your existing
`<VirtualHost>`:

- Ubuntu/Debian: `/etc/apache2/sites-available/ffn.example.com.conf`
- Red Hat family: `/etc/httpd/conf.d/ffn.example.com.conf`. Also change
  `www-data` to `apache` and `${APACHE_LOG_DIR}/` to `logs/` in the example
  below.

Define `WSGIDaemonProcess ffn` only **once** in all of Apache's configuration.
If the site has both an HTTP (`:80`) and an HTTPS (`:443`) `<VirtualHost>`, put
the `WSGIDaemonProcess` line outside both blocks, and keep the other WSGI lines
inside each one.

```apache
<VirtualHost *:80>
    ServerName ffn.example.com
    DocumentRoot /var/www/ffn.example.com/public_html

    # One daemon process group running the virtual environment's Python.
    #   python-home: the virtual environment
    #   home:        working directory (relative paths and CSV output land here)
    WSGIDaemonProcess ffn \
        user=www-data group=www-data \
        python-home=/var/www/ffn.example.com/ffnenv \
        home=/var/www/ffn.example.com \
        processes=2 threads=5 \
        request-timeout=300 \
        display-name=%{GROUP}
    WSGIProcessGroup ffn
    # numpy, GDAL and rasterio do not work in Python sub-interpreters
    WSGIApplicationGroup %{GLOBAL}

    WSGIScriptAlias /wsgi /var/www/ffn.example.com/wsgi/wsgi.py \
        process-group=ffn application-group=%{GLOBAL}

    <Directory /var/www/ffn.example.com/wsgi>
        <Files wsgi.py>
            Require all granted
        </Files>
    </Directory>

    # The client calls /futurefarmnow-backend-<version>/<endpoint>.
    # Send the Python endpoints to mod_wsgi; everything else goes to the
    # Java server (see README).
    RewriteEngine On
    RewriteRule ^/futurefarmnow-backend-[\.0-9]*(-[\w\d]+)?/(soil/sample\.json|ndvi/singlepolygon\.json|etmap.*)$ /wsgi/$2 [PT,L]

    ErrorLog ${APACHE_LOG_DIR}/ffn-error.log
    CustomLog ${APACHE_LOG_DIR}/ffn-access.log combined
</VirtualHost>
```

Notes:

- **`WSGIApplicationGroup %{GLOBAL}` is required.** Without it, numpy and GDAL
  can hang or crash on the first request.
- **`request-timeout`** restarts a process that is stuck in a request for that
  many seconds. Soil sampling on a large field can take a minute or more, so
  do not set it too low. Apache's own `Timeout` (300 s by default) must be at
  least as large.
- **`processes` and `threads`**: start small. Every process loads the full
  scientific Python stack (several hundred MB of memory).
- **The rewrite rule** lists the endpoints currently routed to Python in
  production. Add `soil/singlepolygon\.json` or `vectors\.json` to it if you
  want Python to serve those instead of Java.
- Do not set environment variables for the app with `SetEnv`. mod_wsgi does
  not copy them into `os.environ`. To override ET Map paths
  (`ETMAP_DATA_DIR`, `ETMAP_RESULTS_DIR`, `ETMAP_DB_PATH`), export them in
  `/etc/apache2/envvars` instead.

Check the configuration and reload Apache. **Always run the config test
first.** A broken configuration stops Apache from starting, and
`systemctl status` does not show the reason, but `configtest` does.

```shell
# Ubuntu/Debian
sudo a2ensite ffn.example.com
sudo apache2ctl configtest
sudo systemctl reload apache2

# Red Hat family (files in conf.d/ are loaded automatically)
sudo apachectl configtest
sudo systemctl reload httpd
```

## 7. Check that it works

```shell
# Directly through the /wsgi alias
curl http://ffn.example.com/wsgi/etmap/healthcheck.json
# -> {"module":"et_map","status":"ok"}

# Through the same URL the client uses
curl -X POST "http://ffn.example.com/futurefarmnow-backend-0.3-RC1/soil/sample.json?soildepth=0-5&layer=alpha&num_points=5" \
     -H "Content-Type: application/geo+json" \
     -d '{"type":"Polygon","coordinates":[[[-117.093,33.822],[-117.084,33.822],[-117.084,33.83],[-117.093,33.83],[-117.093,33.822]]]}'
```

A 404 response with `"message": "No soil data found ..."` means that the
application is running but has no data for that polygon. Check
`SOIL_DATA_DIR` and the data permissions.

## 8. Deploying updates

```shell
# From your local checkout
rsync -av --exclude=__pycache__ --exclude=conf.py wsgi/ remote_host:/var/www/ffn.example.com/wsgi/

# On the server: restart only the Python processes
touch /var/www/ffn.example.com/wsgi/wsgi.py
```

Touching `wsgi.py` makes mod_wsgi restart the daemon processes on the next
request, without reloading Apache. If `wsgi/requirements.txt` changed, rerun
the `pip install` commands from step 3 first.

## ET Map worker

ET Map jobs are processed by a separate long-running worker, not by Apache.
Run it with the same virtual environment and as a user who can write to the
same database and results directory:

```shell
cd /var/www/ffn.example.com/wsgi
/var/www/ffn.example.com/ffnenv/bin/python -m app.et_map.worker
```

See the "ET Map worker process" step in the [README](../README.md) for running
it as a systemd service.

## Red Hat-family servers (RHEL, Rocky, Alma, Fedora)

The steps above work on these systems with the following substitutions. These
instructions have not been tested on a Red Hat-family server yet.

| Ubuntu/Debian | Red Hat family |
|---------------|----------------|
| `apt install ...` | `dnf install ...` (packages in step 1) |
| Apache user/group `www-data` | `apache` |
| `apache2-dev`, `python3.X-dev` | `httpd-devel`, `python3.X-devel` |
| `libapache2-mod-wsgi-py3` (step 1: `a2dismod wsgi`) | `python3-mod_wsgi` (remove it: `sudo dnf remove python3-mod_wsgi`) |
| `a2enmod rewrite` | Not needed; `mod_rewrite` is loaded by default |
| `/etc/apache2/mods-available/wsgi_venv.load` + `a2enmod wsgi_venv` | Write the `LoadModule` line to `/etc/httpd/conf.modules.d/10-wsgi-venv.conf` |
| `/etc/apache2/sites-available/ffn.example.com.conf` + `a2ensite` | `/etc/httpd/conf.d/ffn.example.com.conf` (loaded automatically) |
| `${APACHE_LOG_DIR}/ffn-error.log` | `logs/ffn-error.log` (that is, `/var/log/httpd/ffn-error.log`) |
| `/etc/apache2/envvars` | A systemd override: `sudo systemctl edit httpd`, then add `Environment=NAME=value` lines under `[Service]` |
| `apache2ctl configtest`, `systemctl reload apache2` | `apachectl configtest`, `systemctl reload httpd` |

Loading the module (replaces the two commands at the end of step 4):

```shell
mod_wsgi-express module-config | grep LoadModule | sudo tee /etc/httpd/conf.modules.d/10-wsgi-venv.conf
```

**EPEL on RHEL itself.** `epel-release` is not in the RHEL repositories.
Enable CodeReady Builder and install EPEL from its URL (use the major version
of your release instead of 9):

```shell
sudo subscription-manager repos --enable codeready-builder-for-rhel-9-$(arch)-rpms
sudo dnf install https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
```

On Rocky and Alma, `sudo dnf config-manager --set-enabled crb` may also be
needed before installing `gdal-devel`.

### SELinux

Red Hat systems usually run SELinux in enforcing mode, which blocks Apache from
loading libraries or writing files in places not labeled for it, even when the
Unix permissions are correct. Check the mode with `getenforce`. If it prints
`Enforcing`, label the directories (requires `policycoreutils-python-utils`
for `semanage`):

```shell
ROOT=/var/www/ffn.example.com

# Python, the virtual environment and mod_wsgi contain shared libraries that
# Apache loads, so label them as libraries
sudo semanage fcontext -a -t lib_t "/opt/uv-python(/.*)?"
sudo semanage fcontext -a -t lib_t "$ROOT/ffnenv(/.*)?"

# Code and data: read only
sudo semanage fcontext -a -t httpd_sys_content_t "$ROOT/wsgi(/.*)?"
sudo semanage fcontext -a -t httpd_sys_content_t "$ROOT/data(/.*)?"

# Files the application writes (working directory CSVs, ET Map database and results)
sudo semanage fcontext -a -t httpd_sys_rw_content_t "$ROOT(/(output|results)\.csv)?"
sudo semanage fcontext -a -t httpd_sys_rw_content_t "$ROOT/wsgi/etmap\.db"
sudo semanage fcontext -a -t httpd_sys_rw_content_t "$ROOT/wsgi/results(/.*)?"

sudo restorecon -Rv /opt/uv-python $ROOT
```

If `data/` is a symbolic link to another disk, label the link's target path
instead.

If something is still blocked, the audit log names the denied file and action:

```shell
sudo ausearch -m avc -ts recent
```

A denial with `execmem` comes from numerical libraries that generate code at
run time; allow it with `sudo setsebool -P httpd_execmem 1`.

## Troubleshooting

All Python errors and `print` output go to the site's error log:

```shell
sudo tail -f /var/log/apache2/ffn-error.log   # Red Hat: /var/log/httpd/ffn-error.log
```

| Symptom | Likely cause and fix |
|---------|----------------------|
| `pip install mod_wsgi` fails with `fatal error: Python.h: No such file or directory` | The environment uses a system Python without its development headers. Install `python3.X-dev` (Ubuntu) or `python3.X-devel` (Red Hat) for that version, or recreate the environment with uv's Python (step 3). |
| `pip install mod_wsgi` fails with `apxs` not found | Install `apache2-dev` (Ubuntu) or `httpd-devel` (Red Hat). |
| `Cannot load .../mod_wsgi-py311...so: libpython3.11.so.1.0: cannot open shared object file` | `www-data` cannot read `/opt/uv-python`, or Python was installed under a home directory. Redo the shared install in step 3 and the permissions in step 5. |
| `ModuleNotFoundError: No module named 'flask'` (or numpy, ...) | `python-home` does not point to the virtual environment, or mod_wsgi was built for a different Python. Check the `LoadModule` path and rebuild mod_wsgi (step 4). |
| `Fatal Python error: init_fs_encoding` or `No module named 'encodings'` | Same as above: `python-home` is wrong, or mod_wsgi and the environment use different Python versions. |
| Apache refuses to start: `module wsgi_module is already loaded` | Both the Ubuntu package and the pip-built module are enabled. Run `sudo a2dismod wsgi`. |
| First request hangs forever | `WSGIApplicationGroup %{GLOBAL}` is missing. |
| `ModuleNotFoundError: No module named 'conf'` | `wsgi/conf.py` was not created on the server (step 5). |
| `Permission denied: 'output.csv'` | The working directory set by `home=` is not writable by `www-data` (step 5). On Red Hat, also check the SELinux labels. |
| `Permission denied` or `failed to map segment from shared object` although file permissions look right | SELinux is blocking Apache; see [SELinux](#selinux). |
| `Layer directory does not exist` or 404 "No soil data found" | `SOIL_DATA_DIR` is relative or wrong, `data/` is not readable, or `_index.csv` is out of date. Use absolute paths, then rebuild the index with `gridex.py`. |
| `Timeout when reading response headers from daemon process` | The request ran longer than `request-timeout` or Apache's `Timeout`. Increase both. |
