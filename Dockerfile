# Reproducible ROOT + Python environment for anubis-ml.
#
# Built on CERN's official ROOT image, so we inherit a working ROOT / PyROOT /
# RDataFrame instead of fighting the (nonexistent) native-Windows install. On top we
# add the ML Python deps. This is the answer to the whole "ROOT is a pain to install"
# saga: `docker build` once, and the environment is identical on any machine or in CI.
#
# Pinned tag for reproducibility (bump deliberately). See:
# https://hub.docker.com/r/rootproject/root/tags
FROM rootproject/root:6.38.00-ubuntu25.10

WORKDIR /app

# The image's /usr/bin/python3 already imports ROOT (via PYTHONPATH=/opt/root/lib) but
# ships without pip, so install it first. Then pip our ML deps into that SAME interpreter
# (--break-system-packages: the container is disposable and we *want* sklearn alongside
# ROOT in one Python).
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-docker.txt .
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements-docker.txt

COPY . .

# Prepend src/ for imports, but PRESERVE the base image's ROOT path (/opt/root/lib) --
# overwriting PYTHONPATH here would make `import ROOT` fail.
ENV PYTHONPATH=/app/src:$PYTHONPATH
CMD ["bash"]
