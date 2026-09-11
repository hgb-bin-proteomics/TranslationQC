# Dockerfile with PKG NAME
# author: YOUR NAME
# version: 1.0.0

FROM python:3.14

LABEL maintainer="you@yourname.com"

RUN mkdir pkg
COPY ./ pkg/
WORKDIR pkg

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir uv
RUN uv sync --no-dev --no-cache

CMD  ["uv", "run", "--no-sync", "python"]
