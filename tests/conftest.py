#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-local", action="store_true", default=False, help="Run local tests."
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "localonly: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-local"):
        # --run-local given in cli: do not skip tests
        return
    skip_local = pytest.mark.skip(reason="needs --run-local option to run")
    for item in items:
        if "localonly" in item.keywords:
            if not config.getoption("--run-local"):
                item.add_marker(skip_local)
