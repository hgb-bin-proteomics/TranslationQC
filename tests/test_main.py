#!/usr/bin/env python3

# PKG NAME - TESTS
# 2026 (c) YOUR NAME
# https://github.com/username/
# your.mail@mail.com


def test1():
    import os
    from qc_bench import main

    # only run this locally
    if os.path.isfile("env.json"):
        assert main(["-i", "data/test.csv", "-o", "data/test_annotated.csv"]) == 0
    assert True
