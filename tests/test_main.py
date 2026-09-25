#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

import pytest


def test1():
    from llm_judge import main

    with pytest.raises(SystemExit, match="0") as e:
        _ = main(["-h"])

    assert e.value.code == 0


@pytest.mark.localonly
def test2():
    from llm_judge import main

    assert (
        main(
            [
                "-i",
                "data/test.csv",
                "-o",
                "data/test_annotated.csv",
                "--openai",
                "--anthropic",
                "--google",
                "--ollama",
            ]
        )
        == 0
    )
