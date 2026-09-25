#!/usr/bin/env python3

import pytest


@pytest.mark.localonly
def test1():
    from llm_judge import Judge

    judge = Judge(openai=False, anthropic=False, google=False, ollama="mistral:7b")
    jr = judge.score(
        src="The mitochondria is the powerhouse of the cell.",
        mt="Das Mitochondrium ist das Kraftwerk der Zelle.",
        src_lang="English",
        mt_lang="German",
    )
    assert str(type(jr)) == "<class 'llm_judge._judge.JudgeResult'>"
    assert jr.openai is None
    assert jr.anthropic is None
    assert jr.google is None
    assert jr.ollama is not None
    assert str(type(jr.ollama)) == "<class 'llm_judge._judge.JudgeModelResult'>"
    assert jr.ollama.model == "mistral:7b"
    assert jr.ollama.score > 0.8


@pytest.mark.localonly
def test2():
    from llm_judge import Judge

    judge = Judge(
        openai=False,
        anthropic=False,
        google=False,
        ollama=True,
        config="config/judge_config.toml",
    )
    jr = judge.score(
        src="The mitochondria is the powerhouse of the cell.",
        mt="Das Mitochondrium ist das Kraftwerk der Zelle.",
        src_lang="English",
        mt_lang="German",
    )
    assert str(type(jr)) == "<class 'llm_judge._judge.JudgeResult'>"
    assert jr.openai is None
    assert jr.anthropic is None
    assert jr.google is None
    assert jr.ollama is not None
    assert str(type(jr.ollama)) == "<class 'llm_judge._judge.JudgeModelResult'>"
    assert jr.ollama.model == "gemma4:e4b"
    assert jr.ollama.score > 0.8
