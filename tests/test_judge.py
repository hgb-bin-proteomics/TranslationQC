#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

import pytest


def test1():
    from llm_judge import JudgeConfig

    with pytest.raises(ValueError, match="Parameter 'seeds' must at least length"):
        _ = JudgeConfig(max_retry=5, seeds=[1, 2, 3])


def test2():
    from llm_judge import JudgeConfig

    jc = JudgeConfig.model_validate_toml("config/judge_config.toml")

    assert jc.openai_model == "gpt-6-astra"
    assert jc.openai_thinking_level == "medium"
    assert jc.anthropic_model == "claude-opus-5-5"
    assert jc.anthropic_thinking_level == "medium"
    assert jc.google_model == "gemini-3.8-flash"
    assert jc.google_thinking_level == "medium"
    assert jc.ollama_host == "http://localhost:11434"
    assert jc.ollama_model == "qwen3.8:27b"
    assert jc.ollama_keep_alive == "5m"
    assert jc.max_output_tokens == 4096
    assert jc.max_retry == 2
    assert jc.retry_wait_time == pytest.approx(10.0)
    assert jc.seeds == [41467, 26, 77407, 1]


def test3():
    from llm_judge import JudgeConfig

    jc = JudgeConfig.model_validate_toml("config/judge_config.toml")
    assert jc.openai_model == "gpt-6-astra"


@pytest.mark.localonly
def test4():
    from llm_judge import Judge

    judge = Judge(openai=False, anthropic=False, google=False, ollama="mistral:7b")
    # do something with the judge
    judge.close()
    assert judge.closed


@pytest.mark.localonly
def test5():
    from llm_judge import Judge

    with pytest.raises(
        RuntimeError, match="Ollama model is already given in the configuration!"
    ):
        judge = Judge(
            openai=False,
            anthropic=False,
            google=False,
            ollama="mistral:7b",
            config="config/judge_config.toml",
        )
        judge.close()


@pytest.mark.localonly
def test6():
    from llm_judge import Judge, JudgeConfig

    with pytest.raises(
        RuntimeError, match="Ollama model is already given in the configuration!"
    ):
        judge = Judge(
            openai=False,
            anthropic=False,
            google=False,
            ollama="mistral:7b",
            config=JudgeConfig(),
        )
        judge.close()


@pytest.mark.localonly
def test7():
    from llm_judge import Judge

    with pytest.raises(
        ValueError,
        match="No LLM-providers were setup! Please setup at least one LLM-provider!",
    ):
        judge = Judge(openai=False, anthropic=False, google=False, ollama=False)
        judge.close()


@pytest.mark.localonly
def test8():
    from llm_judge import Judge

    with Judge(
        openai=False, anthropic=False, google=False, ollama="mistral:7b"
    ) as judge:
        jr = judge.score(
            src="The mitochondria is the powerhouse of the cell.",
            mt="Das Mitochondrium ist das Kraftwerk der Zelle.",
            src_lang="English",
            mt_lang="German",
        )
    assert jr.ollama.score > 0.8


@pytest.mark.localonly
def test9():
    from llm_judge import Judge

    judge = Judge(openai=False, anthropic=False, google=False, ollama="mistral:7b")
    jr = judge.score(
        src="The mitochondria is the powerhouse of the cell.",
        mt="Das Mitochondrium ist das Kraftwerk der Zelle.",
        src_lang="English",
        mt_lang="German",
    )
    assert jr.ollama.score > 0.8
    judge.close()
    assert judge.closed


@pytest.mark.localonly
def test10():
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
    judge.close()
    assert judge.closed


@pytest.mark.localonly
def test11():
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
    assert jr.ollama.model == "qwen3.8:27b"
    assert jr.ollama.score > 0.8
    judge.close()
    assert judge.closed


@pytest.mark.localonly
def test12():
    from llm_judge import Judge

    with pytest.raises(RuntimeError, match="Judge instance is already closed!"):
        judge = Judge(
            openai=False,
            anthropic=False,
            google=False,
            ollama=True,
            config="config/judge_config.toml",
        )
        judge.close()
        _jr = judge.score(
            src="The mitochondria is the powerhouse of the cell.",
            mt="Das Mitochondrium ist das Kraftwerk der Zelle.",
            src_lang="English",
            mt_lang="German",
        )
