#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

import os
import time
import json
import logging
from openai import OpenAI
from anthropic import Anthropic
from google.genai import Client as Google
from google.genai import types as GoogleTypes  # noqa: N812
from ollama import Client as Ollama
from ollama import GenerateResponse as OllamaGenerateResponse
from ollama import ResponseError as OllamaResponseError
from ollama import pull as pull_ollama_model
from pydantic import BaseModel, Field, ConfigDict, computed_field

from typing import Optional, Annotated, Any, Literal

from ._translation import QualityEstimation
from ._constants import MAX_RETRY, MAX_OUTPUT_TOKENS, RETRY_WAIT_TIME, SEEDS
from ._constants import OPENAI_MODEL, OPENAI_THINKING_LEVEL
from ._constants import ANTHROPIC_MODEL, ANTHROPIC_THINKING_LEVEL
from ._constants import GOOGLE_MODEL, GOOGLE_THINKING_LEVEL
from ._constants import OLLAMA_HOST, OLLAMA_DEFAULT_MODEL, OLLAMA_KEEP_ALIVE

logger = logging.getLogger(__name__)


class JudgeModelResult(BaseModel):
    r"""Quality estimation result for one specific query and LLM."""

    model: Annotated[str, Field(frozen=True, description="The name of the LLM.")]
    r"""The name of the LLM."""
    prompt: Annotated[
        str, Field(frozen=True, description="The prompt used with the LLM.")
    ]
    r"""The prompt used with the LLM."""
    status: Annotated[
        Literal["ok", "error"],
        Field(frozen=True, description="Status of the response."),
    ]
    r"""Status of the response. Can be 'ok' or 'error'. Only 'ok' denote a successful response."""
    parameters: Annotated[
        Optional[dict[str, str]],
        Field(frozen=True, description="Additional parameters passed to the LLM."),
    ]
    r"""Additional parameters passed to the LLM."""
    response: Annotated[
        Optional[str], Field(frozen=True, description="The LLM response as raw text.")
    ]
    r"""The LLM response as raw text."""
    quality_estimation: Annotated[
        Optional[QualityEstimation],
        Field(frozen=True, description="The quality estimation returned by the LLM."),
    ]
    r"""The quality estimation returned by the LLM."""
    quality_estimation_dict: Annotated[
        Optional[dict[str, Any]],
        Field(
            frozen=True,
            description="The quality estimation returned by the LLM as json.",
        ),
    ]
    r"""The quality estimation returned by the LLM as json."""
    model_config = ConfigDict(
        validate_assignment=True, strict=True, str_strip_whitespace=True
    )
    r"""
    Pydantic configuration for the underlying validation model.
    """

    @computed_field(description="Quality estimation score returned by the LLM.")
    @property
    def score(self) -> float:
        r"""The quality estimation score returned by the LLM (if any - maybe 'nan')."""
        if self.quality_estimation is not None:
            return self.quality_estimation.quality_estimation_value
        if self.quality_estimation_dict is not None:
            if "quality_estimation_value" in self.quality_estimation_dict:
                return float(self.quality_estimation_dict["quality_estimation_value"])
        return float("nan")


class JudgeResult(BaseModel):
    r"""Quality estimation results for one specific query and all LLMs."""

    openai: Annotated[
        Optional[JudgeModelResult],
        Field(frozen=True, description="Result of the OpenAI model."),
    ]
    r"""Result of the OpenAI model. 'None' if the model was not setup."""
    anthropic: Annotated[
        Optional[JudgeModelResult],
        Field(frozen=True, description="Result of the Anthropic model."),
    ]
    r"""Result of the Anthropic model. 'None' if the model was not setup."""
    google: Annotated[
        Optional[JudgeModelResult],
        Field(frozen=True, description="Result of the Google model."),
    ]
    r"""Result of the Google model. 'None' if the model was not setup."""
    ollama: Annotated[
        Optional[JudgeModelResult],
        Field(frozen=True, description="Result of the Ollama model."),
    ]
    r"""Result of the Ollama model. 'None' if the model was not setup."""
    model_config = ConfigDict(
        validate_assignment=True, strict=True, str_strip_whitespace=True
    )
    r"""
    Pydantic configuration for the underlying validation model.
    """


class _OpenAIModel:
    @staticmethod
    def _get_openai_api_key() -> str:
        if "OPENAI_API_KEY" in os.environ:
            logger.info("Got OPENAI_API_KEY from environment.")
            return os.environ.get("OPENAI_API_KEY", "")
        if os.path.isfile("env.json"):
            with open("env.json", encoding="utf-8") as f:
                env = json.load(f)
                logger.info("Got OPENAI_API_KEY from env.json file.")
                return str(env["OPENAI_API_KEY"]).strip()
        logger.error("Could not get a token for OpenAI!")
        raise RuntimeError("Could not get a token for OpenAI!")
        return "err"

    @staticmethod
    def _get_system_instruction() -> str:
        # slightly adopted prompt from the MetricX 25 paper
        return """
            You are an annotator for the quality of machine translation. Your task is to
            identify errors and assess the quality of the translation.
            Based on the source segment, human-generated reference translation, and machine
            translation surrounded with triple backticks, identify error types in the
            translation and classify them. The categories of errors are: accuracy
            (addition, mistranslation, omission, untranslated text), fluency (character
            encoding, grammar, inconsistency, punctuation, register, spelling), style
            (awkward), terminology (inappropriate for context, inconsistent use),
            non-translation, other, or no-error.
            Each error is classified as one of three severities: critical, major, and minor.
            Critical errors inhibit comprehension of the text. Major errors disrupt the
            flow, but what the text is trying to say is still understandable. Minor errors
            are technically errors, but do not disrupt the flow or hinder comprehension.
            Give a quality estimation as a value between 0 and 1 where 0 is complete gibberish
            and 1 would be a perfect translation.
            Make sure your response is a strict and valid json object that could be parsed with
            json.loads() in python.
            """

    @staticmethod
    def _get_user_instruction(
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
    ) -> str:
        # slightly adopted prompt from the MetricX 25 paper
        return (
            f"{src_lang} source: ```{src}```\n{mt_lang} machine translation: ```{mt}```"
        )

    @staticmethod
    def _get_openai_response(
        client: Optional[OpenAI],
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
        retry: int = 0,
    ) -> JudgeModelResult | None:
        if client is None:
            return None
        system_instruction: str = _OpenAIModel._get_system_instruction()
        user_instruction: str = _OpenAIModel._get_user_instruction(
            src=src, mt=mt, src_lang=src_lang, mt_lang=mt_lang
        )
        prompt: str = f"{system_instruction}\n{user_instruction}"
        response = None
        try:
            # https://developers.openai.com/api/docs/guides/structured-outputs/
            response = client.responses.parse(
                model=OPENAI_MODEL,
                input=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_instruction},
                ],
                # # https://developers.openai.com/api/docs/guides/reasoning?api-mode=responses
                reasoning={"effort": OPENAI_THINKING_LEVEL},
                text_format=QualityEstimation,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        except Exception as e:
            logger.warning(
                f"Failed getting response at retry {retry} from OpenAI API due to: {e}"
            )
            if retry < MAX_RETRY:
                time.sleep(RETRY_WAIT_TIME)
                return _OpenAIModel._get_openai_response(
                    client,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting response at retry {retry} > MAX_RETRY from OpenAI API due to: {e}"
            )
            return JudgeModelResult(
                model=OPENAI_MODEL,
                prompt=prompt,
                status="error",
                parameters={"effort": OPENAI_THINKING_LEVEL},
                response=None,
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        if response is None:
            if retry < MAX_RETRY:
                return _OpenAIModel._get_openai_response(
                    client,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY."
            )
            return JudgeModelResult(
                model=OPENAI_MODEL,
                prompt=prompt,
                status="error",
                parameters={"effort": OPENAI_THINKING_LEVEL},
                response=None,
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        if response.output_parsed is None:
            if retry < MAX_RETRY:
                return _OpenAIModel._get_openai_response(
                    client,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY."
            )
            return JudgeModelResult(
                model=OPENAI_MODEL,
                prompt=prompt,
                status="error",
                parameters={"effort": OPENAI_THINKING_LEVEL},
                response=str(response),
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        try:
            r = response.output_parsed.model_dump(mode="json")
            logger.info(
                f"Successfully got a valid response after retry {retry} for one query."
            )
            return JudgeModelResult(
                model=OPENAI_MODEL,
                prompt=prompt,
                status="ok",
                parameters={"effort": OPENAI_THINKING_LEVEL},
                response=str(response),
                quality_estimation=response.output_parsed,
                quality_estimation_dict=r,
            )
        except Exception as _e:
            if retry < MAX_RETRY:
                return _OpenAIModel._get_openai_response(
                    client,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY."
            )
            return JudgeModelResult(
                model=OPENAI_MODEL,
                prompt=prompt,
                status="error",
                parameters={"effort": OPENAI_THINKING_LEVEL},
                response=str(response),
                quality_estimation=None,
                quality_estimation_dict=None,
            )
        logger.error(f"Failed getting a valid response at retry {retry} > MAX_RETRY.")
        return JudgeModelResult(
            model=OPENAI_MODEL,
            prompt=prompt,
            status="error",
            parameters={"effort": OPENAI_THINKING_LEVEL},
            response=str(response) if response is not None else None,
            quality_estimation=None,
            quality_estimation_dict=None,
        )


class _AnthropicModel:
    @staticmethod
    def _get_anthropic_api_key() -> str:
        if "ANTHROPIC_API_KEY" in os.environ:
            logger.info("Got ANTHROPIC_API_KEY from environment.")
            return os.environ.get("ANTHROPIC_API_KEY", "")
        if os.path.isfile("env.json"):
            with open("env.json", encoding="utf-8") as f:
                env = json.load(f)
                logger.info("Got ANTHROPIC_API_KEY from env.json file.")
                return str(env["ANTHROPIC_API_KEY"]).strip()
        logger.error("Could not get a token for Anthropic!")
        raise RuntimeError("Could not get a token for Anthropic!")
        return "err"

    @staticmethod
    def _get_system_instruction() -> str:
        # slightly adopted prompt from the MetricX 25 paper
        return """
            You are an annotator for the quality of machine translation. Your task is to
            identify errors and assess the quality of the translation.
            Based on the source segment, human-generated reference translation, and machine
            translation surrounded with triple backticks, identify error types in the
            translation and classify them. The categories of errors are: accuracy
            (addition, mistranslation, omission, untranslated text), fluency (character
            encoding, grammar, inconsistency, punctuation, register, spelling), style
            (awkward), terminology (inappropriate for context, inconsistent use),
            non-translation, other, or no-error.
            Each error is classified as one of three severities: critical, major, and minor.
            Critical errors inhibit comprehension of the text. Major errors disrupt the
            flow, but what the text is trying to say is still understandable. Minor errors
            are technically errors, but do not disrupt the flow or hinder comprehension.
            Give a quality estimation as a value between 0 and 1 where 0 is complete gibberish
            and 1 would be a perfect translation.
            Make sure your response is a strict and valid json object that could be parsed with
            json.loads() in python.
            """

    @staticmethod
    def _get_user_instruction(
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
    ) -> str:
        # slightly adopted prompt from the MetricX 25 paper
        return (
            f"{src_lang} source: ```{src}```\n{mt_lang} machine translation: ```{mt}```"
        )

    @staticmethod
    def _get_anthropic_response(
        client: Optional[Anthropic],
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
        retry: int = 0,
    ) -> JudgeModelResult | None:
        if client is None:
            return None
        system_instruction: str = _AnthropicModel._get_system_instruction()
        user_instruction: str = _AnthropicModel._get_user_instruction(
            src=src, mt=mt, src_lang=src_lang, mt_lang=mt_lang
        )
        prompt: str = f"{system_instruction}\n{user_instruction}"
        response = None
        try:
            # https://platform.claude.com/docs/en/build-with-claude/structured-outputs#quick-start
            response = client.messages.parse(
                model=ANTHROPIC_MODEL,
                # https://platform.claude.com/docs/en/build-with-claude/working-with-messages#system-role-in-messages
                system=system_instruction,
                messages=[
                    {"role": "user", "content": user_instruction},
                ],
                # https://platform.claude.com/docs/en/build-with-claude/effort
                output_config={"effort": ANTHROPIC_THINKING_LEVEL},
                output_format=QualityEstimation,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
        except Exception as e:
            logger.warning(
                f"Failed getting response at retry {retry} from Anthropic API due to: {e}"
            )
            if retry < MAX_RETRY:
                time.sleep(RETRY_WAIT_TIME)
                return _AnthropicModel._get_anthropic_response(
                    client,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting response at retry {retry} > MAX_RETRY from Anthropic API due to: {e}"
            )
            return JudgeModelResult(
                model=ANTHROPIC_MODEL,
                prompt=prompt,
                status="error",
                parameters={"effort": ANTHROPIC_THINKING_LEVEL},
                response=None,
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        if response is None:
            if retry < MAX_RETRY:
                return _AnthropicModel._get_anthropic_response(
                    client,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY."
            )
            return JudgeModelResult(
                model=ANTHROPIC_MODEL,
                prompt=prompt,
                status="error",
                parameters={"effort": ANTHROPIC_THINKING_LEVEL},
                response=None,
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        if response.parsed_output is None:
            if retry < MAX_RETRY:
                return _AnthropicModel._get_anthropic_response(
                    client,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY."
            )
            return JudgeModelResult(
                model=ANTHROPIC_MODEL,
                prompt=prompt,
                status="error",
                parameters={"effort": ANTHROPIC_THINKING_LEVEL},
                response=str(response),
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        try:
            r = response.parsed_output.model_dump(mode="json")
            logger.info(
                f"Successfully got a valid response after retry {retry} for one query."
            )
            return JudgeModelResult(
                model=ANTHROPIC_MODEL,
                prompt=prompt,
                status="ok",
                parameters={"effort": ANTHROPIC_THINKING_LEVEL},
                response=str(response),
                quality_estimation=response.parsed_output,
                quality_estimation_dict=r,
            )
        except Exception as _e:
            if retry < MAX_RETRY:
                return _AnthropicModel._get_anthropic_response(
                    client,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY."
            )
            return JudgeModelResult(
                model=ANTHROPIC_MODEL,
                prompt=prompt,
                status="error",
                parameters={"effort": ANTHROPIC_THINKING_LEVEL},
                response=str(response),
                quality_estimation=None,
                quality_estimation_dict=None,
            )
        logger.error(f"Failed getting a valid response at retry {retry} > MAX_RETRY.")
        return JudgeModelResult(
            model=ANTHROPIC_MODEL,
            prompt=prompt,
            status="error",
            parameters={"effort": ANTHROPIC_THINKING_LEVEL},
            response=str(response) if response is not None else None,
            quality_estimation=None,
            quality_estimation_dict=None,
        )


class _GoogleModel:
    @staticmethod
    def _get_gemini_api_key() -> str:
        if "GEMINI_API_KEY" in os.environ:
            logger.info("Got GEMINI_API_KEY from environment.")
            return os.environ.get("GEMINI_API_KEY", "")
        if os.path.isfile("env.json"):
            with open("env.json", encoding="utf-8") as f:
                env = json.load(f)
                logger.info("Got GEMINI_API_KEY from env.json file.")
                return str(env["GEMINI_API_KEY"]).strip()
        logger.error("Could not get a token for Google Gemini!")
        raise RuntimeError("Could not get a token for Google Gemini!")
        return "err"

    @staticmethod
    def _generate_prompt(
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
    ) -> str:
        # slightly adopted prompt from the MetricX 25 paper
        base = """
            You are an annotator for the quality of machine translation. Your task is to
            identify errors and assess the quality of the translation.
            Based on the source segment, human-generated reference translation, and machine
            translation surrounded with triple backticks, identify error types in the
            translation and classify them. The categories of errors are: accuracy
            (addition, mistranslation, omission, untranslated text), fluency (character
            encoding, grammar, inconsistency, punctuation, register, spelling), style
            (awkward), terminology (inappropriate for context, inconsistent use),
            non-translation, other, or no-error.
            Each error is classified as one of three severities: critical, major, and minor.
            Critical errors inhibit comprehension of the text. Major errors disrupt the
            flow, but what the text is trying to say is still understandable. Minor errors
            are technically errors, but do not disrupt the flow or hinder comprehension.
            Give a quality estimation as a value between 0 and 1 where 0 is complete gibberish
            and 1 would be a perfect translation.
            Make sure your response is a strict and valid json object that could be parsed with
            json.loads() in python.
            """
        return f"{base}{src_lang} source: ```{src}```\n{mt_lang} machine translation: ```{mt}```"

    @staticmethod
    def _get_gemini_response(
        client: Optional[Google],
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
        retry: int = 0,
    ) -> JudgeModelResult | None:
        if client is None:
            return None
        prompt: str = _GoogleModel._generate_prompt(
            src=src, mt=mt, src_lang=src_lang, mt_lang=mt_lang
        )
        response = None
        try:
            response = client.models.generate_content(
                model=GOOGLE_MODEL,
                contents=prompt,
                config=GoogleTypes.GenerateContentConfig(
                    # https://ai.google.dev/gemini-api/docs/gemini-3?hl=de#thinking_level
                    # https://ai.google.dev/gemini-api/docs/thinking#thinking-levels
                    thinking_config=GoogleTypes.ThinkingConfig(
                        thinking_level=GOOGLE_THINKING_LEVEL
                    ),
                    # might be worth checking out: https://ai.google.dev/gemini-api/docs/gemini-3?hl=de#structured_outputs_with_tools
                    response_mime_type="application/json",
                    response_json_schema=QualityEstimation.model_json_schema(),
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    seed=SEEDS[retry],
                ),
            )
        except Exception as e:
            logger.warning(
                f"Failed getting response at retry {retry} from Gemini API due to: {e}"
            )
            if retry < MAX_RETRY:
                time.sleep(RETRY_WAIT_TIME)
                return _GoogleModel._get_gemini_response(
                    client,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY."
            )
            return JudgeModelResult(
                model=GOOGLE_MODEL,
                prompt=prompt,
                status="error",
                parameters={"thinking_config": GOOGLE_THINKING_LEVEL},
                response=None,
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        if response is None:
            if retry < MAX_RETRY:
                time.sleep(RETRY_WAIT_TIME)
                return _GoogleModel._get_gemini_response(
                    client,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY."
            )
            return JudgeModelResult(
                model=GOOGLE_MODEL,
                prompt=prompt,
                status="error",
                parameters={"thinking_config": GOOGLE_THINKING_LEVEL},
                response=None,
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        if response.text is None:
            if retry < MAX_RETRY:
                time.sleep(RETRY_WAIT_TIME)
                return _GoogleModel._get_gemini_response(
                    client,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY."
            )
            return JudgeModelResult(
                model=GOOGLE_MODEL,
                prompt=prompt,
                status="error",
                parameters={"thinking_config": GOOGLE_THINKING_LEVEL},
                response=str(response),
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        try:
            qe = QualityEstimation.model_validate_json(response.text)
            r = json.loads(response.text)
            logger.info(
                f"Successfully got a valid response after retry {retry} for one query."
            )
            return JudgeModelResult(
                model=GOOGLE_MODEL,
                prompt=prompt,
                status="ok",
                parameters={"thinking_config": GOOGLE_THINKING_LEVEL},
                response=str(response),
                quality_estimation=qe,
                quality_estimation_dict=r,
            )
        except Exception as _e:
            try:
                if retry < MAX_RETRY:
                    return _GoogleModel._get_gemini_response(
                        client,
                        src=src,
                        mt=mt,
                        src_lang=src_lang,
                        mt_lang=mt_lang,
                        retry=retry + 1,
                    )
                r = json.loads(response.text)
                logger.error(
                    f"Failed getting a valid response at retry {retry} > MAX_RETRY."
                )
                return JudgeModelResult(
                    model=GOOGLE_MODEL,
                    prompt=prompt,
                    status="error",
                    parameters={"thinking_config": GOOGLE_THINKING_LEVEL},
                    response=str(response),
                    quality_estimation=None,
                    quality_estimation_dict=r,
                )
            except Exception as _e:
                if retry < MAX_RETRY:
                    return _GoogleModel._get_gemini_response(
                        client,
                        src=src,
                        mt=mt,
                        src_lang=src_lang,
                        mt_lang=mt_lang,
                        retry=retry + 1,
                    )
                logger.error(
                    f"Failed getting a valid response at retry {retry} > MAX_RETRY."
                )
                return JudgeModelResult(
                    model=GOOGLE_MODEL,
                    prompt=prompt,
                    status="error",
                    parameters={"thinking_config": GOOGLE_THINKING_LEVEL},
                    response=str(response),
                    quality_estimation=None,
                    quality_estimation_dict=None,
                )
        logger.error(f"Failed getting a valid response at retry {retry} > MAX_RETRY.")
        return JudgeModelResult(
            model=GOOGLE_MODEL,
            prompt=prompt,
            status="error",
            parameters={"thinking_config": GOOGLE_THINKING_LEVEL},
            response=None,
            quality_estimation=None,
            quality_estimation_dict=None,
        )


class _OllamaModel:
    @staticmethod
    def _generate_prompt(
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
    ) -> str:
        # slightly adopted prompt from the MetricX 25 paper
        base = """
            You are an annotator for the quality of machine translation. Your task is to
            identify errors and assess the quality of the translation.
            Based on the source segment, human-generated reference translation, and machine
            translation surrounded with triple backticks, identify error types in the
            translation and classify them. The categories of errors are: accuracy
            (addition, mistranslation, omission, untranslated text), fluency (character
            encoding, grammar, inconsistency, punctuation, register, spelling), style
            (awkward), terminology (inappropriate for context, inconsistent use),
            non-translation, other, or no-error.
            Each error is classified as one of three severities: critical, major, and minor.
            Critical errors inhibit comprehension of the text. Major errors disrupt the
            flow, but what the text is trying to say is still understandable. Minor errors
            are technically errors, but do not disrupt the flow or hinder comprehension.
            Give a quality estimation as a value between 0 and 1 where 0 is complete gibberish
            and 1 would be a perfect translation.
            Make sure your response is a strict and valid json object that could be parsed with
            json.loads() in python.
            """
        return f"{base}{src_lang} source: ```{src}```\n{mt_lang} machine translation: ```{mt}```"

    @staticmethod
    def _get_ollama_response(
        client: Optional[Ollama],
        model: str,
        num_predict: int,
        keep_alive: int | str,
        src: str,
        mt: str,
        src_lang: str,
        mt_lang: str,
        retry: int = 0,
    ) -> JudgeModelResult | None:
        if client is None:
            return None
        response: OllamaGenerateResponse | None = None
        prompt: str = _OllamaModel._generate_prompt(
            src=src, mt=mt, src_lang=src_lang, mt_lang=mt_lang
        )
        try:
            response = client.generate(
                model=model,
                prompt=prompt,
                format=QualityEstimation.model_json_schema(),
                keep_alive=keep_alive,
                options={"num_predict": num_predict, "seed": SEEDS[retry]},
            )
        except OllamaResponseError as e:
            logger.error(f"Error in response: {e.error}")
            # if no model, retrive model and try again
            if e.status_code == 404:
                pull_ollama_model(model)
            if retry < MAX_RETRY:
                return _OllamaModel._get_ollama_response(
                    client,
                    model=model,
                    num_predict=num_predict,
                    keep_alive=keep_alive,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
        except Exception as e:
            logger.error(f"Error in response: {e}")

        if response is None:
            if retry < MAX_RETRY:
                return _OllamaModel._get_ollama_response(
                    client,
                    model=model,
                    num_predict=num_predict,
                    keep_alive=keep_alive,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY."
            )
            return JudgeModelResult(
                model=model,
                prompt=prompt,
                status="error",
                parameters={"num_predict": str(num_predict), "seed": str(SEEDS[retry])},
                response=None,
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        if response.response is None:
            if retry < MAX_RETRY:
                return _OllamaModel._get_ollama_response(
                    client,
                    model=model,
                    num_predict=num_predict,
                    keep_alive=keep_alive,
                    src=src,
                    mt=mt,
                    src_lang=src_lang,
                    mt_lang=mt_lang,
                    retry=retry + 1,
                )
            logger.error(
                f"Failed getting a valid response at retry {retry} > MAX_RETRY."
            )
            return JudgeModelResult(
                model=model,
                prompt=prompt,
                status="error",
                parameters={"num_predict": str(num_predict), "seed": str(SEEDS[retry])},
                response=str(response),
                quality_estimation=None,
                quality_estimation_dict=None,
            )

        try:
            qe = QualityEstimation.model_validate_json(response.response)
            r = json.loads(response.response)
            logger.info(
                "Successfully got a valid response after retry {retry} for one query."
            )
            return JudgeModelResult(
                model=model,
                prompt=prompt,
                status="ok",
                parameters={"num_predict": str(num_predict), "seed": str(SEEDS[retry])},
                response=str(response),
                quality_estimation=qe,
                quality_estimation_dict=r,
            )
        except Exception as _e:
            try:
                if retry < MAX_RETRY:
                    return _OllamaModel._get_ollama_response(
                        client,
                        model=model,
                        num_predict=num_predict,
                        keep_alive=keep_alive,
                        src=src,
                        mt=mt,
                        src_lang=src_lang,
                        mt_lang=mt_lang,
                        retry=retry + 1,
                    )
                r = json.loads(response.response)
                logger.error(
                    f"Failed getting a valid response at retry {retry} > MAX_RETRY."
                )
                return JudgeModelResult(
                    model=model,
                    prompt=prompt,
                    status="error",
                    parameters={
                        "num_predict": str(num_predict),
                        "seed": str(SEEDS[retry]),
                    },
                    response=str(response),
                    quality_estimation=None,
                    quality_estimation_dict=r,
                )
            except Exception as _e:
                if retry < MAX_RETRY:
                    return _OllamaModel._get_ollama_response(
                        client,
                        model=model,
                        num_predict=num_predict,
                        keep_alive=keep_alive,
                        src=src,
                        mt=mt,
                        src_lang=src_lang,
                        mt_lang=mt_lang,
                        retry=retry + 1,
                    )
                logger.error(
                    f"Failed getting a valid response at retry {retry} > MAX_RETRY."
                )
                return JudgeModelResult(
                    model=model,
                    prompt=prompt,
                    status="error",
                    parameters={
                        "num_predict": str(num_predict),
                        "seed": str(SEEDS[retry]),
                    },
                    response=str(response),
                    quality_estimation=None,
                    quality_estimation_dict=None,
                )
        logger.error(f"Failed getting a valid response at retry {retry} > MAX_RETRY.")
        return JudgeModelResult(
            model=model,
            prompt=prompt,
            status="error",
            parameters={"num_predict": str(num_predict), "seed": str(SEEDS[retry])},
            response=None,
            quality_estimation=None,
            quality_estimation_dict=None,
        )


class Judge:
    r"""Judge to rate translation quality.

    Judge to rate translation quality using LLMs from OpenAI, Anthropic, Google, and Ollama.

    Parameters
    ----------
    openai : str, or bool, or None, default = None
        - If a string is given, an ``OPENAI_API_KEY`` is expected.
        - If ``None`` or ``True`` the API key will be attempted to be read from the environment.
        - If ``False`` the OpenAI model will not be used as a judge.
    anthropic : str, or bool, or None, default = None
        - If a string is given, an ``ANTHROPIC_API_KEY`` is expected.
        - If ``None`` or ``True`` the API key will be attempted to be read from the environment.
        - If ``False`` the Anthropic model will not be used as a judge.
    google : str, or bool, or None, default = None
        - If a string is given, an ``GEMINI_API_KEY`` is expected.
        - If ``None`` or ``True`` the API key will be attempted to be read from the environment.
        - If ``False`` the Google model will not be used as a judge.
    ollama : str, or bool, or None, default = None
        - If a string is given, an Ollama model identifier (e.g. ``gemma4:e4b``) is expected.
        - If ``None`` or ``True`` the default Ollama model will be used.
        - If ``False`` the Ollama model will not be used as a judge.

    Examples
    --------
    >>> from qc_bench import Judge
    >>> judge = Judge(openai=False, anthropic=False, google=False, ollama="mistral:7b")
    """

    __openai: OpenAI | None = None
    __anthropic: Anthropic | None = None
    __google: Google | None = None
    __ollama: Ollama | None = None
    ollama_model: str = OLLAMA_DEFAULT_MODEL
    r"""The Ollama model identifier that is being used as a judge."""

    def __init__(
        self,
        openai: Optional[str | bool] = None,
        anthropic: Optional[str | bool] = None,
        google: Optional[str | bool] = None,
        ollama: Optional[str | bool] = None,
    ):
        # openai
        if isinstance(openai, str):
            self.__openai = OpenAI(api_key=str(openai).strip())
        elif openai is None or openai:
            self.__openai = OpenAI(api_key=_OpenAIModel._get_openai_api_key())
        # anthropic
        if isinstance(anthropic, str):
            self.__anthropic = Anthropic(api_key=str(anthropic).strip())
        elif anthropic is None or anthropic:
            self.__anthropic = Anthropic(
                api_key=_AnthropicModel._get_anthropic_api_key()
            )
        # google
        if isinstance(google, str):
            self.__google = Google(api_key=str(google).strip())
        elif google is None or google:
            self.__google = Google(api_key=_GoogleModel._get_gemini_api_key())
        # ollama
        if isinstance(ollama, str):
            self.__ollama = Ollama(host=OLLAMA_HOST, headers={})
            self.ollama_model = str(ollama).strip()
        elif ollama is None or ollama:
            self.__ollama = Ollama(host=OLLAMA_HOST, headers={})
            self.ollama_model = OLLAMA_DEFAULT_MODEL

    def score(self, src: str, mt: str, src_lang: str, mt_lang: str) -> JudgeResult:
        r"""Performs quality estimation using all setup LLMs for one translation.

        Parameters
        ----------
        src : str
            The source text.
        mt : str
            The machine translation.
        src_lang : str
            The language of the source text, e.g. ``"English"``.
        mt_lang : str
            The language of the machine translation, e.g. ``"German"``.

        Returns
        -------
        JudgeResult
            The results of all LLMs in a result container, see ``JudgeResult``.

        Examples
        --------
        >>> from qc_bench import Judge
        >>> judge = Judge(
        ...     openai=False, anthropic=False, google=False, ollama="mistral:7b"
        ... )
        >>> jr = judge.score(
        ...     src="The mitochondria is the powerhouse of the cell.",
        ...     mt="Das Mitochondrium ist das Kraftwerk der Zelle.",
        ...     src_lang="English",
        ...     mt_lang="German",
        ... )
        >>> type(jr)
        <class 'qc_bench._judge.JudgeResult'>
        >>> jr.openai is None
        True
        >>> jr.anthropic is None
        True
        >>> jr.google is None
        True
        >>> jr.ollama is None
        False
        >>> type(jr.ollama)
        <class 'qc_bench._judge.JudgeModelResult'>
        >>> jr.ollama.model
        'mistral:7b'
        >>> jr.ollama.score
        0.95
        """
        return JudgeResult(
            openai=_OpenAIModel._get_openai_response(
                self.__openai, src=src, mt=mt, src_lang=src_lang, mt_lang=mt_lang
            ),
            anthropic=None,
            google=_GoogleModel._get_gemini_response(
                self.__google, src=src, mt=mt, src_lang=src_lang, mt_lang=mt_lang
            ),
            ollama=_OllamaModel._get_ollama_response(
                self.__ollama,
                model=self.ollama_model,
                num_predict=MAX_OUTPUT_TOKENS,
                keep_alive=OLLAMA_KEEP_ALIVE,
                src=src,
                mt=mt,
                src_lang=src_lang,
                mt_lang=mt_lang,
            ),
        )
