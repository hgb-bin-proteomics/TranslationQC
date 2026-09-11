#!/usr/bin/env python3

# 2026 (c) Micha Birklbauer
# https://github.com/michabirklbauer/

import os
import time
import json
import logging
from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict, computed_field

from typing import Optional, Annotated, Literal

from ._translation import QualityEstimation
from ._constants import MAX_RETRY, MAX_OUTPUT_TOKENS, RETRY_WAIT_TIME
from ._constants import OPENAI_MODEL, OPENAI_THINKING_LEVEL

logger = logging.getLogger(__name__)


class JudgeModelResult(BaseModel):
    model: Annotated[str, Field(frozen=True, description="The name of the LLM.")]
    prompt: Annotated[
        str, Field(frozen=True, description="The prompt used with the LLM.")
    ]
    status: Annotated[
        Literal["ok", "error"],
        Field(frozen=True, description="Status of the response."),
    ]
    parameters: Annotated[
        Optional[dict[str, str]],
        Field(frozen=True, description="Additional parameters passed to the LLM."),
    ]
    response: Annotated[
        Optional[str], Field(frozen=True, description="The LLM response as raw text.")
    ]
    quality_estimation: Annotated[
        Optional[QualityEstimation],
        Field(frozen=True, description="The quality estimation returned by the LLM."),
    ]
    model_config = ConfigDict(
        validate_assignment=True, strict=True, str_strip_whitespace=True
    )

    @computed_field(description="Quality estimation score returned by the LLM.")
    @property
    def score(self) -> float:
        if self.quality_estimation is not None:
            return self.quality_estimation.quality_estimation_value
        return float("nan")


class JudgeResult(BaseModel):
    openai: Annotated[
        Optional[JudgeModelResult],
        Field(frozen=True, description="Result of the OpenAI model."),
    ]
    anthropic: Annotated[
        Optional[JudgeModelResult],
        Field(frozen=True, description="Result of the Anthropic model."),
    ]
    google: Annotated[
        Optional[JudgeModelResult],
        Field(frozen=True, description="Result of the Google model."),
    ]
    ollama: Annotated[
        Optional[JudgeModelResult],
        Field(frozen=True, description="Result of the Ollama model."),
    ]


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
        try:
            # https://developers.openai.com/api/docs/guides/structured-outputs/
            response = client.responses.parse(
                model=OPENAI_MODEL,
                input=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_instruction},
                ],
                reasoning={"effort": OPENAI_THINKING_LEVEL},
                text_format=QualityEstimation,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        except Exception as e:
            logger.error(
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
            return JudgeModelResult(
                model=OPENAI_MODEL,
                prompt=prompt,
                status="error",
                parameters={"effort": OPENAI_THINKING_LEVEL},
                response=None,
                quality_estimation=None,
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
            return JudgeModelResult(
                model=OPENAI_MODEL,
                prompt=prompt,
                status="error",
                parameters={"effort": OPENAI_THINKING_LEVEL},
                response=str(response.text),
                quality_estimation=None,
            )

        try:
            _r = response.output_parsed.model_dump(mode="json")
            return JudgeModelResult(
                model=OPENAI_MODEL,
                prompt=prompt,
                status="ok",
                parameters={"effort": OPENAI_THINKING_LEVEL},
                response=str(response.text),
                quality_estimation=response.output_parsed,
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
            return JudgeModelResult(
                model=OPENAI_MODEL,
                prompt=prompt,
                status="error",
                parameters={"effort": OPENAI_THINKING_LEVEL},
                response=str(response.text),
                quality_estimation=None,
            )
        return JudgeModelResult(
            model=OPENAI_MODEL,
            prompt=prompt,
            status="error",
            parameters={"effort": OPENAI_THINKING_LEVEL},
            response=None,
            quality_estimation=None,
        )


class _AnthropicModel:
    pass


class _GoogleModel:
    pass


class _OllamaModel:
    pass


class Judge:
    openai: OpenAI | None = None
    anthropic: None = None
    google: None = None
    ollama: None = None
    ollama_model: str | None = None

    def __init__(
        self,
        openai: Optional[str | bool] = None,
        anthropic: Optional[str | bool] = None,
        google: Optional[str | bool] = None,
        ollama: Optional[str | bool] = None,
    ):
        if isinstance(openai, str):
            self.openai = OpenAI(api_key=openai)
        elif openai is None or openai:
            self.openai = OpenAI(api_key=_OpenAIModel._get_openai_api_key())

    def score(self, src: str, mt: str, src_lang: str, mt_lang: str) -> JudgeResult:
        return JudgeResult(
            openai=_OpenAIModel._get_openai_response(
                self.openai, src=src, mt=mt, src_lang=src_lang, mt_lang=mt_lang
            ),
            anthropic=None,
            google=None,
            ollama=None,
        )
