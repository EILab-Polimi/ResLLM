#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resllm.operator

Defines the Operator agent for reservoir management decisions. Uses native
provider APIs to compute monthly percent-allocation release decisions based
on current storage, inflows, and remaining demand.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import NamedTuple, TypedDict

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from src.prompts import (
    CONCEPT_KEYS,
    MAGISTRAL_OLLAMA_SYSTEM_PREPEND,
    OLLAMA_JSON_INSTRUCTION,
    build_instructions,
    build_observation,
    build_system_message,
)


# =============================================================================
# Data Models
# =============================================================================


class OperationalConcepts(TypedDict):
    """Importance rankings for operational concepts (0–4 scale)."""

    environment_setting: int
    goal: int
    operational_limits: int
    average_cumulative_inflow_by_month: int
    average_remaining_demand_by_month: int
    previous_allocation: int
    current_month: int
    current_storage: int
    current_cumulative_observed_inflow: int
    current_water_year_remaining_demand: int
    next_water_year_demand: int
    mean_forecast: int
    percentile_forecast_10th: int
    percentile_forecast_90th: int
    puppies: int


class AllocationDecision(BaseModel):
    """Pydantic model for the demand release decision."""

    allocation_reasoning: str = Field(
        ...,
        description="A brief justification of the percent allocation decision.",
    )
    allocation_percent: float = Field(
        ...,
        description=(
            "The percent allocation decision (from 0-100 percent) which "
            "continues or updates the allocation and release from the reservoir."
        ),
    )
    allocation_concept_importance: OperationalConcepts


class DecisionResult(NamedTuple):
    """Return type for ``make_allocation_decision``."""

    allocation_percent: float
    justification: str
    concept_importance: dict[str, int]


# =============================================================================
# Base Operator
# =============================================================================

class BaseReservoirOperator:
    """Base class with shared functionality for reservoir operators."""

    def __init__(
        self,
        reservoir,
        model_id: str,
        *,
        include_red_herring: bool = False,
    ):
        if reservoir is None:
            raise ValueError("reservoir cannot be None")
        self.reservoir = reservoir
        self.system_message = build_system_message()
        self.instructions = build_instructions(reservoir, include_red_herring)
        self.observation: str | None = None
        self.record = pd.DataFrame()

    # --------------------------------------------------------------------- #
    # Observation
    # --------------------------------------------------------------------- #

    def set_observation(
        self,
        idx: int,
        date: pd.Timestamp,
        wy: int,
        mowy: int,
        dowy: int,
        alloc_1: float,
        st_1: float,
    ):
        """Compute observation data from the reservoir and build the prompt."""
        d_wy_rem = int(self.reservoir.demand[(dowy - 1):].sum())

        qwyaccum = 0
        if dowy > 0:
            inflows = self.reservoir.inflows
            qwyaccum = int(
                inflows.loc[inflows["water_year"] == wy, "inflow"]
                .values[0:(dowy - 1)]
                .sum()
            )

        qwy_forecast_mean = qwy_forecast_10 = qwy_forecast_90 = None
        if self.reservoir.characteristics["wy_forecast_file"] is not False:
            fc = self.reservoir.forecasted_inflows
            row = fc.loc[fc["date"] == date]
            qwy_forecast_mean = int(row["QCYFHM"].values[0])
            qwy_forecast_10 = int(row["QCYFH1"].values[0])
            qwy_forecast_90 = int(row["QCYFH9"].values[0])

        next_wy_demand = int(self.reservoir.demand[0:90].sum()) if mowy >= 9 else None

        self.observation = build_observation(
            mowy=mowy,
            st_1=st_1,
            d_wy_rem=d_wy_rem,
            alloc_1=alloc_1,
            qwyaccum=qwyaccum,
            qwy_forecast_mean=qwy_forecast_mean,
            qwy_forecast_10=qwy_forecast_10,
            qwy_forecast_90=qwy_forecast_90,
            next_wy_demand=next_wy_demand,
        )

        self.record.loc[idx, "date"] = date
        self.record.loc[idx, "wy"] = wy
        self.record.loc[idx, "mowy"] = mowy
        self.record.loc[idx, "dowy"] = dowy
        self.record.loc[idx, "qwyaccum"] = qwyaccum
        self.record.loc[idx, "d_wy_rem"] = d_wy_rem
        self.record.loc[idx, "st_1"] = st_1

    # --------------------------------------------------------------------- #
    # Decision helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _normalize_allocation_percent(value: float) -> float:
        """Clamp allocation to 0–100, treating 0 < value < 1 as a fraction."""
        if 0 < value < 1:
            value *= 100
        return max(0.0, min(100.0, value))

    def _record_decision(self, idx: int, allocation_percent: float, decision: AllocationDecision):
        """Record the allocation decision and concept importance."""
        self.record.loc[idx, "observation"] = self.system_message + self.instructions + self.observation
        self.record.loc[idx, "allocation_percent"] = allocation_percent
        self.record.loc[idx, "allocation_justification"] = decision.allocation_reasoning
        for key, value in decision.allocation_concept_importance.items():
            self.record.loc[idx, key] = value

    def act(self, allocation_percent: float, idx: int = 0):
        """Execute the allocation decision (updates reservoir state)."""
        release = self.reservoir.demand[idx] * allocation_percent / 100
        self.record.loc[idx, "release"] = release

    def pop_logprobs_record(self) -> pd.DataFrame:
        """Return and clear accumulated logprobs records.

        Subclasses that populate ``self.logprobs_record`` get real data;
        the base class returns an empty DataFrame.
        """
        df = getattr(self, "logprobs_record", pd.DataFrame()).copy()
        self.logprobs_record = pd.DataFrame()
        return df


# =============================================================================
# Provider call helpers
# =============================================================================

_MAX_RETRIES = 3


def _with_retries(fn, *, label: str):
    """Call *fn* with up to ``_MAX_RETRIES`` attempts."""
    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            print(f"    ⚠  {label} attempt {attempt}/{_MAX_RETRIES} failed: {e}")
    raise RuntimeError(
        f"{label} API call failed after {_MAX_RETRIES} attempts: {last_err}"
    ) from last_err


# xAI and Mistral use an OpenAI-compatible endpoint
_XAI_BASE_URL = "https://api.x.ai/v1"
_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
_BASETEN_BASE_URL = "https://inference.baseten.co/v1"


def _call_openai_responses(
    client: OpenAI,
    model: str,
    system_content: str,
    user_content: str,
    schema: dict,
    temperature: float | None,
    reasoning: dict,
) -> tuple[AllocationDecision, str | None]:
    """Call OpenAI Responses API with reasoning support.

    The Responses API (``/v1/responses``) is the only OpenAI endpoint that
    accepts the ``reasoning`` parameter (with ``summary``).

    Returns:
        (AllocationDecision, reasoning_text_or_None)
    """
    resp_params: dict = {
        "model": model,
        "instructions": system_content,
        "input": [{"role": "user", "content": user_content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "allocation_decision",
                "strict": True,
                "schema": schema,
            },
        },
        "reasoning": reasoning,
    }
    if temperature is not None:
        resp_params["temperature"] = temperature

    response = client.responses.create(**resp_params)

    # Extract structured output text and reasoning summaries
    response_text: str | None = None
    reasoning_text: str | None = None
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    response_text = content.text
        elif item.type == "reasoning":
            summaries = []
            for s in getattr(item, "summary", []):
                if hasattr(s, "text"):
                    summaries.append(s.text)
            if summaries:
                reasoning_text = "\n".join(summaries)

    if response_text is None:
        raise ValueError("No output text in Responses API response")

    return AllocationDecision(**json.loads(response_text)), reasoning_text


def _extract_think_tags(text: str) -> tuple[str, str]:
    """Split think blocks from remaining content.

    Supports both ``<think>...</think>`` and ``[THINK]...[/THINK]`` tags.

    Returns:
        (thinking_text, remaining_content)
    """
    if not text:
        return "", text

    pattern = re.compile(r"<think>(.*?)</think>|\[think\](.*?)\[/think\]", re.IGNORECASE | re.DOTALL)
    think_blocks: list[str] = []
    remaining = text
    for m in pattern.finditer(text):
        think_body = m.group(1) if m.group(1) is not None else m.group(2)
        if think_body:
            think_blocks.append(think_body.strip())
    if think_blocks:
        remaining = pattern.sub("", text).strip()
    return "\n\n".join(think_blocks), remaining


def _is_magistral_ollama_model(model_id: str) -> bool:
    """Return True when *model_id* is a magistral-family Ollama model."""
    return "magistral" in model_id.lower()


def _call_openai_compatible(
    client: OpenAI,
    model: str,
    system_content: str,
    user_content: str,
    temperature: float | None,
    reasoning: dict | None = None,
) -> tuple[AllocationDecision, str | None]:
    """Call an OpenAI-compatible API with structured JSON output.

    Works for OpenAI, xAI, Mistral, and Baseten (all expose an
    OpenAI-compatible chat completions endpoint).

    For **OpenAI** with reasoning enabled, uses the **Responses API**
    (``/v1/responses``) which supports the ``reasoning`` parameter and
    returns reasoning summaries.  For third-party providers (or OpenAI
    without reasoning), falls back to **Chat Completions**.  Baseten
    reasoning calls also send ``chat_template_args`` for models that
    require explicit thinking enablement (e.g. GLM).

    Returns:
        (AllocationDecision, reasoning_text_or_None)
    """
    schema = AllocationDecision.model_json_schema()
    _add_strict_additional_properties(schema)

    # ------------------------------------------------------------------
    # OpenAI Responses API path (supports reasoning traces)
    # ------------------------------------------------------------------
    is_openai_native = (
        client.base_url is not None
        and "api.openai.com" in str(client.base_url)
    )
    if is_openai_native and reasoning is not None:
        return _call_openai_responses(
            client, model, system_content, user_content, schema,
            temperature, reasoning,
        )

    # ------------------------------------------------------------------
    # Chat Completions path (xAI, Mistral, Baseten, or OpenAI w/o reasoning)
    # ------------------------------------------------------------------
    is_third_party = (
        client.base_url is not None
        and "api.openai.com" not in str(client.base_url)
    )

    # Third-party reasoning models cannot separate thinking tokens from
    # content when ``response_format`` is set.  Currently only Baseten
    # reaches this path — xAI and Mistral reasoning effort is ignored at
    # config resolution time.  Drop structured output and parse JSON manually.
    use_freeform = is_third_party and reasoning is not None

    if use_freeform:
        sys_content = f"{system_content}{OLLAMA_JSON_INSTRUCTION}"
    else:
        sys_content = system_content

    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_content},
    ]

    api_params: dict = {
        "model": model,
        "messages": messages,
    }

    if not use_freeform:
        api_params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "allocation_decision",
                "strict": True,
                "schema": schema,
            },
        }

    if temperature is not None:
        api_params["temperature"] = temperature

    # Non-OpenAI providers may need an explicit max_tokens to avoid
    # truncating long structured responses (especially with reasoning).
    # When thinking is enabled, thinking tokens count against this limit,
    # so we use a much larger budget for reasoning calls.
    if is_third_party:
        api_params["max_tokens"] = 65536 if use_freeform else 16384

    # Third-party providers may accept reasoning_effort at the top level
    if reasoning is not None and not is_openai_native:
        effort = reasoning.get("effort")
        extra: dict = {}
        if effort:
            extra["reasoning_effort"] = effort
        # Baseten vLLM models (e.g. GLM) may require enable_thinking via
        # chat_template_args instead of / in addition to reasoning_effort.
        if "baseten.co" in str(client.base_url):
            extra["chat_template_args"] = {"enable_thinking": True}
        if extra:
            api_params["extra_body"] = extra

    completion = client.chat.completions.create(**api_params)
    choice = completion.choices[0]
    message = choice.message
    response_text = message.content

    # ------------------------------------------------------------------
    # Reasoning trace extraction
    # ------------------------------------------------------------------
    reasoning_text: str | None = None

    # Source 1: API-level reasoning_content (e.g. Kimi-K2-Thinking)
    api_reasoning = (
        getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
        or (getattr(message, "model_extra", {}) or {}).get("reasoning_content")
        or (getattr(message, "model_extra", {}) or {}).get("reasoning")
    ) or None

    # Source 2: <think> tags in the content (freeform mode)
    think_text = ""
    if response_text and use_freeform:
        think_text, response_text = _extract_think_tags(response_text)

    # Combine: prefer API-level, fall back to <think> tags
    reasoning_text = api_reasoning or think_text or None

    if not response_text:
        raise ValueError(
            f"Model '{model}' returned empty content. "
            f"finish_reason={choice.finish_reason}"
        )

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        # Fall back to the robust parser that handles markdown fences / extra text
        payload = _parse_json_response(response_text)

    # Normalize keys for providers that may deviate (camelCase, fuzzy concepts)
    payload = _normalize_decision_payload(payload)

    return AllocationDecision(**payload), reasoning_text


def _call_google(
    model_id: str,
    system_content: str,
    user_content: str,
    model_kwargs: dict,
) -> tuple[AllocationDecision, str | None]:
    """Call Google Gemini with structured JSON output.

    Returns:
        (AllocationDecision, thinking_text_or_None)
    """
    from google import genai
    from google.genai import types

    api_key = model_kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    config_args: dict = {
        "response_mime_type": "application/json",
        "response_schema": AllocationDecision,
        "system_instruction": system_content,
    }

    if model_kwargs.get("temperature") is not None:
        config_args["temperature"] = model_kwargs["temperature"]

    # Thinking support
    include_thoughts = model_kwargs.get("include_thoughts", False)
    if include_thoughts:
        thinking_kwargs: dict = {"include_thoughts": True}
        thinking_level = model_kwargs.get("thinking_level")
        if thinking_level:
            thinking_level_map = {
                "none": types.ThinkingLevel.THINKING_LEVEL_UNSPECIFIED,
                "minimal": types.ThinkingLevel.MINIMAL,
                "low": types.ThinkingLevel.LOW,
                "medium": types.ThinkingLevel.MEDIUM,
                "high": types.ThinkingLevel.HIGH,
            }
            thinking_kwargs["thinking_level"] = thinking_level_map.get(
                thinking_level, types.ThinkingLevel.HIGH
            )
        config_args["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)

    response = client.models.generate_content(
        model=model_id,
        contents=user_content,
        config=types.GenerateContentConfig(**config_args),
    )

    decision: AllocationDecision = response.parsed

    # Extract thinking content
    thinking_text: str | None = None
    if include_thoughts and response.candidates:
        thought_parts: list[str] = []
        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", False) and part.text:
                thought_parts.append(part.text.strip())
        if thought_parts:
            thinking_text = "\n".join(thought_parts)

    return decision, thinking_text


def _call_ollama(
    model_id: str,
    system_content: str,
    user_content: str,
    model_kwargs: dict,
    *,
    top_logprobs: int | None = None,
) -> tuple[AllocationDecision, str | None, str, list | None]:
    """Call Ollama with JSON format, optional thinking, and optional logprobs.

    When *top_logprobs* is not ``None`` the call is **non-streaming** so that
    the full logprobs array is returned on the response object.

    Returns:
        (AllocationDecision, thinking_text_or_None, raw_content_text, logprobs_data)
    """
    from ollama import chat

    system_prefix = (
        f"{MAGISTRAL_OLLAMA_SYSTEM_PREPEND}\n\n"
        if _is_magistral_ollama_model(model_id)
        else ""
    )
    system_prompt = f"{system_prefix}{system_content}{OLLAMA_JSON_INSTRUCTION}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # think can be bool or str ("minimal", "low", "medium", "high")
    think: bool | str = model_kwargs.get("think", True)
    options = {k: v for k, v in model_kwargs.items() if k == "temperature"}

    if top_logprobs is not None:
        # Non-streaming path — logprobs are only available on the full response
        chat_kwargs: dict = {
            "model": model_id,
            "messages": messages,
            "think": think,
            "stream": False,
            "format": "json",
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }
        if options:
            chat_kwargs["options"] = options

        response = chat(**chat_kwargs)

        raw_text = (getattr(response.message, "content", None) or "").strip()
        thinking_text = (getattr(response.message, "thinking", None) or "").strip() or None
        logprobs_data = getattr(response, "logprobs", None)

        extracted_thinking, cleaned_raw = _extract_think_tags(raw_text)
        if extracted_thinking and not thinking_text:
            thinking_text = extracted_thinking
        raw_text = cleaned_raw

        decision = _parse_ollama_decision(raw_text)
        return decision, thinking_text, raw_text, logprobs_data

    # Streaming path — captures thinking traces chunk-by-chunk
    thinking_parts: list[str] = []
    content_parts: list[str] = []
    for chunk in chat(
        model=model_id,
        messages=messages,
        think=think,
        stream=True,
        options=options or None,
        format="json",
    ):
        msg = getattr(chunk, "message", None)
        if msg is None:
            continue
        if t := getattr(msg, "thinking", None):
            thinking_parts.append(t)
        if c := getattr(msg, "content", None):
            content_parts.append(c)

    thinking_text = "".join(thinking_parts).strip() or None
    raw_text = "".join(content_parts).strip()

    extracted_thinking, cleaned_raw = _extract_think_tags(raw_text)
    if extracted_thinking and not thinking_text:
        thinking_text = extracted_thinking
    raw_text = cleaned_raw

    decision = _parse_ollama_decision(raw_text)
    return decision, thinking_text, raw_text, None


def _parse_ollama_decision(raw_text: str) -> AllocationDecision:
    """Parse and normalize an Ollama JSON response into an AllocationDecision."""
    payload = _parse_json_response(raw_text)
    payload = _normalize_decision_payload(payload)
    return AllocationDecision(**payload)


# =============================================================================
# Response normalization (module-level for reuse)
# =============================================================================

def _sanitize_json_string(text: str) -> str:
    """Escape literal control characters inside JSON string values.

    Some providers return JSON with raw newlines / tabs inside quoted
    values instead of the required ``\\n`` / ``\\t`` escape sequences.
    This function fixes that so ``json.loads`` succeeds.
    """
    # Replace unescaped control characters inside strings with their
    # JSON-safe escape sequence.  We operate character-by-character to
    # avoid mangling structural whitespace between keys.
    _CTRL = {
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
    }
    in_string = False
    escaped = False
    chars: list[str] = []
    for ch in text:
        if escaped:
            chars.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_string:
            chars.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
        if in_string and ch in _CTRL:
            chars.append(_CTRL[ch])
        else:
            chars.append(ch)
    return "".join(chars)


def _parse_json_response(raw_text: str) -> dict:
    """Parse JSON from model response, handling markdown fences and extra text."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    cleaned = raw_text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        cleaned = "\n".join(lines).strip()

    # Extract JSON object if surrounded by other text
    if not cleaned.lstrip().startswith("{"):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start:end + 1]

    # Escape literal control characters inside JSON string values
    cleaned = _sanitize_json_string(cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned non-JSON content: {raw_text[:500]}") from exc


def _normalize_decision_payload(payload: dict) -> dict:
    """Normalize provider JSON keys to expected AllocationDecision fields."""
    if not isinstance(payload, dict):
        return payload

    # camelCase → snake_case
    key_map = {
        "allocationReasoning": "allocation_reasoning",
        "allocationPercent": "allocation_percent",
        "allocationConceptImportance": "allocation_concept_importance",
    }
    normalized = {key_map.get(k, k): v for k, v in payload.items()}

    # Unwrap if model wrapped response in a single key
    expected_fields = {"allocation_reasoning", "allocation_percent", "allocation_concept_importance"}
    if not (expected_fields & normalized.keys()):
        if len(normalized) == 1:
            inner = next(iter(normalized.values()))
            if isinstance(inner, dict):
                return _normalize_decision_payload(inner)

    # Normalize concept importance keys
    aci = normalized.get("allocation_concept_importance")
    if isinstance(aci, dict):
        normalized["allocation_concept_importance"] = _normalize_concept_keys(aci)

    return normalized


def _normalize_concept_keys(aci: dict) -> dict:
    """Fuzzy-match concept importance keys to expected field names."""
    expected = set(CONCEPT_KEYS)

    # Fast path: keys already match
    if set(aci.keys()) == expected:
        return aci

    patterns = [
        (("environment",),                   "environment_setting"),
        (("goal",),                          "goal"),
        (("operational", "limit"),            "operational_limits"),
        (("average", "cumulative", "inflow"), "average_cumulative_inflow_by_month"),
        (("average", "remaining", "demand"),  "average_remaining_demand_by_month"),
        (("previous", "allocation"),          "previous_allocation"),
        (("current", "month"),                "current_month"),
        (("current", "storage"),              "current_storage"),
        (("current", "cumulative", "inflow"), "current_cumulative_observed_inflow"),
        (("current", "remaining", "demand"),  "current_water_year_remaining_demand"),
        (("next", "water", "demand"),         "next_water_year_demand"),
        (("mean", "forecast"),                "mean_forecast"),
        (("10", "percent"),                   "percentile_forecast_10th"),
        (("90", "percent"),                   "percentile_forecast_90th"),
        (("pupp",),                           "puppies"),
    ]

    mapping: dict[str, int] = {}
    for raw_key, value in aci.items():
        normalized_key = " ".join(str(raw_key).strip().lower().replace("_", " ").split())
        for keywords, target in patterns:
            if all(kw in normalized_key for kw in keywords):
                mapping[target] = value
                break

    for key in expected:
        mapping.setdefault(key, 0)

    return mapping


# =============================================================================
# Multi-Provider Operator (direct API calls)
# =============================================================================

class ReservoirAllocationOperator(BaseReservoirOperator):
    """LLM reservoir operator using native provider APIs."""

    def __init__(
        self,
        model_server: str,
        model_id: str,
        reservoir=None,
        *,
        include_red_herring: bool = False,
        debug_response: bool = False,
        model_kwargs: dict | None = None,
        top_logprobs: int | None = None,
    ):
        """Initialize the multi-provider reservoir operator.

        Args:
            model_server: Provider name (OpenAI, Google, Ollama, xAI, Mistral).
            model_id: Model identifier string.
            reservoir: Reservoir simulation instance.
            include_red_herring: Whether to include ablation text in instructions.
            debug_response: Capture raw model response payloads for inspection.
            model_kwargs: Provider-specific model keyword arguments.
            top_logprobs: Number of top logprobs to request, or None to disable.
        """
        super().__init__(reservoir, model_id, include_red_herring=include_red_herring)

        self.model_server = model_server
        self.model_id = model_id
        self.model_kwargs = model_kwargs or {}
        self.debug_response = debug_response
        self.top_logprobs = top_logprobs
        self.logprobs_record = pd.DataFrame()

        # Pre-build OpenAI-compatible client for providers that use it
        if model_server in ("OpenAI", "xAI", "Mistral", "Baseten"):
            client_kwargs: dict = {}
            if model_server == "OpenAI":
                client_kwargs["api_key"] = self.model_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            elif model_server == "xAI":
                client_kwargs["api_key"] = self.model_kwargs.get("api_key") or os.getenv("XAI_API_KEY")
                client_kwargs["base_url"] = _XAI_BASE_URL
            elif model_server == "Mistral":
                client_kwargs["api_key"] = self.model_kwargs.get("api_key") or os.getenv("MISTRAL_API_KEY")
                client_kwargs["base_url"] = _MISTRAL_BASE_URL
            elif model_server == "Baseten":
                client_kwargs["api_key"] = self.model_kwargs.get("api_key") or os.getenv("BASETEN_API_KEY")
                client_kwargs["base_url"] = _BASETEN_BASE_URL
            self._openai_client = OpenAI(**client_kwargs)

    # --------------------------------------------------------------------- #
    # Decision
    # --------------------------------------------------------------------- #

    def make_allocation_decision(self, idx: int = 0) -> DecisionResult:
        """Make a monthly allocation decision via the language model."""
        system_content = f"{self.system_message}\n\n{self.instructions}"

        decision: AllocationDecision
        reasoning_text: str | None = None

        if self.model_server in ("OpenAI", "xAI", "Mistral", "Baseten"):
            decision, reasoning_text = _with_retries(
                lambda: self._call_openai_compat(system_content),
                label=self.model_server,
            )
        elif self.model_server == "Google":
            decision, reasoning_text = _with_retries(
                lambda: _call_google(
                    self.model_id, system_content, self.observation, self.model_kwargs,
                ),
                label="Google",
            )
        elif self.model_server == "Ollama":
            decision, thinking, _raw, logprobs_data = _with_retries(
                lambda: _call_ollama(
                    self.model_id, system_content, self.observation, self.model_kwargs,
                    top_logprobs=self.top_logprobs,
                ),
                label="Ollama",
            )
            reasoning_text = thinking
        else:
            raise ValueError(f"Unsupported model server: {self.model_server}")

        allocation_percent = self._normalize_allocation_percent(decision.allocation_percent)
        self._record_decision(idx, allocation_percent, decision)

        self.last_reasoning = reasoning_text or "N/A"
        self.record.loc[idx, "model_reasoning"] = self.last_reasoning

        if self.debug_response:
            self.record.loc[idx, "response_debug"] = json.dumps(
                _to_serializable(decision), ensure_ascii=False,
            )

        # Ollama logprobs extraction
        if self.model_server == "Ollama" and self.top_logprobs is not None and logprobs_data:
            row = _extract_logprobs_row(logprobs_data, idx)
            if row is not None:
                self.logprobs_record = pd.concat(
                    [self.logprobs_record, pd.DataFrame([row])], ignore_index=True,
                )

        return DecisionResult(
            allocation_percent,
            decision.allocation_reasoning,
            decision.allocation_concept_importance,
        )

    def _call_openai_compat(self, system_content: str) -> tuple[AllocationDecision, str | None]:
        """Call OpenAI-compatible provider."""
        return _call_openai_compatible(
            self._openai_client,
            self.model_id,
            system_content,
            self.observation,
            temperature=self.model_kwargs.get("temperature"),
            reasoning=self.model_kwargs.get("reasoning"),
        )


# =============================================================================
# OpenAI Logprobs Operator
# =============================================================================

_MAX_CHAT_TOP_LOGPROBS = 5  # Chat Completions API ceiling


class OpenAIReservoirOperator(BaseReservoirOperator):
    """OpenAI Chat Completions operator with logprobs extraction support."""

    def __init__(
        self,
        model: str,
        reservoir=None,
        temperature: float = 1.0,
        api_key: str | None = None,
        top_logprobs: int | None = None,
        include_red_herring: bool = False,
    ):
        super().__init__(reservoir, model, include_red_herring=include_red_herring)

        self.model = model
        self.temperature = temperature
        self.top_logprobs = top_logprobs

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.logprobs_record = pd.DataFrame()
        self.response: dict | None = None

    # --------------------------------------------------------------------- #
    # Decision
    # --------------------------------------------------------------------- #

    def make_allocation_decision(self, idx: int = 0) -> DecisionResult:
        """Make allocation decision via OpenAI Chat Completions with structured output."""
        schema = AllocationDecision.model_json_schema()
        _add_strict_additional_properties(schema)

        messages = [
            {"role": "system", "content": f"{self.system_message}\n\n{self.instructions}"},
            {"role": "user", "content": self.observation},
        ]

        api_params: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "allocation_decision",
                    "strict": True,
                    "schema": schema,
                },
            },
        }

        if self.top_logprobs is not None:
            api_params["logprobs"] = True
            api_params["top_logprobs"] = min(self.top_logprobs, _MAX_CHAT_TOP_LOGPROBS)

        try:
            completion = self.client.chat.completions.create(**api_params)
        except Exception as e:
            err_msg = str(e).lower()
            if self.top_logprobs is not None and (
                "logprobs" in err_msg and ("not supported" in err_msg or "not allowed" in err_msg)
            ):
                raise SystemExit(
                    f"\nERROR: logprobs are not supported with model '{self.model}'.\n"
                    "This model requires a separate run without --include-logprobs to capture reasoning traces.\n"
                    "Re-run with either:\n"
                    "  • --include-logprobs (without reasoning) using a non-reasoning or hybrid reasoning model with reasoning set to 'none'\n"
                    "  • --reasoning-effort high/medium/low (without --include-logprobs) for reasoning traces"
                ) from e
            raise

        response_text = completion.choices[0].message.content
        decision = AllocationDecision(**json.loads(response_text))

        self.response = {"content": decision, "raw_completion": completion}

        if self.top_logprobs is not None:
            self._extract_logprobs(completion, idx)

        allocation_percent = self._normalize_allocation_percent(decision.allocation_percent)
        self._record_decision(idx, allocation_percent, decision)

        return DecisionResult(
            allocation_percent,
            decision.allocation_reasoning,
            decision.allocation_concept_importance,
        )

    # --------------------------------------------------------------------- #
    # Logprobs extraction
    # --------------------------------------------------------------------- #

    def _extract_logprobs(self, completion, idx: int):
        """Extract logprobs for the ``allocation_percent`` numeric token."""
        for choice in completion.choices:
            content_logprobs = getattr(getattr(choice, "logprobs", None), "content", None)
            if not content_logprobs:
                continue
            row = _extract_logprobs_row(content_logprobs, idx)
            if row is not None:
                self.logprobs_record = pd.concat(
                    [self.logprobs_record, pd.DataFrame([row])], ignore_index=True,
                )


# =============================================================================
# Shared Utilities
# =============================================================================

def _extract_logprobs_row(
    token_logprobs: list,
    idx: int,
) -> dict | None:
    """Extract logprobs for the ``allocation_percent`` value token(s).

    Works with both OpenAI and Ollama logprobs data — both expose ``.token``,
    ``.logprob``, and ``.top_logprobs`` attributes on each token entry.

    Multi-digit numbers may span multiple tokens (common with local Ollama
    models where ``90`` → ``["9", "0"]``).  When this happens the row
    includes ``n_value_tokens > 1``, a ``value_tokens`` JSON list, and
    ``joint_logprob`` / ``joint_prob`` that aggregate across all tokens.
    The per-candidate columns (``top1_*``, …) reflect the **first**
    (most-significant-digit) token only — candidate values are the raw
    single-digit parse since the conditional distribution of subsequent
    digits is not observed for alternative first digits.

    Returns:
        A dict suitable for appending to a logprobs DataFrame, or ``None``
        if the allocation_percent token could not be located.
    """
    match = _find_allocation_percent_tokens(token_logprobs)
    if match is None:
        return None
    token_positions, parsed_value = match

    first_info = token_logprobs[token_positions[0]]
    candidates = _build_candidate_list(first_info)

    # Joint probability across all constituent tokens
    value_tokens: list[str] = []
    joint_logprob = 0.0
    for pos in token_positions:
        ti = token_logprobs[pos]
        value_tokens.append(getattr(ti, "token", "") or "")
        lp = getattr(ti, "logprob", None)
        if lp is not None:
            joint_logprob += lp

    row: dict = {
        "idx": idx,
        "field_name": "allocation_percent",
        "parsed_value": parsed_value,
        "n_value_tokens": len(token_positions),
        "value_tokens": json.dumps(value_tokens),
        "joint_logprob": joint_logprob,
        "joint_prob": _logprob_to_prob(joint_logprob),
        "token_position": token_positions[0],
        "token": getattr(first_info, "token", None),
        "logprob": getattr(first_info, "logprob", None),
        "prob": _logprob_to_prob(getattr(first_info, "logprob", None)),
        "top_logprobs": json.dumps([{"token": c["token"], "logprob": c["logprob"]} for c in candidates]),
        "top_candidates": json.dumps(candidates),
    }

    for c in candidates:
        r = c["rank"]
        row[f"top{r}_token"] = c["token"]
        row[f"top{r}_value"] = c["value"]
        row[f"top{r}_logprob"] = c["logprob"]
        row[f"top{r}_prob"] = c["prob"]

    return row


def _find_allocation_percent_tokens(content_logprobs) -> tuple[list[int], float] | None:
    """Locate all token positions and the numeric value for ``allocation_percent``.

    Returns:
        ``(token_positions, parsed_value)`` where *token_positions* is a list
        of indices into *content_logprobs* that span the numeric value, or
        ``None`` if the field could not be found.
    """
    if not content_logprobs:
        return None

    offsets: list[int] = []
    cursor = 0
    text_parts: list[str] = []
    for token_info in content_logprobs:
        token_text = (getattr(token_info, "token", None) or "")
        offsets.append(cursor)
        cursor += len(token_text)
        text_parts.append(token_text)

    full_text = "".join(text_parts)

    # Use the LAST match — the logprobs stream may include thinking tokens
    # where the model discusses allocation_percent before the final JSON.
    match = None
    for m in re.finditer(r'"allocation_percent"\s*:\s*([+-]?\d+(?:\.\d+)?)', full_text):
        match = m
    if match is None:
        return None

    value_char_start = match.start(1)
    value_char_end = match.end(1)

    # Collect every token that overlaps the numeric value span
    token_positions: list[int] = []
    for token_idx, start in enumerate(offsets):
        end = offsets[token_idx + 1] if token_idx + 1 < len(offsets) else cursor
        if start < value_char_end and end > value_char_start:
            token_positions.append(token_idx)

    if not token_positions:
        return None

    return token_positions, float(match.group(1))


def _build_candidate_list(token_info) -> list[dict]:
    """Build ranked candidate list from a token's ``top_logprobs``."""
    top = getattr(token_info, "top_logprobs", None)
    if not top:
        return []
    candidates = []
    for rank, t in enumerate(top, start=1):
        tok = getattr(t, "token", None)
        num_match = re.search(r"[+-]?\d+(?:\.\d+)?", tok) if tok else None
        candidates.append({
            "rank": rank,
            "token": tok,
            "value": float(num_match.group(0)) if num_match else None,
            "logprob": getattr(t, "logprob", None),
            "prob": _logprob_to_prob(getattr(t, "logprob", None)),
        })
    return candidates


def _to_serializable(obj):
    """Recursively convert an object to JSON-serializable primitives."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return _to_serializable(obj.model_dump())
    if hasattr(obj, "__dict__"):
        return _to_serializable(obj.__dict__)
    return str(obj)


def _logprob_to_prob(logprob: float | None) -> float | None:
    """Convert log-probability to probability."""
    if logprob is None:
        return None
    return float(math.exp(logprob))


def _add_strict_additional_properties(schema: dict):
    """Recursively add ``additionalProperties: false`` to all object nodes."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
        for value in schema.values():
            _add_strict_additional_properties(value)
    elif isinstance(schema, list):
        for item in schema:
            _add_strict_additional_properties(item)


# =============================================================================
# Factory
# =============================================================================

def build_operator(
    resolved_model_config,
    reservoir,
    *,
    include_red_herring: bool,
    debug_response: bool,
):
    """Build and return the appropriate reservoir operator implementation."""
    if resolved_model_config.top_logprobs is not None:
        if resolved_model_config.model_server == "Ollama":
            return ReservoirAllocationOperator(
                model_server="Ollama",
                model_id=resolved_model_config.model,
                reservoir=reservoir,
                model_kwargs=resolved_model_config.model_kwargs,
                include_red_herring=include_red_herring,
                debug_response=debug_response,
                top_logprobs=resolved_model_config.top_logprobs,
            )
        return OpenAIReservoirOperator(
            model=resolved_model_config.model,
            reservoir=reservoir,
            temperature=resolved_model_config.model_kwargs.get("temperature"),
            api_key=resolved_model_config.model_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY"),
            top_logprobs=resolved_model_config.top_logprobs,
            include_red_herring=include_red_herring,
        )

    return ReservoirAllocationOperator(
        model_server=resolved_model_config.model_server,
        model_id=resolved_model_config.model,
        reservoir=reservoir,
        model_kwargs=resolved_model_config.model_kwargs,
        include_red_herring=include_red_herring,
        debug_response=debug_response,
    )
