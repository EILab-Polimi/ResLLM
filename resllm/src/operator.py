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
import os
import re
from typing import TypedDict

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from src.prompts import (
    CONCEPT_KEYS,
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
        objective: str = "minimize-shortages",
    ):
        if reservoir is None:
            raise ValueError("reservoir cannot be None")
        self.reservoir = reservoir
        self.system_message = build_system_message(objective)
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


# =============================================================================
# Provider call helpers
# =============================================================================

_MAX_RETRIES = 10


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


def _call_openai(
    client: OpenAI,
    model: str,
    system_content: str,
    user_content: str,
    temperature: float | None,
    reasoning: dict | None = None,
) -> tuple[AllocationDecision, str | None]:
    """Call OpenAI with structured JSON output.

    With reasoning enabled, uses the **Responses API** (``/v1/responses``),
    the only endpoint that accepts the ``reasoning`` parameter and returns
    reasoning summaries.  Otherwise uses **Chat Completions** with a strict
    ``json_schema`` response format (no reasoning trace available).

    Returns:
        (AllocationDecision, reasoning_text_or_None)
    """
    schema = AllocationDecision.model_json_schema()
    _add_strict_additional_properties(schema)

    # Reasoning models capture traces only via the Responses API.
    if reasoning is not None:
        return _call_openai_responses(
            client, model, system_content, user_content, schema,
            temperature, reasoning,
        )

    # Non-reasoning models: Chat Completions with structured output.
    api_params: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "allocation_decision",
                "strict": True,
                "schema": schema,
            },
        },
    }
    if temperature is not None:
        api_params["temperature"] = temperature

    completion = client.chat.completions.create(**api_params)
    choice = completion.choices[0]
    response_text = choice.message.content

    if not response_text:
        raise ValueError(
            f"Model '{model}' returned empty content. "
            f"finish_reason={choice.finish_reason}"
        )

    # Strict structured output guarantees the schema — parse directly.
    return AllocationDecision(**json.loads(response_text)), None


def _call_ollama(
    model_id: str,
    system_content: str,
    user_content: str,
    model_kwargs: dict,
) -> tuple[AllocationDecision, str | None]:
    """Call Ollama via the generate endpoint with optional thinking.

    Uses ``generate`` rather than ``chat`` because thinking models often
    leave the chat ``content`` field empty.  When thinking is enabled,
    ``format="json"`` is omitted so the model reasons freely; the system
    prompt JSON instruction guides output format instead.

    Returns:
        (AllocationDecision, thinking_text_or_None)
    """
    from ollama import generate

    system_prompt = f"{system_content}{OLLAMA_JSON_INSTRUCTION}"
    think: bool | str = model_kwargs.get("think", True)
    is_thinking = think is not False

    kwargs: dict = {
        "model": model_id,
        "prompt": user_content,
        "system": system_prompt,
        "think": think,
        "stream": True,
    }
    if not is_thinking:
        kwargs["format"] = "json"
    if "temperature" in model_kwargs:
        kwargs["options"] = {"temperature": model_kwargs["temperature"]}

    # Streaming — captures thinking traces chunk-by-chunk
    thinking_parts: list[str] = []
    content_parts: list[str] = []
    for chunk in generate(**kwargs):
        if t := getattr(chunk, "thinking", None):
            thinking_parts.append(t)
        if c := getattr(chunk, "response", None):
            content_parts.append(c)
    raw_text = "".join(content_parts).strip()
    thinking_text = "".join(thinking_parts).strip() or None

    # Some thinking models place everything in the thinking field
    if not raw_text and thinking_text:
        raw_text, thinking_text = _split_thinking_json(thinking_text)

    decision = _parse_ollama_decision(raw_text)
    return decision, thinking_text


def _split_thinking_json(text: str) -> tuple[str, str | None]:
    """Separate a JSON payload from preceding reasoning in a thinking blob.

    Some Ollama thinking models place everything — free-text reasoning
    *and* the final JSON object — in the ``thinking`` field, leaving
    ``response`` empty.  This helper extracts the **last** top-level
    JSON object (``{…}``) as the payload and treats everything before it
    as the reasoning trace.

    Returns:
        (json_text, reasoning_text_or_None)
    """
    # Find the last top-level '{' ... '}' block using brace depth
    depth = 0
    obj_start: int | None = None
    obj_end: int | None = None
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                obj_end = i + 1

    if obj_start is not None and obj_end is not None:
        json_text = text[obj_start:obj_end].strip()
        reasoning = text[:obj_start].strip()
        # Also grab any text after the JSON (rare but possible)
        trailing = text[obj_end:].strip()
        if trailing:
            reasoning = f"{reasoning}\n{trailing}".strip() if reasoning else trailing
        return json_text, reasoning or None

    # No JSON object found — return the whole thing as JSON (will fail at parse)
    return text, None


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
        objective: str = "minimize-shortages",
    ):
        """Initialize the reservoir operator.

        Args:
            model_server: Provider name (OpenAI or Ollama).
            model_id: Model identifier string.
            reservoir: Reservoir simulation instance.
            include_red_herring: Whether to include ablation text in instructions.
            debug_response: Capture raw model response payloads for inspection.
            model_kwargs: Provider-specific model keyword arguments.
            objective: Goal sentence key from prompts.OBJECTIVES.
        """
        super().__init__(reservoir, model_id, include_red_herring=include_red_herring, objective=objective)

        self.model_server = model_server
        self.model_id = model_id
        self.model_kwargs = model_kwargs or {}
        self.debug_response = debug_response

        if model_server == "OpenAI":
            self._openai_client = OpenAI(
                api_key=self.model_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            )

    # --------------------------------------------------------------------- #
    # Decision
    # --------------------------------------------------------------------- #

    def make_allocation_decision(self, idx: int = 0) -> float:
        """Make a monthly allocation decision via the language model.

        Returns the normalized allocation percent (0–100). The reasoning,
        justification, and concept-importance rankings are written to
        ``self.record`` rather than returned.
        """
        system_content = f"{self.system_message}\n\n{self.instructions}"

        decision: AllocationDecision
        reasoning_text: str | None = None

        if self.model_server == "OpenAI":
            decision, reasoning_text = _with_retries(
                lambda: _call_openai(
                    self._openai_client,
                    self.model_id,
                    system_content,
                    self.observation,
                    temperature=self.model_kwargs.get("temperature"),
                    reasoning=self.model_kwargs.get("reasoning"),
                ),
                label="OpenAI",
            )
        elif self.model_server == "Ollama":
            decision, reasoning_text = _with_retries(
                lambda: _call_ollama(
                    self.model_id, system_content, self.observation, self.model_kwargs,
                ),
                label="Ollama",
            )
        else:
            raise ValueError(f"Unsupported model server: {self.model_server}")

        allocation_percent = self._normalize_allocation_percent(decision.allocation_percent)
        self._record_decision(idx, allocation_percent, decision)

        self.record.loc[idx, "model_reasoning"] = reasoning_text or "N/A"

        if self.debug_response:
            self.record.loc[idx, "response_debug"] = json.dumps(
                _to_serializable(decision), ensure_ascii=False,
            )

        return allocation_percent


# =============================================================================
# Shared Utilities
# =============================================================================

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
    objective: str = "minimize-shortages",
):
    """Build and return the reservoir operator implementation."""
    return ReservoirAllocationOperator(
        model_server=resolved_model_config.model_server,
        model_id=resolved_model_config.model,
        reservoir=reservoir,
        model_kwargs=resolved_model_config.model_kwargs,
        include_red_herring=include_red_herring,
        debug_response=debug_response,
        objective=objective,
    )
