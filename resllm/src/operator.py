#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resllm.operator

Operator agent for reservoir management. Calls native provider APIs (OpenAI,
Ollama) to compute monthly percent-allocation release decisions from current
storage, inflows, and remaining demand.
"""
from __future__ import annotations

import json
import os
import re
from typing import TypedDict

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field, create_model

from src.ablation import apply_ablation
from src.prompts import (
    CONCEPT_KEYS_SIMPLE,
    OLLAMA_JSON_INSTRUCTION,
    build_instructions,
    build_observation,
    build_ollama_json_instruction,
    build_system_message,
    get_concept_keys,
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


def build_decision_model(
    concept_keys: tuple[str, ...], complexity_mode: bool,
    include_concept_importance: bool = True,
):
    """Return the Pydantic decision model for the active mode.

    Simple mode returns the static :class:`AllocationDecision` unchanged. Complex mode
    builds an extended model (adds ``junior_delivery_percent``, ``meet_min_flow``,
    ``carryover_target_taf`` and the expanded concept set). The concept set is a
    dynamically-built ``TypedDict`` so the validated ``allocation_concept_importance``
    stays a plain dict supporting ``.items()`` like the static model.

    ``include_concept_importance=False`` drops the rankings — ablation runs use this so
    the model is never asked to rank a concept that may have been removed from the prompt.
    """
    if not complexity_mode:
        if include_concept_importance:
            return AllocationDecision
        # Reduced simple-mode schema for ablation runs (drops concept-importance rankings).
        return create_model(
            "AllocationDecision",
            allocation_reasoning=(
                str,
                Field(..., description="A brief justification of the percent allocation decision."),
            ),
            allocation_percent=(
                float,
                Field(
                    ...,
                    description=(
                        "The percent allocation decision (from 0-100 percent) which "
                        "continues or updates the allocation and release from the reservoir."
                    ),
                ),
            ),
        )

    field_defs = {
        "allocation_reasoning": (
            str,
            Field(..., description="A brief justification of the allocation decision."),
        ),
        "allocation_percent": (
            float,
            Field(..., description="The percent (0-100) of the supplemental downstream-flow "
                                  "demand to deliver; below 100 shorts downstream support and "
                                  "conserves the remainder as carryover."),
        ),
    }
    field_defs["junior_delivery_percent"] = (
        float,
        Field(..., description="The percent (0-100) of the junior committed water-supply "
                              "deliveries to make this month; below 100 is a supply shortage."),
    )
    field_defs["meet_min_flow"] = (
        bool,
        Field(..., description="Whether to deliver the full required minimum downstream flow "
                              "this month (true) or curtail it to the firm regulatory floor "
                              "(false); curtailing is a last-resort regulatory violation, "
                              "reserved for the most extreme low-storage conditions."),
    )
    field_defs["carryover_target_taf"] = (
        float,
        Field(..., description="End-of-season carryover storage target in TAF; -1 for no target."),
    )
    if include_concept_importance:
        concepts_td = TypedDict("OperationalConcepts", {k: int for k in concept_keys})
        field_defs["allocation_concept_importance"] = (concepts_td, ...)
    return create_model("AllocationDecision", **field_defs)


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
        ablation_type: str | None = None,
    ):
        if reservoir is None:
            raise ValueError("reservoir cannot be None")
        self.reservoir = reservoir
        self.complexity_mode = bool(getattr(reservoir, "complexity_mode", False))
        self.ablation_type = ablation_type

        # Ablation runs drop concept-importance rankings from both prompt and schema:
        # ranking a concept that may have been removed would reintroduce it.
        self._include_concept_importance = ablation_type is None

        # Active decision schema (single source of truth for prompt + parsing).
        self._concept_keys = get_concept_keys(self.complexity_mode)
        self._decision_model = build_decision_model(
            self._concept_keys, self.complexity_mode,
            include_concept_importance=self._include_concept_importance,
        )
        self._ollama_instruction = build_ollama_json_instruction(
            self._concept_keys, self.complexity_mode,
            include_concept_importance=self._include_concept_importance,
        )

        self.system_message = build_system_message(objective)
        self.instructions = build_instructions(
            reservoir, include_red_herring,
            include_importance_ranking=self._include_concept_importance,
            complexity_mode=self.complexity_mode,
        )
        self.observation: str | None = None
        self.last_decision = None
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
        spill_vol: float = 0.0,
    ):
        """Compute observation data from the reservoir and build the prompt.

        In complex mode, ``spill_vol`` is the held physics state from the daily loop, so the
        observation shows the same American River Index the physics will enforce.
        """
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

        if not self.complexity_mode:
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
        else:
            self._set_complex_observation(
                idx=idx, date=date, wy=wy, mowy=mowy, dowy=dowy, alloc_1=alloc_1,
                st_1=st_1, d_wy_rem=d_wy_rem, qwyaccum=qwyaccum,
                qwy_forecast_mean=qwy_forecast_mean,
                qwy_forecast_10=qwy_forecast_10,
                qwy_forecast_90=qwy_forecast_90,
                next_wy_demand=next_wy_demand,
                spill_vol=spill_vol,
            )

        self.record.loc[idx, "date"] = date
        self.record.loc[idx, "wy"] = wy
        self.record.loc[idx, "mowy"] = mowy
        self.record.loc[idx, "dowy"] = dowy
        self.record.loc[idx, "qwyaccum"] = qwyaccum
        self.record.loc[idx, "d_wy_rem"] = d_wy_rem
        self.record.loc[idx, "st_1"] = st_1

    def _set_complex_observation(
        self, *, idx, date, wy, mowy, dowy, alloc_1, st_1, d_wy_rem, qwyaccum,
        qwy_forecast_mean, qwy_forecast_10, qwy_forecast_90, next_wy_demand,
        spill_vol=0.0,
    ):
        """Build the complex-mode observation and record its context columns."""
        res = self.reservoir

        # Forecast percentiles. Volume percentile ↔ exceedance code:
        # 25th pctile = 75% exceedance = QCYFHG; 75th pctile = 25% exceedance = QCYFHH.
        median = pct25 = pct75 = None
        if res.characteristics["wy_forecast_file"] is not False:
            row = res.forecasted_inflows.loc[res.forecasted_inflows["date"] == date]
            median = int(row["QCYFH5"].values[0])
            pct25 = int(row["QCYFHG"].values[0])
            pct75 = int(row["QCYFHH"].values[0])

        # Held monthly drivers (match the daily-loop physics). The American River Index
        # drives the supplemental demand and Water-Forum cutback; the dry/normal/wet class
        # is only a legibility label.
        wy_index = qwyaccum + qwy_forecast_mean if qwy_forecast_mean is not None else res.average_wy_total_inflow
        ari = res.american_river_index(wy_index, spill_vol)
        hydro_class = res.classify_water_year(wy_index)
        near_term = res.near_term_inflow(date)
        # Required minimum flow (FMS Minimum Flows Requirement) and current flood limit.
        # The firm floor (always released) is floor_frac of the required flow; the agent's
        # meet_min_flow governs whether the discretionary remainder above the floor is delivered.
        min_flow = res.compute_min_flow()
        min_flow_floor = min_flow * float(
            res.complexity.get("min_flow_decision", {}).get("floor_frac", 0.5)
        )
        tocs = res.compute_tocs(dowy=dowy, date=date.strftime("%Y-%m-%d"),
                                near_term=near_term)
        # Demand stack for this month (additive, all served by the release): senior
        # committed M&I is firm/auto-served; junior committed M&I is the agent's delivery
        # choice; above these sits the supplemental downstream-flow demand the agent's
        # allocation governs. Monthly totals.
        days = res._WY_MONTH_DAYS[mowy - 1]
        # wf_factor is the hydrology-suggested junior delivery fraction (a soft reference);
        # the agent sets the actual junior_delivery_percent.
        wf_factor = res.compute_water_forum_factor(wy_index)
        umi_month = res.upstream_mi_day(mowy, dowy) * days
        senior_mi = umi_month * res._upstream_mi_senior_frac          # firm, auto-served
        junior_mi_full = umi_month * res._upstream_mi_wf_cvp_frac     # full junior; agent sets % delivered
        # Lower-American M&I is diverted below the dam from the released flow, so it is not
        # a separate storage draw here.
        # Supplemental (Delta-support) demand is indexed on the annual water-year hydrology
        # (wy_index), NOT the spill-decremented running ARI: the downstream obligation is set by
        # water-year type and does not relax just because Folsom spilled over winter/spring.
        delta_demand = res.delta_demand_day(mowy, wy_index, st=st_1) * days

        self.observation = build_observation(
            mowy=mowy, st_1=st_1, d_wy_rem=d_wy_rem, alloc_1=alloc_1,
            qwyaccum=qwyaccum,
            qwy_forecast_mean=qwy_forecast_mean,
            qwy_forecast_10=qwy_forecast_10,
            qwy_forecast_90=qwy_forecast_90,
            next_wy_demand=next_wy_demand,
            complexity_mode=True,
            qwy_forecast_median=median,
            qwy_forecast_25=pct25,
            qwy_forecast_75=pct75,
            near_term=near_term,
            hydro_class=hydro_class,
            wy_index=wy_index,
            min_flow=min_flow,
            min_flow_month=min_flow * days,
            min_flow_floor=min_flow_floor,
            tocs=tocs,
            senior_mi=senior_mi,
            junior_mi_full=junior_mi_full,
            supplemental=delta_demand,
            wf_factor=wf_factor,
            supplemental_rem=res.delta_demand_remaining(mowy, wy_index),
            next_wy_committed=res.committed_first_months(3),
        )

        # Record context to the decision output (the flush drops only on allocation_percent).
        self.record.loc[idx, "hydro_class"] = hydro_class
        self.record.loc[idx, "wy_index"] = wy_index
        self.record.loc[idx, "ari"] = ari
        self.record.loc[idx, "min_flow_obs"] = min_flow
        self.record.loc[idx, "min_flow_floor"] = min_flow_floor
        self.record.loc[idx, "tocs_obs"] = tocs
        self.record.loc[idx, "senior_mi"] = senior_mi
        self.record.loc[idx, "junior_mi_full"] = junior_mi_full
        self.record.loc[idx, "delta_demand"] = delta_demand
        self.record.loc[idx, "wf_factor"] = wf_factor

    # --------------------------------------------------------------------- #
    # Decision helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _normalize_allocation_percent(value: float) -> float:
        """Clamp allocation to 0–100, treating 0 < value < 1 as a fraction."""
        if 0 < value < 1:
            value *= 100
        return max(0.0, min(100.0, value))

    def _record_decision(
        self, idx: int, allocation_percent: float, decision,
        *, sent_system: str | None = None, sent_user: str | None = None,
    ):
        """Record the allocation decision and concept importance.

        ``sent_system``/``sent_user`` are the exact strings sent to the model after any
        ablation. On ablation runs the recorded ``observation`` is that post-ablation prompt
        (so the artifact reflects the removed material); otherwise it is the full rendered prompt.
        """
        if getattr(self, "ablation_type", None):
            # Record what was actually sent (post-ablation), not the original prompt.
            self.record.loc[idx, "observation"] = (sent_system or "") + (sent_user or "")
            # Tag the ablated element (written only when ablating, so non-ablation runs
            # don't gain an extra column).
            self.record.loc[idx, "ablation_type"] = self.ablation_type
        else:
            self.record.loc[idx, "observation"] = (
                self.system_message + self.instructions + self.observation
            )
        self.record.loc[idx, "allocation_percent"] = allocation_percent
        self.record.loc[idx, "allocation_justification"] = decision.allocation_reasoning
        if self.complexity_mode:
            self.record.loc[idx, "junior_allocation_percent"] = float(
                getattr(decision, "junior_delivery_percent", 100.0)
            )
            self.record.loc[idx, "carryover_target_taf"] = float(
                getattr(decision, "carryover_target_taf", -1.0)
            )
            self.record.loc[idx, "meet_min_flow"] = bool(
                getattr(decision, "meet_min_flow", True)
            )
        # Concept-importance rankings are absent on ablation runs (reduced schema).
        aci = getattr(decision, "allocation_concept_importance", None)
        if aci:
            for key, value in aci.items():
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
    decision_model=AllocationDecision,
) -> tuple[AllocationDecision, str | None]:
    """Call the OpenAI Responses API with reasoning support.

    The Responses API (``/v1/responses``) is the only OpenAI endpoint that accepts the
    ``reasoning`` parameter and returns reasoning summaries.

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

    # Extract structured output text and reasoning summaries.
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

    return decision_model(**json.loads(response_text)), reasoning_text


def _call_openai(
    client: OpenAI,
    model: str,
    system_content: str,
    user_content: str,
    temperature: float | None,
    reasoning: dict | None = None,
    decision_model=AllocationDecision,
) -> tuple[AllocationDecision, str | None]:
    """Call OpenAI with structured JSON output.

    With reasoning enabled, uses the Responses API (``/v1/responses``). Otherwise uses
    Chat Completions with a strict ``json_schema`` response format (no reasoning trace).

    Returns:
        (AllocationDecision, reasoning_text_or_None)
    """
    schema = decision_model.model_json_schema()
    _add_strict_additional_properties(schema)

    # Reasoning traces are only available via the Responses API.
    if reasoning is not None:
        return _call_openai_responses(
            client, model, system_content, user_content, schema,
            temperature, reasoning, decision_model=decision_model,
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
    return decision_model(**json.loads(response_text)), None


def _call_ollama(
    model_id: str,
    system_content: str,
    user_content: str,
    model_kwargs: dict,
    ollama_instruction: str = OLLAMA_JSON_INSTRUCTION,
    decision_model=AllocationDecision,
    concept_keys: tuple[str, ...] = CONCEPT_KEYS_SIMPLE,
) -> tuple[AllocationDecision, str | None]:
    """Call Ollama via the generate endpoint with optional thinking.

    Uses ``generate`` rather than ``chat`` because thinking models often leave the chat
    ``content`` field empty. When thinking is enabled, ``format="json"`` is omitted so the
    model reasons freely; the system-prompt JSON instruction guides output format instead.

    Returns:
        (AllocationDecision, thinking_text_or_None)
    """
    from ollama import generate

    system_prompt = f"{system_content}{ollama_instruction}"
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

    # Stream chunks, capturing thinking traces chunk-by-chunk.
    thinking_parts: list[str] = []
    content_parts: list[str] = []
    for chunk in generate(**kwargs):
        if t := getattr(chunk, "thinking", None):
            thinking_parts.append(t)
        if c := getattr(chunk, "response", None):
            content_parts.append(c)
    raw_text = "".join(content_parts).strip()
    thinking_text = "".join(thinking_parts).strip() or None

    # Some thinking models place everything in the thinking field.
    if not raw_text and thinking_text:
        raw_text, thinking_text = _split_thinking_json(thinking_text)

    decision = _parse_ollama_decision(raw_text, decision_model, concept_keys)
    return decision, thinking_text


def _split_thinking_json(text: str) -> tuple[str, str | None]:
    """Separate a JSON payload from preceding reasoning in a thinking blob.

    Some Ollama thinking models place everything — free-text reasoning *and* the final
    JSON object — in the ``thinking`` field, leaving ``response`` empty. This extracts the
    last top-level JSON object (``{…}``) as the payload and treats everything before it as
    the reasoning trace.

    Returns:
        (json_text, reasoning_text_or_None)
    """
    # Find the last top-level '{'...'}' block using brace depth.
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
        # Also grab any text after the JSON (rare but possible).
        trailing = text[obj_end:].strip()
        if trailing:
            reasoning = f"{reasoning}\n{trailing}".strip() if reasoning else trailing
        return json_text, reasoning or None

    # No JSON object found — return the whole thing (will fail at parse).
    return text, None


def _parse_ollama_decision(
    raw_text: str,
    decision_model=AllocationDecision,
    concept_keys: tuple[str, ...] = CONCEPT_KEYS_SIMPLE,
) -> "AllocationDecision":
    """Parse and normalize an Ollama JSON response into a decision model."""
    payload = _parse_json_response(raw_text)
    payload = _normalize_decision_payload(payload, concept_keys=concept_keys)
    return decision_model(**payload)


# =============================================================================
# Response normalization (module-level for reuse)
# =============================================================================

def _sanitize_json_string(text: str) -> str:
    """Escape literal control characters inside JSON string values.

    Some providers return JSON with raw newlines/tabs inside quoted values instead of the
    required ``\\n``/``\\t`` escapes; this fixes that so ``json.loads`` succeeds.
    """
    # Operate character-by-character so structural whitespace between keys is left intact.
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


def _normalize_decision_payload(
    payload: dict, concept_keys: tuple[str, ...] = CONCEPT_KEYS_SIMPLE
) -> dict:
    """Normalize provider JSON keys to expected decision-model fields."""
    if not isinstance(payload, dict):
        return payload

    # camelCase → snake_case (incl. complex-mode secondary fields).
    key_map = {
        "allocationReasoning": "allocation_reasoning",
        "allocationPercent": "allocation_percent",
        "allocationConceptImportance": "allocation_concept_importance",
        "carryoverTargetTaf": "carryover_target_taf",
        "carryoverTarget": "carryover_target_taf",
    }
    normalized = {key_map.get(k, k): v for k, v in payload.items()}

    # Unwrap if the model nested the response under a single key.
    expected_fields = {"allocation_reasoning", "allocation_percent", "allocation_concept_importance"}
    if not (expected_fields & normalized.keys()):
        if len(normalized) == 1:
            inner = next(iter(normalized.values()))
            if isinstance(inner, dict):
                return _normalize_decision_payload(inner, concept_keys=concept_keys)

    # Normalize concept-importance keys.
    aci = normalized.get("allocation_concept_importance")
    if isinstance(aci, dict):
        normalized["allocation_concept_importance"] = _normalize_concept_keys(
            aci, concept_keys
        )

    return normalized


def _normalize_concept_keys(
    aci: dict, concept_keys: tuple[str, ...] = CONCEPT_KEYS_SIMPLE
) -> dict:
    """Fuzzy-match concept importance keys to expected field names."""
    expected = set(concept_keys)

    # Fast path: keys already match.
    if set(aci.keys()) == expected:
        return aci

    # Order matters: more-specific patterns first. The environmental-flow pattern
    # must precede the bare "environment" pattern so it isn't swallowed by it.
    patterns = [
        (("minimum", "environment"),          "minimum_environmental_flow"),
        (("environment", "flow"),             "minimum_environmental_flow"),
        (("environment",),                    "environment_setting"),
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
        # Complex-mode concepts
        (("hydrolog",),                       "hydrologic_class"),
        (("flood",),                          "flood_control_curve"),
        (("carryover",),                      "carryover_storage_target"),
        (("supplemental",),                   "supplemental_flow_demand"),
        (("downstream", "flow"),              "supplemental_flow_demand"),
        (("surplus",),                        "supplemental_flow_demand"),
        (("discretionary",),                  "supplemental_flow_demand"),
        (("committed",),                      "committed_water_supply"),
        (("water", "supply"),                 "committed_water_supply"),
    ]

    mapping: dict[str, int] = {}
    for raw_key, value in aci.items():
        normalized_key = " ".join(str(raw_key).strip().lower().replace("_", " ").split())
        for keywords, target in patterns:
            if target not in expected:
                continue
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
        ablation_type: str | None = None,
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
            ablation_type: If set, one element is removed from every rendered prompt before
                the model call (see src.ablation.ABLATION_TYPES). ``None`` leaves the prompt
                untouched.
        """
        super().__init__(
            reservoir, model_id, include_red_herring=include_red_herring,
            objective=objective, ablation_type=ablation_type,
        )

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

        # Ablate one element from the rendered prompt before the call.
        # With ablation_type=None this is an exact pass-through.
        system_for_call, obs_for_call = apply_ablation(
            system_content, self.observation, self.ablation_type,
            complexity_mode=self.complexity_mode,
        )

        decision: AllocationDecision
        reasoning_text: str | None = None

        decision_model = self._decision_model
        ollama_instruction = self._ollama_instruction

        if self.model_server == "OpenAI":
            decision, reasoning_text = _with_retries(
                lambda: _call_openai(
                    self._openai_client,
                    self.model_id,
                    system_for_call,
                    obs_for_call,
                    temperature=self.model_kwargs.get("temperature"),
                    reasoning=self.model_kwargs.get("reasoning"),
                    decision_model=decision_model,
                ),
                label="OpenAI",
            )
        elif self.model_server == "Ollama":
            decision, reasoning_text = _with_retries(
                lambda: _call_ollama(
                    self.model_id, system_for_call, obs_for_call, self.model_kwargs,
                    ollama_instruction=ollama_instruction,
                    decision_model=decision_model,
                    concept_keys=self._concept_keys,
                ),
                label="Ollama",
            )
        else:
            raise ValueError(f"Unsupported model server: {self.model_server}")

        # Stash the full parsed decision so the simulation loop can read the secondary
        # complex-mode fields (junior_delivery_percent, meet_min_flow, carryover_target_taf).
        self.last_decision = decision
        allocation_percent = self._normalize_allocation_percent(decision.allocation_percent)
        self._record_decision(
            idx, allocation_percent, decision,
            sent_system=system_for_call, sent_user=obs_for_call,
        )

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
    ablation_type: str | None = None,
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
        ablation_type=ablation_type,
    )
