#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""resllm.model_config

Centralized model configuration resolver for provider-specific runtime settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os


SUPPORTED_MODEL_SERVERS = {"Ollama", "OpenAI"}
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high"}

# OpenAI model families that do NOT support the Responses API reasoning
# parameter. Prefixes are matched against the model name.
_OPENAI_NON_REASONING_PREFIXES = ("gpt-4.1", "gpt-4o", "gpt-4-")


@dataclass(frozen=True)
class RunIntent:
    """User intent captured from CLI arguments."""

    model_server: str
    model: str
    reasoning_effort: str | None = "high"
    temperature: float | None = None


@dataclass
class ResolvedModelConfig:
    """Resolved provider configuration for simulation runtime."""

    model_server: str
    model: str
    model_kwargs: dict
    warnings: list[str] = field(default_factory=list)


def _normalize_reasoning_effort(reasoning_effort: str | None) -> str | None:
    if reasoning_effort is None:
        return None

    normalized = reasoning_effort.strip().lower()

    if normalized not in REASONING_EFFORTS:
        allowed = ", ".join(sorted(REASONING_EFFORTS))
        raise ValueError(
            f"Unsupported reasoning effort '{reasoning_effort}'. "
            f"Expected one of: {allowed}."
        )

    return normalized


def _ollama_think_from_effort(
    normalized_effort: str | None, *, is_cloud: bool = False
) -> bool | str:
    """Map normalized reasoning effort to Ollama ``think`` parameter.

    **Cloud models** accept ``bool | Literal['low','medium','high']`` for
    ``think``, which the SDK sends as ``reasoning_effort``.  **Local
    models** only accept ``bool``.

    ``"none"`` maps to ``False`` (disable thinking), ``None`` defaults to
    ``True``, and ``"minimal"`` is mapped to ``"low"`` for cloud models.
    """
    if normalized_effort is None:
        return True          # default: think enabled
    if normalized_effort == "none":
        return False
    if not is_cloud:
        return True          # local models: any effort → think=True
    if normalized_effort == "minimal":
        return "low"         # SDK doesn't accept "minimal"; closest is "low"
    return normalized_effort  # "low", "medium", "high"


def resolve_model_config(intent: RunIntent, env: dict | None = None) -> ResolvedModelConfig:
    """Resolve and validate provider-specific model configuration.

    Args:
        intent: Normalized run intent from CLI arguments.
        env: Optional environment dictionary (defaults to os.environ).

    Returns:
        ResolvedModelConfig containing sanitized kwargs and capability flags.
    """
    if intent.model_server not in SUPPORTED_MODEL_SERVERS:
        raise ValueError(f"Unsupported model server: {intent.model_server}")

    env_values = os.environ if env is None else env
    warnings: list[str] = []

    normalized_effort = _normalize_reasoning_effort(intent.reasoning_effort)

    if intent.model_server == "Ollama":
        is_cloud = intent.model.endswith("-cloud") or intent.model.endswith(":cloud")
        if normalized_effort == "minimal" and is_cloud:
            warnings.append(
                "Ollama SDK does not accept 'minimal' — "
                "mapping reasoning effort to 'low'."
            )
        think_value = _ollama_think_from_effort(normalized_effort, is_cloud=is_cloud)
        model_kwargs = {
            "temperature": intent.temperature,
            "think": think_value,
            "timeout": 3600,
        }

    elif intent.model_server == "OpenAI":
        model_kwargs = {
            "api_key": env_values.get("OPENAI_API_KEY"),
            "temperature": intent.temperature,
        }
        is_non_reasoning = intent.model.startswith(_OPENAI_NON_REASONING_PREFIXES)
        if normalized_effort is not None and is_non_reasoning:
            warnings.append(
                f"{intent.model} does not support reasoning traces — "
                f"running via Chat Completions without reasoning."
            )
        elif normalized_effort is not None:
            # OpenAI Responses API accepts none/low/medium/high — map minimal→low
            openai_effort = "low" if normalized_effort == "minimal" else normalized_effort
            model_kwargs["reasoning"] = {
                "effort": openai_effort,
                "summary": "detailed",
            }
            # OpenAI does not support temperature with reasoning enabled
            if intent.temperature is not None:
                warnings.append(
                    "OpenAI does not support temperature with reasoning enabled — "
                    "ignoring --temperature."
                )
                model_kwargs.pop("temperature", None)

    return ResolvedModelConfig(
        model_server=intent.model_server,
        model=intent.model,
        model_kwargs=model_kwargs,
        warnings=warnings,
    )
