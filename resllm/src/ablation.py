#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resllm.ablation

Single source of truth for prompt *ablation* — removing one element from a rendered
reservoir-operator prompt to measure that element's influence on the allocation decision.

Two consumers share this module:
  * the online full ablated simulation (``simulate.py`` / ``operator.py``) — Option A,
  * the online re-decision study (``run_ablation_redecision.py``) — Option B.

Two prompt modes are supported:
  * **simple** — patterns *frozen* so the published simple-mode ablation study reproduces
    byte-for-byte. They match the historical rendered
    prompt text, which differs slightly from the current ``prompts.py``, so they are NOT
    re-derived from the current templates.
  * **complex** — patterns written against the current ``src/prompts.py`` constants;
    :func:`_verify_complex_anchors` asserts at import that each anchor is still a substring of
    its source template, so the patterns fail loudly if the prompts drift.

Public API:
  * :data:`ABLATION_TYPES` — all valid ablation names (argparse choices for the online paths).
  * :func:`split_system_and_user` — split a rendered prompt at the ``OBSERVATION_MONTH`` anchor.
  * :func:`remove_element_from_observation` — text surgery on one blob (system *or* user).
  * :func:`apply_ablation` — ablate a ``(system_content, observation)`` pair (online paths).

Text surgery operates line-by-line on a single rendered blob, so the same function ablates
either the system message (system + instructions) or the user observation; callers split the
prompt first and apply it to each part.
"""
from __future__ import annotations

from src.prompts import (
    INSTRUCTIONS_CARRYOVER,
    INSTRUCTIONS_REMAINING_SUPPLEMENTAL_HEADER,
    OBSERVATION_FLOOD_LIMIT,
    OBSERVATION_FORECAST_COMPLEX,
    OBSERVATION_HYDRO_CLASS,
    OBSERVATION_INFLOW_TO_DATE,
    OBSERVATION_MIN_FLOW,
    OBSERVATION_MONTH,
    OBSERVATION_NEAR_TERM,
    OBSERVATION_NEXT_WY_COMMITTED,
    OBSERVATION_RELEASE_STRUCTURE,
    OBSERVATION_REMAINING_SUPPLEMENTAL,
    OBSERVATION_STORAGE,
)


# =============================================================================
# Ablation taxonomy
# =============================================================================

# Simple-mode types — frozen to reproduce the published simple-mode ablation study byte-for-byte.
SIMPLE_ABLATION_TYPES: tuple[str, ...] = (
    "current_storage",
    "forecasts",
    "forecast_p10",
    "forecast_mean",
    "forecast_p90",
    "previous_allocation",
    "demand",
    "current_month",
    "cumulative_inflow",
    "storage_and_inflow",
    "no_system",
    "minimal",
    "bare_minimal",
    "default",
)

# Complex-only element types — lines that exist only in the complex prompt.
COMPLEX_ONLY_ABLATION_TYPES: tuple[str, ...] = (
    "near_term",
    "hydrologic_class",
    "min_flow",
    "flood_limit",
    "release_structure",
    "carryover",
)

# Full set exposed to the online paths (red_herring = ablate the puppies text on demand).
ABLATION_TYPES: tuple[str, ...] = (
    SIMPLE_ABLATION_TYPES + ("red_herring",) + COMPLEX_ONLY_ABLATION_TYPES
)

# Structural ablations replace the message rather than removing an element.
STRUCTURAL_ABLATION_TYPES: tuple[str, ...] = ("no_system", "minimal", "bare_minimal")

# Stub prompts for the structural ablations.
MINIMAL_PROMPT = (
    "You are operating a water-supply reservoir. Provide a percent allocation "
    "decision (from 0-100 percent) which continues or updates the allocation."
)
BARE_MINIMAL_PROMPT = "Provide a percent value from 0 to 100."

# The system/user split anchor (start of the user observation).
_SPLIT_ANCHOR = "It is the beginning of month"


# =============================================================================
# System / user split
# =============================================================================

def split_system_and_user(observation: str) -> tuple[str, str]:
    """Split a rendered prompt into ``(system_message, user_message)``.

    The user section begins at the first line starting with :data:`_SPLIT_ANCHOR`
    (``"It is the beginning of month"``); everything before it is the system message
    (system message + instructions).
    """
    lines = observation.split("\n")
    system_lines: list[str] = []
    user_lines: list[str] = []
    in_user_section = False

    for line in lines:
        if line.strip().startswith(_SPLIT_ANCHOR):
            in_user_section = True
        (user_lines if in_user_section else system_lines).append(line)

    return "\n".join(system_lines), "\n".join(user_lines)


# =============================================================================
# Simple-mode removal (FROZEN — reproduces the published study byte-for-byte)
# =============================================================================

def _remove_simple(observation: str, ablation_type: str, strip: set[str]) -> str:
    """Frozen simple-mode element removal, preserved byte-for-byte for study reproduction.

    ``strip`` names the always-removed lines (importance ranking / puppies). Pass
    ``{"importance_ranking", "red_herring"}`` to strip both unconditionally; the online paths
    pass a narrower (often empty) set.
    """
    lines = observation.split("\n")
    filtered_lines: list[str] = []
    skip_forecast_data = False
    skip_forecast_instruction = False

    for line in lines:
        # Always-strip lines (gated so the online faithful paths can keep them).
        if "importance_ranking" in strip and "Assign an importance ranking" in line:
            continue
        if "red_herring" in strip and "Puppies like to play" in line:
            continue

        if ablation_type == "current_storage":
            if "There is currently" in line and "TAF in storage" in line:
                continue
            if "consider the volume currently in storage" in line:
                line = line.replace(", the volume currently in storage,", "")
                line = line.replace(
                    "consider the volume currently in storage, inflow", "consider inflow"
                )

        elif ablation_type == "forecasts":
            if "The probabilistic forecasted inflows for the remainder of the water year are:" in line:
                skip_forecast_data = True
                continue
            if skip_forecast_data:
                if (
                    line.strip().startswith("- Mean")
                    or line.strip().startswith("- 10th")
                    or line.strip().startswith("- 90th")
                ):
                    continue
                else:
                    skip_forecast_data = False
            if "Starting in month 4 of the water year, you have access to a probabilistic forecast" in line:
                skip_forecast_instruction = True
                continue
            if skip_forecast_instruction:
                if "- The probabilistic forecast includes" in line or "- Use this forecast to inform" in line:
                    continue
                else:
                    skip_forecast_instruction = False

        elif ablation_type == "forecast_p10":
            if "The probabilistic forecasted inflows for the remainder of the water year are:" in line:
                skip_forecast_data = True
            elif skip_forecast_data:
                if line.strip().startswith("- 10th"):
                    continue
                elif line.strip().startswith("- Mean") or line.strip().startswith("- 90th"):
                    pass
                elif line.strip() and not line.strip().startswith("-"):
                    skip_forecast_data = False
            if "- The probabilistic forecast includes the ensemble mean, and 10th and 90th percentile" in line:
                line = line.replace(
                    "the ensemble mean, and 10th and 90th percentile",
                    "the ensemble mean and 90th percentile",
                )

        elif ablation_type == "forecast_mean":
            if "The probabilistic forecasted inflows for the remainder of the water year are:" in line:
                skip_forecast_data = True
            elif skip_forecast_data:
                if line.strip().startswith("- Mean"):
                    continue
                elif line.strip().startswith("- 10th") or line.strip().startswith("- 90th"):
                    pass
                elif line.strip() and not line.strip().startswith("-"):
                    skip_forecast_data = False
            if "- The probabilistic forecast includes the ensemble mean, and 10th and 90th percentile" in line:
                line = line.replace(
                    "the ensemble mean, and 10th and 90th percentile",
                    "10th and 90th percentile",
                )

        elif ablation_type == "forecast_p90":
            if "The probabilistic forecasted inflows for the remainder of the water year are:" in line:
                skip_forecast_data = True
            elif skip_forecast_data:
                if line.strip().startswith("- 90th"):
                    continue
                elif line.strip().startswith("- Mean") or line.strip().startswith("- 10th"):
                    pass
                elif line.strip() and not line.strip().startswith("-"):
                    skip_forecast_data = False
            if "- The probabilistic forecast includes the ensemble mean, and 10th and 90th percentile" in line:
                line = line.replace(
                    "the ensemble mean, and 10th and 90th percentile",
                    "the ensemble mean and 10th percentile",
                )

        elif ablation_type == "previous_allocation":
            if "The previous percent allocation decision was" in line:
                continue

        elif ablation_type == "demand":
            if "There is approximately" in line and "TAF of water demand to meet over the remainder" in line:
                continue
            if "Also, note that next water year is approaching" in line:
                continue
            if "The average remaining demand by beginning of month of the water year:" in line:
                continue
            if "The average total water year demand:" in line:
                continue
            if "balance meeting current demands against conserving water for future demands" in line:
                line = line.replace(
                    ", inflow to date compared to expected inflows, and the need to balance "
                    "meeting current demands against conserving water for future demands",
                    " and inflow to date compared to expected inflows",
                )

        elif ablation_type == "current_month":
            if "It is the beginning of month" in line and "of the water year." in line:
                continue

        elif ablation_type == "cumulative_inflow":
            if "So far this water year," in line and "TAF of reservoir inflow has been observed" in line:
                continue
            if "The average cumulative inflow by beginning of month of the water year:" in line:
                continue
            if "inflow to date compared to expected inflows" in line:
                line = line.replace(", inflow to date compared to expected inflows,", "")
                line = line.replace(
                    "consider the volume currently in storage, inflow to date compared to "
                    "expected inflows, and the need",
                    "consider the volume currently in storage and the need",
                )

        elif ablation_type == "storage_and_inflow":
            if "There is currently" in line and "TAF in storage" in line:
                continue
            if "The average cumulative inflow by beginning of month of the water year:" in line:
                continue
            if "So far this water year," in line and "TAF of reservoir inflow has been observed" in line:
                continue
            if "consider the volume currently in storage, inflow to date compared to expected inflows" in line:
                line = line.replace(
                    "consider the volume currently in storage, inflow to date compared to "
                    "expected inflows, and the need",
                    "consider the volume currently in storage and the need",
                )

        elif ablation_type in ("no_system", "minimal", "bare_minimal", "default", "red_herring"):
            # No per-line removal here — only the always-strip set (above) applies.
            pass

        filtered_lines.append(line)

    return "\n".join(filtered_lines)


# =============================================================================
# Complex-mode removal (written against the current src/prompts.py)
# =============================================================================

# Static substring anchors of the complex observation / instruction lines. Each is verified
# at import time to still be present in its source template (see _verify_complex_anchors).
_A_STORAGE = "There is currently"
_A_INFLOW = "So far this water year,"
_A_MONTH = "It is the beginning of month"
_A_NEAR_TERM = "The near-term inflow outlook for the next few months is approximately"
_A_HYDRO = "The projected full-water-year inflow is approximately"
_A_MIN_FLOW = "The required minimum downstream flow this month is about"
_A_FLOOD = "The current flood-control storage limit is"
_A_RELEASE = "Above the minimum flow, about"
_A_RELEASE2 = "senior committed deliveries are served automatically"
_A_CARRYOVER = "The carryover target paces how fast you draw storage down"
_A_SUPP_REM = "supplemental downstream-flow demand remains over the rest of the water year"
_A_SUPP_HEADER = "The average remaining supplemental downstream-flow demand by beginning of month"
_A_NEXT_WY = "Also, note that next water year is approaching"
_A_PREV_ALLOC = "The previous supplemental-flow allocation was"
_A_FORECAST_HEADER = "The probabilistic forecasted inflows for the remainder of the water year are:"
_A_FORECAST_INSTR = (
    "- You have access to a probabilistic forecast",
    "- The probabilistic forecast includes",
    "- Use this forecast to inform",
    "- You also have a near-term",
)
# Complex forecast block bullets (mean + 10/25/50/75/90 percentiles).
_FORECAST_BULLETS = ("- Mean", "- 10th", "- 25th", "- 50th", "- 75th", "- 90th")


def _remove_complex(observation: str, ablation_type: str, strip: set[str]) -> str:
    """Element removal for the *complex* prompt (current ``prompts.py`` text).

    Mirrors :func:`_remove_simple` structurally but targets the complex-mode anchors and the
    forecast block. Single-percentile forecast ablations drop only the data line (the
    instruction's percentile description is left intact); the full ``forecasts`` ablation
    removes the whole block and the forecast instruction lines.
    """
    lines = observation.split("\n")
    out: list[str] = []
    skip_forecast_data = False

    for line in lines:
        stripped = line.strip()

        # Always-strip lines (gated; online faithful paths keep them).
        if "importance_ranking" in strip and "Assign an importance ranking" in line:
            continue
        if "red_herring" in strip and "Puppies like to play" in line:
            continue

        if ablation_type == "current_storage":
            if _A_STORAGE in line and "TAF in storage" in line:
                continue
            if "consider the volume currently in storage" in line:
                line = line.replace(", the volume currently in storage,", "")
                line = line.replace(
                    "consider the volume currently in storage, inflow", "consider inflow"
                )

        elif ablation_type == "cumulative_inflow":
            if _A_INFLOW in line and "TAF of reservoir inflow has been observed" in line:
                continue
            if "The average cumulative inflow by beginning of month of the water year:" in line:
                continue
            if "inflow to date compared to expected inflows" in line:
                line = line.replace(", inflow to date compared to expected inflows,", "")

        elif ablation_type == "storage_and_inflow":
            if _A_STORAGE in line and "TAF in storage" in line:
                continue
            if "The average cumulative inflow by beginning of month of the water year:" in line:
                continue
            if _A_INFLOW in line and "TAF of reservoir inflow has been observed" in line:
                continue
            if "consider the volume currently in storage, inflow to date compared to expected inflows" in line:
                line = line.replace(
                    "consider the volume currently in storage, inflow to date compared to "
                    "expected inflows, and the",
                    "consider the volume currently in storage and the",
                )

        elif ablation_type == "current_month":
            if _A_MONTH in line and "of the water year." in line:
                continue

        elif ablation_type == "previous_allocation":
            if _A_PREV_ALLOC in line:
                continue

        elif ablation_type == "forecasts":
            if _A_FORECAST_HEADER in line:
                skip_forecast_data = True
                continue
            if skip_forecast_data:
                if any(stripped.startswith(b) for b in _FORECAST_BULLETS):
                    continue
                skip_forecast_data = False
            if any(a in line for a in _A_FORECAST_INSTR):
                continue

        elif ablation_type == "forecast_p10":
            if _A_FORECAST_HEADER in line:
                skip_forecast_data = True
            elif skip_forecast_data:
                if stripped.startswith("- 10th"):
                    continue
                elif stripped and not stripped.startswith("-"):
                    skip_forecast_data = False

        elif ablation_type == "forecast_mean":
            if _A_FORECAST_HEADER in line:
                skip_forecast_data = True
            elif skip_forecast_data:
                if stripped.startswith("- Mean"):
                    continue
                elif stripped and not stripped.startswith("-"):
                    skip_forecast_data = False

        elif ablation_type == "forecast_p90":
            if _A_FORECAST_HEADER in line:
                skip_forecast_data = True
            elif skip_forecast_data:
                if stripped.startswith("- 90th"):
                    continue
                elif stripped and not stripped.startswith("-"):
                    skip_forecast_data = False

        elif ablation_type == "demand":
            # Complex analog of "demand": the supplemental downstream-flow demand context
            # (current remaining, the by-month table, and the next-WY committed note).
            if _A_SUPP_REM in line:
                continue
            if _A_SUPP_HEADER in line:
                continue
            if _A_NEXT_WY in line:
                continue

        elif ablation_type == "near_term":
            if _A_NEAR_TERM in line:
                continue

        elif ablation_type == "hydrologic_class":
            if _A_HYDRO in line:
                continue

        elif ablation_type == "min_flow":
            if _A_MIN_FLOW in line:
                continue

        elif ablation_type == "flood_limit":
            if _A_FLOOD in line:
                continue

        elif ablation_type == "release_structure":
            if _A_RELEASE in line and _A_RELEASE2 in line:
                continue

        elif ablation_type == "carryover":
            if _A_CARRYOVER in line:
                continue

        elif ablation_type in ("no_system", "minimal", "bare_minimal", "default", "red_herring"):
            pass

        out.append(line)

    return "\n".join(out)


# =============================================================================
# Public surgery API
# =============================================================================

def remove_element_from_observation(
    text: str,
    ablation_type: str,
    *,
    complexity_mode: bool = False,
    extra_strip: tuple[str, ...] = (),
) -> str:
    """Remove ``ablation_type``'s element from one rendered prompt blob (system or user).

    Args:
        text: A rendered system or user message.
        ablation_type: Which element to remove (see :data:`ABLATION_TYPES`).
        complexity_mode: Use the complex-prompt patterns; otherwise the frozen simple-mode
            patterns (byte-identical to the published study).
        extra_strip: Extra always-removed elements. Passing
            ``("importance_ranking", "red_herring")`` drops concept rankings and the red herring
            unconditionally (reduced-schema behavior); the online faithful paths pass ``()`` to
            keep concept rankings and the red herring (the red herring stays removable via
            ``ablation_type="red_herring"``).
    """
    if ablation_type not in ABLATION_TYPES:
        raise ValueError(
            f"Unknown ablation_type {ablation_type!r}. Choose from: {', '.join(ABLATION_TYPES)}"
        )
    strip = set(extra_strip)
    if ablation_type == "red_herring":
        strip.add("red_herring")
    if complexity_mode:
        return _remove_complex(text, ablation_type, strip)
    return _remove_simple(text, ablation_type, strip)


def apply_ablation(
    system_content: str,
    observation: str,
    ablation_type: str | None,
    *,
    complexity_mode: bool = False,
) -> tuple[str, str]:
    """Ablate a ``(system_content, observation)`` pair for the online call layer.

    Returns the transformed ``(system_out, user_out)`` ready for ``_call_openai`` /
    ``_call_ollama``. ``None`` or ``"default"`` is a pass-through. Structural ablations drop
    the system (``no_system``) or replace the user with a stub (``minimal`` / ``bare_minimal``).
    """
    if ablation_type in (None, "default"):
        return system_content, observation
    if ablation_type == "no_system":
        return "", observation
    if ablation_type == "minimal":
        return "", MINIMAL_PROMPT
    if ablation_type == "bare_minimal":
        return "", BARE_MINIMAL_PROMPT
    system_out = remove_element_from_observation(
        system_content, ablation_type, complexity_mode=complexity_mode
    )
    user_out = remove_element_from_observation(
        observation, ablation_type, complexity_mode=complexity_mode
    )
    return system_out, user_out


# =============================================================================
# Import-time sync check (complex anchors must still exist in prompts.py)
# =============================================================================

def _verify_complex_anchors() -> None:
    """Assert each complex anchor is a substring of its source ``prompts.py`` template.

    Fails loudly at import if the complex prompts drift, so the complex-mode patterns can
    never silently no-op. Simple-mode patterns are NOT checked — they are frozen against the
    historical prompt text to preserve byte-identical study reproduction.
    """
    checks = [
        (_A_STORAGE, OBSERVATION_STORAGE),
        (_A_INFLOW, OBSERVATION_INFLOW_TO_DATE),
        (_A_MONTH, OBSERVATION_MONTH),
        (_A_NEAR_TERM, OBSERVATION_NEAR_TERM),
        (_A_HYDRO, OBSERVATION_HYDRO_CLASS),
        (_A_MIN_FLOW, OBSERVATION_MIN_FLOW),
        (_A_FLOOD, OBSERVATION_FLOOD_LIMIT),
        (_A_RELEASE, OBSERVATION_RELEASE_STRUCTURE),
        (_A_RELEASE2, OBSERVATION_RELEASE_STRUCTURE),
        (_A_CARRYOVER, INSTRUCTIONS_CARRYOVER),
        (_A_SUPP_REM, OBSERVATION_REMAINING_SUPPLEMENTAL),
        (_A_SUPP_HEADER, INSTRUCTIONS_REMAINING_SUPPLEMENTAL_HEADER),
        (_A_NEXT_WY, OBSERVATION_NEXT_WY_COMMITTED),
        (_A_FORECAST_HEADER, OBSERVATION_FORECAST_COMPLEX),
    ]
    for anchor, template in checks:
        if anchor not in template:
            raise AssertionError(
                f"ablation complex anchor {anchor!r} no longer found in its prompts.py "
                f"template — update src/ablation.py complex patterns."
            )


_verify_complex_anchors()
