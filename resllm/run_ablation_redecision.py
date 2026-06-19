#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resllm.run_ablation_redecision  —  Option B: online prompt-ablation re-decision study.

Takes a *previous* simulation's recorded prompts (the ``observation`` column of its
``decision_output`` CSVs), removes one element from each, re-queries the model, and writes the
new decisions alongside the originals so the shift in the decision distribution can be
compared. The Ollama-friendly analog of the OpenAI Batch ablation pipeline
(``batch/src/create_ablation_batch_requests.py``): same idea, but it hits the Ollama / OpenAI
*synchronous* SDKs concurrently (Ollama has no batch-upload endpoint).

It is a re-DECISION on recorded observations, NOT a re-simulation — the reservoir is never
stepped. It reuses, rather than re-implements, the operator's call layer:
  * ``src.ablation``        — the shared, mode-aware text surgery (one source of truth),
  * ``build_decision_model`` / ``build_ollama_json_instruction`` — the decision schema (same
    levers as the source run, but concept-importance rankings dropped; see build_query_context),
  * ``_call_ollama`` / ``_call_openai`` / ``_with_retries`` — the synchronous provider calls.

Run from ``resllm/`` (``conda activate llm``). Example — re-decide every June (mowy 6) prompt
of a complex DeepSeek run with the minimum-flow line removed, using a fast local model::

    python run_ablation_redecision.py \
        --input-dir output \
        --model-prefix deepseek-v4-pro-cloud_r-high_obj-balance-multiple_complex \
        --ablation-type min_flow --month 6 \
        --model-server Ollama --model gpt-oss:20b --reasoning-effort high \
        --max-workers 6

Output goes to ``output/ablation/`` (a dedicated subdirectory — never a live run's filename).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import pandas as pd

# Make ``from src...`` importable regardless of the caller's working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))            # resllm/ (this file's dir)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv  # noqa: E402

from src.ablation import ABLATION_TYPES, apply_ablation  # noqa: E402
from src.model_config import RunIntent, resolve_model_config  # noqa: E402
from src.operator import (  # noqa: E402
    ReservoirAllocationOperator,
    _call_ollama,
    _call_openai,
    _with_retries,
    build_decision_model,
)
from src.prompts import build_ollama_json_instruction, get_concept_keys  # noqa: E402

load_dotenv()

_norm_pct = ReservoirAllocationOperator._normalize_allocation_percent


# =============================================================================
# Loading recorded observations
# =============================================================================

# Columns whose presence marks a complex-mode decision_output (used to auto-detect the mode).
_COMPLEX_MARKER_COLS = ("hydro_class", "min_flow_obs", "junior_allocation_percent", "wf_factor")


def _infer_complexity(df: pd.DataFrame) -> bool:
    return any(c in df.columns for c in _COMPLEX_MARKER_COLS)


def load_observations(
    input_dir: str, model_prefix: str, months: list[int] | None
) -> tuple[list[dict], bool]:
    """Load recorded prompts for ``model_prefix`` across its ``n0``–``n9`` files.

    Returns ``(rows, complexity_mode)`` where each row carries the recorded ``observation``
    (the full rendered prompt), the source decision, and identifying metadata. ``months=None``
    keeps every month.
    """
    files = sorted(glob(os.path.join(input_dir, f"{model_prefix}_decision_output_n[0-9].csv")))
    if not files:
        raise ValueError(
            f"No decision_output files matching '{model_prefix}_decision_output_n[0-9].csv' "
            f"in {input_dir}"
        )

    rows: list[dict] = []
    complexity_mode = False
    for path in files:
        df = pd.read_csv(path)
        if "observation" not in df.columns:
            raise ValueError(f"{os.path.basename(path)} has no 'observation' column")
        complexity_mode = complexity_mode or _infer_complexity(df)
        sample_match = re.search(r"_n(\d+)\.csv$", os.path.basename(path))
        sample_num = int(sample_match.group(1)) if sample_match else 0
        if months is not None:
            df = df[df["mowy"].isin(months)]
        for _, r in df.iterrows():
            if not isinstance(r.get("observation"), str):
                continue  # skip rows without a recorded prompt
            rows.append({
                "source_file": os.path.basename(path),
                "sample_num": sample_num,
                "date": r.get("date"),
                "wy": int(r["wy"]) if not pd.isna(r.get("wy")) else None,
                "mowy": int(r["mowy"]) if not pd.isna(r.get("mowy")) else None,
                "observation": r["observation"],
                "orig_allocation_percent": r.get("allocation_percent"),
                "orig_junior_delivery_percent": r.get("junior_allocation_percent"),
            })
    rows.sort(key=lambda x: (x["sample_num"], str(x["date"])))
    return rows, complexity_mode


# =============================================================================
# Re-decision
# =============================================================================

def _split_observation(observation: str) -> tuple[str, str]:
    """Split a recorded full prompt into (system_content, user_observation).

    The recorded blob is ``system_message + instructions + observation`` (from
    operator._record_decision); the user section begins at the ``OBSERVATION_MONTH`` anchor.
    """
    from src.ablation import split_system_and_user
    return split_system_and_user(observation)


def build_query_context(complexity_mode: bool):
    """Pre-build the decision schema and Ollama instruction for the re-decision.

    Simple mode collapses to the single static :class:`AllocationDecision`.
    """
    # Every re-decision here is an ablation, so concept-importance rankings are dropped from
    # both the schema and the Ollama instruction (consistent with the live ablation path and
    # the batch pipeline — never ask the model to rank a concept that may be removed).
    concept_keys = get_concept_keys(complexity_mode)
    return {
        "concept_keys": concept_keys,
        "model": build_decision_model(
            concept_keys, complexity_mode, include_concept_importance=False),
        "ollama_instruction": build_ollama_json_instruction(
            concept_keys, complexity_mode, include_concept_importance=False),
    }


def redecide_one(
    row: dict,
    ablation_type: str,
    *,
    complexity_mode: bool,
    ctx: dict,
    model_server: str,
    model: str,
    model_kwargs: dict,
    openai_client,
) -> dict:
    """Re-decide a single recorded prompt with one element ablated.

    Returns a flat result dict (new decision + original + metadata + any error). Exceptions are
    captured into the ``error`` field rather than raised, so one bad row never aborts the run.
    """
    observation = row["observation"]
    custom_id = (
        f"abl_{ablation_type}_n{row['sample_num']}"
        f"_date{str(row['date']).replace('-', '')}"
    )
    out: dict = {
        "custom_id": custom_id,
        "source_file": row["source_file"],
        "sample_number": row["sample_num"],
        "date": row["date"],
        "water_year": row["wy"],
        "month_of_water_year": row["mowy"],
        "ablation_type": ablation_type,
        "requery_model": f"{model_server}:{model}",
        "original_allocation_percent": row["orig_allocation_percent"],
        "allocation_percent": None,
        "allocation_reasoning": None,
        "error": None,
    }
    if complexity_mode:
        out["original_junior_delivery_percent"] = row["orig_junior_delivery_percent"]

    try:
        decision_model = ctx["model"]
        ollama_instruction = ctx["ollama_instruction"]

        system_content, user_content = _split_observation(observation)
        system_out, user_out = apply_ablation(
            system_content, user_content, ablation_type, complexity_mode=complexity_mode
        )

        if model_server == "OpenAI":
            decision, reasoning = _with_retries(
                lambda: _call_openai(
                    openai_client, model, system_out, user_out,
                    temperature=model_kwargs.get("temperature"),
                    reasoning=model_kwargs.get("reasoning"),
                    decision_model=decision_model,
                ),
                label="OpenAI",
            )
        else:
            decision, reasoning = _with_retries(
                lambda: _call_ollama(
                    model, system_out, user_out, model_kwargs,
                    ollama_instruction=ollama_instruction,
                    decision_model=decision_model,
                    concept_keys=ctx["concept_keys"],
                ),
                label="Ollama",
            )

        out["allocation_percent"] = _norm_pct(decision.allocation_percent)
        out["allocation_reasoning"] = decision.allocation_reasoning
        if complexity_mode:
            jd = getattr(decision, "junior_delivery_percent", None)
            out["junior_delivery_percent"] = _norm_pct(jd) if jd is not None else None
            out["carryover_target_taf"] = getattr(decision, "carryover_target_taf", None)
        # Copy any concept-importance rankings (normally empty — dropped by the reduced schema).
        aci = getattr(decision, "allocation_concept_importance", None)
        if aci:
            for k, v in aci.items():
                out[k] = v
    except Exception as e:  # noqa: BLE001 — capture per-row, keep the run alive
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# =============================================================================
# Driver
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-dir", default="output",
                   help="Directory holding the baseline decision_output CSVs (default: output/). "
                        "Read-only; never written to.")
    p.add_argument("--model-prefix", required=True,
                   help="Filename prefix of the baseline run, e.g. "
                        "'deepseek-v4-pro-cloud_r-high_obj-balance-multiple_complex' "
                        "(selects <prefix>_decision_output_n[0-9].csv).")
    p.add_argument("--ablation-type", required=True,
                   help="One ablation type, a comma-separated list, or 'all'. "
                        f"Choices: {', '.join(ABLATION_TYPES)}.")
    p.add_argument("--month", default="all",
                   help="Month of the water year to re-decide (1-12), a comma-separated list, "
                        "or 'all' (default).")
    p.add_argument("--model-server", default="Ollama", choices=["Ollama", "OpenAI"],
                   help="Provider for the re-query (default: Ollama).")
    p.add_argument("--model", required=True, help="Model id for the re-query.")
    p.add_argument("--reasoning-effort", default="high",
                   help="Reasoning effort for the re-query (default: high).")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--complexity", default=None, action=argparse.BooleanOptionalAction,
                   help="Force complex/simple schema. Default: auto-detect from the source columns.")
    p.add_argument("--max-workers", type=int, default=6,
                   help="Concurrent requests (default: 6).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of rows per ablation type (smoke testing).")
    p.add_argument("--output", default=None,
                   help="Output CSV path. Default: output/ablation/<prefix>_redecision_abl-"
                        "<types>_month-<months>.csv. Never written under a live run's filename.")
    return p.parse_args()


def _resolve_list(value: str, valid: tuple[str, ...], label: str) -> list[str]:
    if value == "all":
        return list(valid)
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in items if v not in valid]
    if bad:
        raise ValueError(f"Unknown {label}: {bad}. Choose from: {', '.join(valid)}")
    return items


def main():
    args = parse_args()

    ablation_types = _resolve_list(args.ablation_type, ABLATION_TYPES, "ablation-type")
    months = None
    if args.month != "all":
        months = [int(m) for m in str(args.month).split(",")]

    input_dir = os.path.abspath(args.input_dir)
    rows, detected_complex = load_observations(input_dir, args.model_prefix, months)
    complexity_mode = detected_complex if args.complexity is None else bool(args.complexity)

    print("─" * 64)
    print(f"  Source run:     {args.model_prefix}")
    print(f"  Input dir:      {input_dir}")
    print(f"  Rows loaded:    {len(rows)}  (months: {args.month})")
    print(f"  Mode:           {'COMPLEX' if complexity_mode else 'SIMPLE'}"
          f" ({'detected' if args.complexity is None else 'forced'})")
    print(f"  Ablations:      {', '.join(ablation_types)}")
    print(f"  Re-query model: {args.model_server}:{args.model} (effort={args.reasoning_effort})")
    print(f"  Workers:        {args.max_workers}")
    print("─" * 64)

    resolved = resolve_model_config(
        RunIntent(
            model_server=args.model_server,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
        )
    )
    for w in resolved.warnings:
        print(f"⚠  {w}")
    model_kwargs = resolved.model_kwargs

    openai_client = None
    if args.model_server == "OpenAI":
        from openai import OpenAI
        openai_client = OpenAI(api_key=model_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY"))

    ctx = build_query_context(complexity_mode)

    # Build the work list (one task per row x ablation type), applying the optional per-type cap.
    tasks: list[tuple[dict, str]] = []
    for ablation_type in ablation_types:
        selected = rows[: args.limit] if args.limit else rows
        tasks.extend((row, ablation_type) for row in selected)

    print(f"  Total re-decisions: {len(tasks)}\n")

    results: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(
                redecide_one, row, ablation_type,
                complexity_mode=complexity_mode, ctx=ctx,
                model_server=args.model_server, model=resolved.model,
                model_kwargs=model_kwargs, openai_client=openai_client,
            ): (row, ablation_type)
            for row, ablation_type in tasks
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                n_err = sum(1 for r in results if r["error"])
                print(f"  [{done}/{len(tasks)}] complete  ({n_err} errors)")

    df = pd.DataFrame(results)

    # Output path — a dedicated subdir, never a live run's filename.
    out_dir = os.path.join(input_dir, "ablation")
    if args.output:
        out_path = os.path.abspath(args.output)
        out_dir = os.path.dirname(out_path)
    else:
        abl_tag = "all" if set(ablation_types) == set(ABLATION_TYPES) else "-".join(ablation_types)
        month_tag = "all" if months is None else "-".join(str(m) for m in months)
        out_path = os.path.join(
            out_dir, f"{args.model_prefix}_redecision_abl-{abl_tag}_month-{month_tag}.csv"
        )
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nWrote {len(df)} re-decisions -> {out_path}")
    _summary(df, complexity_mode)


def _summary(df: pd.DataFrame, complexity_mode: bool) -> None:
    ok = df[df["error"].isna()]
    print("\n== Allocation shift by ablation (mean over re-decided prompts) ==")
    print(f"{'ablation':>20} {'n':>4} {'orig%':>7} {'new%':>7} {'Δ%':>7}")
    for abl, g in ok.groupby("ablation_type"):
        o = pd.to_numeric(g["original_allocation_percent"], errors="coerce").mean()
        nnew = pd.to_numeric(g["allocation_percent"], errors="coerce").mean()
        print(f"{abl:>20} {len(g):>4} {o:>7.1f} {nnew:>7.1f} {nnew - o:>7.1f}")
    n_err = int(df["error"].notna().sum())
    if n_err:
        print(f"\n⚠  {n_err} rows errored (see the 'error' column).")


if __name__ == "__main__":
    main()
