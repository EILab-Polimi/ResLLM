#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resllm.prompts

Centralized prompt templates and message strings for the reservoir operator agent.
"""
from textwrap import dedent


# =============================================================================
# SYSTEM MESSAGE TEMPLATES
# =============================================================================


SYSTEM_MESSAGE_BASE = dedent("""\
    You are a water reservoir operator.
    Your goal is to minimize shortages to downstream water supply by releasing water from the reservoir.
    The reservoir is located in a region with a Mediterranean climate, characterized by hot, dry summers and highly variable wet winters.
    The reservoir is operated to meet the municipal and agricultural water supply needs of the region while also maintaining flood control and environmental flow requirements.
    The water year is defined as the period from October through September.
    """)


# =============================================================================
# INSTRUCTION TEMPLATES
# =============================================================================

INSTRUCTIONS_BASE = dedent("""\
    - You are tasked with determining the percent allocation of water demand to release from the reservoir.
    - At the beginning of each month, you will be asked to update the percent allocation decision based on your current observations.
    - In your determination, consider the volume currently in storage, inflow to date compared to expected inflows, and the need to balance meeting current demands against conserving water for future demands.
    - Note that a shortage is calculated by demand x (100 - percent allocation).
    You have the following information about the reservoir:
    - The maximum operable storage level is {operable_storage_max} TAF.
    - The minimum operable storage level is {operable_storage_min} TAF.
    """)

INSTRUCTIONS_AVERAGE_DEMAND = "- The average total water year demand: {average_water_year_total_demand}\n"

INSTRUCTIONS_CUMULATIVE_INFLOW_HEADER = "- The average cumulative inflow by beginning of month of the water year: "
INSTRUCTIONS_CUMULATIVE_INFLOW_MONTH = "Month {month}: {value} TAF | "

INSTRUCTIONS_REMAINING_DEMAND_HEADER = "- The average remaining demand by beginning of month of the water year: "
INSTRUCTIONS_REMAINING_DEMAND_MONTH = "Month {month}: {value} TAF | "

INSTRUCTIONS_FORECAST = dedent("""\
    - You have access to a probabilistic forecast of inflows for the remainder of the water year.
    - The probabilistic forecast includes the ensemble mean, and 10th and 90th percentile expected water year inflow.
    - Use this forecast to inform your allocation decision.
    """)

INSTRUCTIONS_RED_HERRING = dedent("""\
    - Puppies like to play, explore their surroundings with boundless curiosity, and chew on just about everything they can get their teeth on. They also love to sleep deeply after their bursts of energy, often curling up in the coziest spots they can find.
    """)

INSTRUCTIONS_IMPORTANCE_RANKING = dedent("""\
    - Assign an importance ranking ("very high"=1, "high"=2, "medium"=3, "low"=4, or "no importance"=0) to the reservoir management concepts supporting your decision.
    """)


# =============================================================================
# OBSERVATION TEMPLATES
# =============================================================================

OBSERVATION_MONTH = dedent("""\
    It is the beginning of month {mowy} of the water year.
    """)

OBSERVATION_INFLOW_TO_DATE = dedent("""\
    So far this water year, {qwyaccum} TAF of reservoir inflow has been observed.
    """)

OBSERVATION_STORAGE = dedent("""\
    There is currently {storage} TAF in storage.
    """)

OBSERVATION_FORECAST = dedent("""\
    The probabilistic forecasted inflows for the remainder of the water year are:
    - Mean (expected): {forecast_mean} TAF
    - 10th percentile: {forecast_10} TAF
    - 90th percentile: {forecast_90} TAF
    """)

OBSERVATION_REMAINING_DEMAND = dedent("""\
    There is approximately {d_wy_rem} TAF of water demand to meet over the remainder of the water year.
    """)

OBSERVATION_NEXT_WY_DEMAND = dedent("""\
    Also, note that next water year is approaching and the first three months have a demand of {next_wy_demand} TAF.
    """)

OBSERVATION_ALLOCATION_DECISION = dedent("""\
    The previous percent allocation decision was {alloc_1} percent.
    Provide a percent allocation decision (from 0-100 percent) which continues or updates the allocation.
    """)


# =============================================================================
# SHARED CONSTANTS
# =============================================================================

# Canonical concept key list — single source of truth used by the fuzzy
# matcher in operator.py, the OperationalConcepts TypedDict, and the
# Ollama/freeform JSON instruction prompt below.
CONCEPT_KEYS: tuple[str, ...] = (
    "environment_setting", "goal", "operational_limits",
    "average_cumulative_inflow_by_month", "average_remaining_demand_by_month",
    "previous_allocation", "current_month", "current_storage",
    "current_cumulative_observed_inflow", "current_water_year_remaining_demand",
    "next_water_year_demand", "mean_forecast", "percentile_forecast_10th",
    "percentile_forecast_90th", "puppies",
)


# =============================================================================
# OLLAMA NATIVE PROMPTS
# =============================================================================

OLLAMA_JSON_INSTRUCTION = (
    "\nRespond with valid JSON for the AllocationDecision schema."
    "\nUse exact keys: allocation_reasoning, allocation_percent, allocation_concept_importance."
    "\nThe allocation_concept_importance object MUST include these exact keys and the values MUST be integers: "
    + ", ".join(CONCEPT_KEYS) + "."
)

MAGISTRAL_OLLAMA_SYSTEM_PREPEND = dedent("""\
    First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.

    Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response. Use the same language as the input.[/THINK]Here, provide a self-contained response.
    """)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_system_message() -> str:
    """Build the system message."""
    return SYSTEM_MESSAGE_BASE


def build_instructions(reservoir, include_red_herring: bool = False) -> str:
    """Build the full instruction string based on reservoir characteristics."""
    instructions = INSTRUCTIONS_BASE.format(
        operable_storage_max=reservoir.characteristics["operable_storage_max"],
        operable_storage_min=reservoir.characteristics["operable_storage_min"],
    )

    instructions += INSTRUCTIONS_AVERAGE_DEMAND.format(
        average_water_year_total_demand=reservoir.characteristics["average_water_year_total_demand"]
    )

    instructions += INSTRUCTIONS_CUMULATIVE_INFLOW_HEADER
    for month in range(12):
        instructions += INSTRUCTIONS_CUMULATIVE_INFLOW_MONTH.format(
            month=month + 1,
            value=reservoir.characteristics["average_cumulative_inflow_by_month"][month]
        )
    instructions += "\n"

    instructions += INSTRUCTIONS_REMAINING_DEMAND_HEADER
    for month in range(12):
        instructions += INSTRUCTIONS_REMAINING_DEMAND_MONTH.format(
            month=month + 1,
            value=reservoir.characteristics["average_remaining_demand_by_month"][month]
        )
    instructions += "\n"

    if reservoir.characteristics["wy_forecast_file"] is not False:
        instructions += INSTRUCTIONS_FORECAST

    if include_red_herring:
        instructions += INSTRUCTIONS_RED_HERRING

    instructions += INSTRUCTIONS_IMPORTANCE_RANKING

    return instructions


def build_observation(
    mowy: int,
    st_1: float,
    d_wy_rem: float,
    alloc_1: float,
    qwyaccum: float = None,
    qwy_forecast_mean: float = None,
    qwy_forecast_10: float = None,
    qwy_forecast_90: float = None,
    next_wy_demand: float = None,
) -> str:
    """Build the observation string for a given timestep."""
    observation = OBSERVATION_MONTH.format(mowy=int(mowy) if mowy is not None else 0)

    # Add cumulative inflow if past month 1
    if mowy > 1:
        observation += OBSERVATION_INFLOW_TO_DATE.format(
            qwyaccum=int(qwyaccum) if qwyaccum is not None else 0
        )

    # Add current storage
    observation += OBSERVATION_STORAGE.format(
        storage=int(st_1) if st_1 is not None else 0
    )

    # Add forecast if available
    if qwy_forecast_mean is not None:
        observation += OBSERVATION_FORECAST.format(
            forecast_mean=int(qwy_forecast_mean),
            forecast_10=int(qwy_forecast_10) if qwy_forecast_10 is not None else 0,
            forecast_90=int(qwy_forecast_90) if qwy_forecast_90 is not None else 0,
        )

    # Add remaining demand
    observation += OBSERVATION_REMAINING_DEMAND.format(
        d_wy_rem=int(d_wy_rem) if d_wy_rem is not None else 0
    )

    # Add next water year demand warning if approaching end of year
    if mowy >= 9 and next_wy_demand is not None:
        observation += OBSERVATION_NEXT_WY_DEMAND.format(
            next_wy_demand=int(next_wy_demand)
        )

    # Add allocation decision request
    observation += OBSERVATION_ALLOCATION_DECISION.format(
        alloc_1=int(alloc_1) if alloc_1 is not None else 0
    )

    return observation
