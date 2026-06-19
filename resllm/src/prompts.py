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


OBJECTIVE_MINIMIZE_SHORTAGES = (
    "Your goal is to minimize shortages to downstream water supply by releasing water from the reservoir."
)

OBJECTIVE_MINIMIZE_LARGE_SHORTAGES_CARRYOVER = (
    "Your goal is to minimize large shortages to downstream water supply and avoid low end of year carryover storage."
)

# Complex-mode objective: balance delivering water supply now against conserving carryover.
OBJECTIVE_BALANCE_DELIVERY = (
    "Your goal is to meet downstream water-supply and environmental-flow demands while maintaining "
    "required flood-control capacity and managing carryover storage as a buffer across years. "
    "Deliver water supply as fully as conditions allow, and conserve storage as carryover when low "
    "storage or dry forecasts put later supply reliability at risk — including across multiyear "
    "droughts — without stranding water that could be delivered."
)

OBJECTIVES: dict[str, str] = {
    "minimize-shortages": OBJECTIVE_MINIMIZE_SHORTAGES,
    "minimize-large-shortages-carryover": OBJECTIVE_MINIMIZE_LARGE_SHORTAGES_CARRYOVER,
    "balance-delivery": OBJECTIVE_BALANCE_DELIVERY,
}

_SYSTEM_MESSAGE_TEMPLATE = dedent("""\
    You are a water reservoir operator.
    {objective}
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

# Complex-mode base framing: the decision sets two curtailable demands (junior committed
# supply, supplemental downstream flow) above a firm floor served automatically. Replaces
# INSTRUCTIONS_BASE / INSTRUCTIONS_AVERAGE_DEMAND when complexity_mode is True.
INSTRUCTIONS_BASE_COMPLEX = dedent("""\
    - You operate the reservoir's monthly releases. Your two routine supply choices are: (1) what percent of the junior committed water-supply deliveries to make, and (2) what percent of the supplemental downstream-flow demand to deliver — both downstream demands met from storage, so delivering less than full is a shortage you weigh against the storage it conserves. The senior committed deliveries and a firm floor of the minimum downstream flow are served automatically. You also decide, only in exceptional conditions, whether to curtail the minimum flow below its full required level (meet_min_flow), and you set a carryover-storage target that paces your drawdown — both described below.
    - At the beginning of each month, you update these choices based on your current observations.
    - In your determination, consider the volume currently in storage, inflow to date compared to expected inflows, and the trade-off between delivering water supply now and conserving it as carryover for later in the year and for future years. Conserve more when low storage or dry forecasts put later supply at risk, and deliver more fully when conditions are ample.
    You have the following information about the reservoir:
    - The maximum operable storage level is {operable_storage_max} TAF.
    - The minimum operable storage level is {operable_storage_min} TAF.
    """)

INSTRUCTIONS_SUPPLY_SCALE_COMPLEX = (
    "- Committed water-supply deliveries average about {committed} TAF/yr (the senior portion "
    "firm; the junior portion your choice). Above the required minimum instream flow and these "
    "deliveries is the supplemental downstream-flow demand you allocate — about {supplemental} "
    "TAF/yr in a typical year, but it scales with the water-year hydrology, rising in wet years "
    "(more water to move downstream) and falling in dry years. You decide what percent of each "
    "month's supplemental demand to deliver; delivering less is a shortage you weigh against the "
    "storage it conserves.\n"
)

INSTRUCTIONS_CUMULATIVE_INFLOW_HEADER = "- The average cumulative inflow by beginning of month of the water year: "
INSTRUCTIONS_CUMULATIVE_INFLOW_MONTH = "Month {month}: {value} TAF | "

INSTRUCTIONS_REMAINING_DEMAND_HEADER = "- The average remaining demand by beginning of month of the water year: "
INSTRUCTIONS_REMAINING_DEMAND_MONTH = "Month {month}: {value} TAF | "

# Complex-mode analog: remaining supplemental downstream-flow demand by month.
INSTRUCTIONS_REMAINING_SUPPLEMENTAL_HEADER = "- The average remaining supplemental downstream-flow demand by beginning of month of the water year (a typical, normal-year reference — the demand scales up in wetter years and down in drier ones; the actual amount for the current month is given in the monthly state): "

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

# Complex-mode operating context. Gated on complexity_mode; never emitted in simple mode.
INSTRUCTIONS_COMPLEX = dedent("""\
    - Above the required minimum flow (described below) come the committed water-supply deliveries, drawn from storage. The senior portion is firm and served automatically whenever water is available. The junior portion is your decision: you set what percent of it to deliver each month, and delivering less than full is a water-supply shortage — weighed against the storage it conserves. Drier years generally warrant lower junior deliveries, but the choice is yours.
    - Above the minimum flow and the committed deliveries is the supplemental downstream-flow demand: additional release, beyond the minimum, needed to support downstream water management. Its size tracks the water-year hydrology — larger in wet years and smaller in dry ones. You set what percent of it to deliver. It is a genuine demand, so delivering less than full is a shortage — but, like the junior deliveries, it is a demand you may curtail to conserve storage. Releasing it meets downstream needs now and draws storage down; withholding it is a genuine shortage and is only worthwhile when forecasts or low storage make conservation necessary for reliability.
    - The minimum flow, the senior deliveries, the junior deliveries, and the supplemental downstream flow all draw on the same storage, in that order of priority. If you draw storage down too far, the lowest-priority demands are cut first — the supplemental flow, then the junior deliveries, then the senior deliveries, and in the extreme even the minimum environmental flow. Restraint on the demands you control protects these higher-priority needs in genuinely dry conditions; in normal and wet conditions the priority stack is already protected by storage and inflow, so curtailing lower-priority demands is unnecessary.
    - The minimum downstream environmental flow is a regulatory requirement with the highest priority. Each month you decide whether to deliver it in full (meet_min_flow = true) or to curtail it down to a firm regulatory floor that is always released (meet_min_flow = false). Curtailing it shorts a regulatory environmental requirement — a more serious failure than any water-supply shortage — so you should deliver the full minimum flow in all but the most extreme conditions, choosing to curtail ONLY as a last resort, when storage is so depleted that meeting it would compromise even higher-priority needs or leave the reservoir unable to provide basic flows going forward. This ordering is strict: never curtail the minimum flow while you are still delivering any lower-priority water. Before you may set meet_min_flow = false you must already have cut both lower-priority supply demands to zero this month (allocation_percent = 0 and junior_delivery_percent = 0) — curtailing the minimum flow while still delivering supplemental or junior water shorts a higher-priority requirement to serve lower-priority ones, which is never correct. Only if storage is so depleted that even with both already at zero the firm floor itself is at risk should you then choose meet_min_flow = false; reaching for it should be rare and exceptional. The full minimum flow already satisfies the required downstream instream needs and the supply diversions that lie within it; the supplemental flow and the committed deliveries you provide are demands above this baseline. Its level rises and falls continuously with the hydrologic condition of the water year and the month; once set in late autumn it is also held up through the spawning season to avoid stranding downstream habitat, so winter and spring floors are linked to earlier ones.
    - A flood-control storage limit (the top of the conservation pool) is enforced automatically: storage above it is released for dam safety. This limit is forecast-based — it tightens (reserving more empty space and forcing releases) when near-term inflow is forecast to be high, and relaxes back toward full capacity as the remaining wet-season inflow is forecast to decline. You do not set this limit, but your allocation interacts with it.
    """)

# Carryover instruction: a target >= 0 makes the operation defend it with an automatic
# release cap on the two supply deliveries; -1 leaves it unenforced.
INSTRUCTIONS_CARRYOVER = dedent("""\
    - The carryover target paces how fast you draw storage down: committing a target (any value of 0 or greater) tells the operation to cap your junior committed and supplemental deliveries so storage stays at or above it through the rest of the operating window. It throttles only those two supply deliveries — it never reduces the minimum flow (which you set separately with meet_min_flow), the senior committed deliveries, or flood-control releases, and it can never hold storage above the flood-control limit. You decide the target yourself from the conditions — there is no suggested level. Raise the target to conserve supply for later in the year or across a multiyear drought; lower it, or set it to -1 (no target), to deliver supply more fully now. You commit a carryover storage target each month.
    """)

INSTRUCTIONS_FORECAST_COMPLEX = dedent("""\
    - You also have a near-term (next few months) inflow outlook in addition to the remaining-water-year forecast. The water-year forecast is reported as the ensemble mean together with the 10th, 25th, 50th (median), 75th, and 90th percentiles of expected inflow.
    """)


# =============================================================================
# OBSERVATION TEMPLATES
# =============================================================================

# Month line: states both the water-year month and the calendar month to disambiguate the
# two numbering schemes. The "It is the beginning of month ... of the water year." text is
# the anchor for ablation._SPLIT_ANCHOR and the current_month ablation (matches
# "of the water year."), so it must stay intact.
OBSERVATION_MONTH = dedent("""\
    It is the beginning of month {mowy} of the water year. This is the calendar month of {month_name}; the water year runs October through September, so water-year month 1 is October and month 12 is September.
    """)

# Water-year month index (mowy 1..12) -> calendar month name (water year starts in October).
_WY_MONTH_NAMES: tuple[str, ...] = (
    "October", "November", "December", "January", "February", "March",
    "April", "May", "June", "July", "August", "September",
)

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

# Complex-mode analogs: remaining quantities framed as supplemental downstream-flow
# demand and committed obligations, not a single "demand" figure.
OBSERVATION_REMAINING_SUPPLEMENTAL = dedent("""\
    Approximately {supplemental_rem} TAF of supplemental downstream-flow demand remains over the rest of the water year (above the required minimum flow and committed deliveries).
    """)

OBSERVATION_NEXT_WY_DEMAND = dedent("""\
    Also, note that next water year is approaching and the first three months have a demand of {next_wy_demand} TAF.
    """)

OBSERVATION_NEXT_WY_COMMITTED = dedent("""\
    Also, note that next water year is approaching: its first three months carry approximately {next_wy_committed} TAF of committed deliveries (plus the required minimum flow).
    """)

OBSERVATION_ALLOCATION_DECISION = dedent("""\
    The previous percent allocation decision was {alloc_1} percent.
    Provide a percent allocation decision (from 0-100 percent) which continues or updates the allocation.
    """)

# --- Complex-mode observation lines (generic; gated on complexity_mode) ---

OBSERVATION_FORECAST_COMPLEX = dedent("""\
    The probabilistic forecasted inflows for the remainder of the water year are:
    - Mean (expected): {forecast_mean} TAF
    - 10th percentile (dry): {forecast_10} TAF
    - 25th percentile: {forecast_25} TAF
    - 50th percentile (median): {forecast_median} TAF
    - 75th percentile: {forecast_75} TAF
    - 90th percentile (wet): {forecast_90} TAF
    """)

OBSERVATION_NEAR_TERM = dedent("""\
    The near-term inflow outlook for the next few months is approximately {near_term} TAF.
    """)

OBSERVATION_HYDRO_CLASS = dedent("""\
    The projected full-water-year inflow is approximately {wy_index} TAF, classifying this as a {hydro_class} year.
    """)

OBSERVATION_MIN_FLOW = dedent("""\
    The required minimum downstream flow this month is about {min_flow_month} TAF ({min_flow} TAF per day). Of this, about {min_flow_floor} TAF per day is a firm regulatory floor that is always released; the remainder is the discretionary portion you decide whether to deliver (meet_min_flow). Released ahead of all supply, the full minimum flow already meets the required downstream instream needs and the supply diversions within it; the supplemental flow and committed deliveries you provide are demands above this baseline.
    """)

OBSERVATION_FLOOD_LIMIT = dedent("""\
    The current flood-control storage limit is {tocs} TAF; storage above this is released automatically.
    """)

OBSERVATION_RELEASE_STRUCTURE = dedent("""\
    Above the minimum flow, about {senior_mi} TAF of senior committed deliveries are served automatically this month. The junior committed deliveries are about {junior_mi_full} TAF at full delivery — you set the percent to deliver; hydrologic conditions support roughly {wf_pct} percent, and full delivery is appropriate unless storage or forecasts indicate otherwise. Above these, this month's supplemental downstream-flow demand is about {supplemental} TAF — its level reflects the water year's hydrology (higher in wetter years, lower in drier). You set the percent of it to deliver; delivering less than full is a real shortage to downstream support, worthwhile only when conserving storage for later reliability is genuinely warranted.
    """)

OBSERVATION_ALLOCATION_DECISION_COMPLEX = dedent("""\
    The previous supplemental-flow allocation was {alloc_1} percent.
    Provide your decision:
    - allocation_percent: the percent (0-100) of the supplemental downstream-flow demand to deliver this month (default 100; below 100 is a real shortage to downstream support, justified only when conservation is genuinely needed).
    - junior_delivery_percent: the percent (0-100) of the junior committed deliveries to make this month (default 100; below 100 is a committed water-supply shortage, justified only in dry or low-storage conditions).
    - meet_min_flow: whether to deliver the full required minimum downstream flow this month (true) or curtail it to the firm regulatory floor (false). Curtailing is a regulatory violation — choose false only as a last resort in the most extreme low-storage conditions, and only after you have already set allocation_percent = 0 and junior_delivery_percent = 0 (never curtail the minimum flow while still delivering lower-priority supplemental or junior water).
    - carryover_target_taf: the end-of-season storage you commit to leave in the reservoir, in TAF. It caps your junior committed and supplemental deliveries to keep storage at or above the target — it never reduces the minimum flow (set separately by meet_min_flow), the senior deliveries, or flood-control releases, and never holds storage above the flood-control limit. Raise it to conserve supply for later in the year or a multiyear drought; lower it, or use -1 (no target), to deliver supply more fully now.
    """)


# =============================================================================
# SHARED CONSTANTS
# =============================================================================

# Canonical concept key lists — single source of truth for the operator.py fuzzy
# matcher, the decision-model factories, and the Ollama JSON instruction builder below.
CONCEPT_KEYS_SIMPLE: tuple[str, ...] = (
    "environment_setting", "goal", "operational_limits",
    "average_cumulative_inflow_by_month", "average_remaining_demand_by_month",
    "previous_allocation", "current_month", "current_storage",
    "current_cumulative_observed_inflow", "current_water_year_remaining_demand",
    "next_water_year_demand", "mean_forecast", "percentile_forecast_10th",
    "percentile_forecast_90th", "puppies",
)

# Additional concepts surfaced only in complex mode (floor + committed supply +
# supplemental downstream-flow demand).
CONCEPT_KEYS_COMPLEX_EXTRA: tuple[str, ...] = (
    "hydrologic_class", "minimum_environmental_flow", "flood_control_curve",
    "carryover_storage_target", "committed_water_supply",
    "supplemental_flow_demand",
)

CONCEPT_KEYS_COMPLEX: tuple[str, ...] = CONCEPT_KEYS_SIMPLE + CONCEPT_KEYS_COMPLEX_EXTRA

# Backward-compatible alias (simple baseline + batch pipeline import this name).
CONCEPT_KEYS: tuple[str, ...] = CONCEPT_KEYS_SIMPLE


def get_concept_keys(complexity_mode: bool = False) -> tuple[str, ...]:
    """Return the active concept-key tuple for the requested mode."""
    return CONCEPT_KEYS_COMPLEX if complexity_mode else CONCEPT_KEYS_SIMPLE


# =============================================================================
# OLLAMA NATIVE PROMPTS
# =============================================================================

def build_ollama_json_instruction(
    concept_keys: tuple[str, ...] = CONCEPT_KEYS_SIMPLE,
    complexity_mode: bool = False,
    include_concept_importance: bool = True,
) -> str:
    """Build the Ollama JSON-mode instruction for the active schema.

    ``include_concept_importance`` controls whether the ``allocation_concept_importance``
    rankings are requested; ablation runs disable it so the model is not asked to rank
    concepts that may have been removed from the prompt (mirrors the batch pipeline).
    """
    fields = "allocation_reasoning, allocation_percent"
    if complexity_mode:
        fields += ", junior_delivery_percent, meet_min_flow, carryover_target_taf"
    if include_concept_importance:
        fields += ", allocation_concept_importance"

    instruction = (
        "\nRespond with valid JSON for the AllocationDecision schema."
        f"\nUse exact keys: {fields}."
    )
    if complexity_mode:
        instruction += (
            "\njunior_delivery_percent MUST be a number from 0 to 100."
        )
        instruction += "\nmeet_min_flow MUST be a boolean (true or false)."
        instruction += (
            "\ncarryover_target_taf MUST be a number in TAF (use -1 for no target)."
        )
    if include_concept_importance:
        instruction += (
            "\nThe allocation_concept_importance object MUST include these exact keys "
            "and the values MUST be integers: " + ", ".join(concept_keys) + "."
        )
    return instruction


# Backward-compatible module-level constant (simple-mode default).
OLLAMA_JSON_INSTRUCTION = build_ollama_json_instruction(CONCEPT_KEYS_SIMPLE, False)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_system_message(objective: str = "minimize-shortages") -> str:
    """Build the system message.

    Args:
        objective: Key from OBJECTIVES dict controlling the goal sentence.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective '{objective}'. Choose from: {list(OBJECTIVES)}")
    return _SYSTEM_MESSAGE_TEMPLATE.format(objective=OBJECTIVES[objective])


def build_instructions(
    reservoir,
    include_red_herring: bool = False,
    include_importance_ranking: bool = True,
    complexity_mode: bool = False,
) -> str:
    """Build the full instruction string from reservoir characteristics.

    Args:
        reservoir: Reservoir (or shim) exposing a ``characteristics`` dict.
        include_red_herring: Append the puppies ablation text.
        include_importance_ranking: Append the concept importance-ranking
            instruction. Disable for the batch ablation pipeline, which uses
            a reduced schema without ``allocation_concept_importance``.
        complexity_mode: Append the complex operating-context instructions
            (demand tiers, min flow, forecast-based flood curve, carryover
            target) and the richer forecast description. When False the output
            is byte-identical to the simple baseline.
    """
    # Base framing + supply-scale context. Complex mode frames the decision as two
    # curtailable demands (junior committed supply + supplemental downstream flow) above a
    # firm floor and reports the supply-stack scale; simple mode keeps the legacy demand
    # framing byte-identical.
    if complexity_mode:
        instructions = INSTRUCTIONS_BASE_COMPLEX.format(
            operable_storage_max=reservoir.characteristics["operable_storage_max"],
            operable_storage_min=reservoir.characteristics["operable_storage_min"],
        )
        instructions += INSTRUCTIONS_SUPPLY_SCALE_COMPLEX.format(
            committed=int(round(reservoir.committed_total())),
            supplemental=int(round(reservoir.delta_demand_total())),
        )
    else:
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

    if complexity_mode:
        instructions += INSTRUCTIONS_REMAINING_SUPPLEMENTAL_HEADER
        for month in range(12):
            instructions += INSTRUCTIONS_REMAINING_DEMAND_MONTH.format(
                month=month + 1, value=int(round(reservoir.delta_demand_remaining(month + 1)))
            )
        instructions += "\n"
    else:
        instructions += INSTRUCTIONS_REMAINING_DEMAND_HEADER
        for month in range(12):
            instructions += INSTRUCTIONS_REMAINING_DEMAND_MONTH.format(
                month=month + 1,
                value=reservoir.characteristics["average_remaining_demand_by_month"][month]
            )
        instructions += "\n"

    if reservoir.characteristics["wy_forecast_file"] is not False:
        instructions += INSTRUCTIONS_FORECAST

    if complexity_mode:
        instructions += INSTRUCTIONS_COMPLEX
        instructions += INSTRUCTIONS_CARRYOVER
        if reservoir.characteristics["wy_forecast_file"] is not False:
            instructions += INSTRUCTIONS_FORECAST_COMPLEX

    if include_red_herring:
        instructions += INSTRUCTIONS_RED_HERRING

    if include_importance_ranking:
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
    complexity_mode: bool = False,
    qwy_forecast_median: float = None,
    qwy_forecast_25: float = None,
    qwy_forecast_75: float = None,
    near_term: float = None,
    hydro_class: str = None,
    wy_index: float = None,
    min_flow: float = None,
    min_flow_month: float = None,
    min_flow_floor: float = None,
    tocs: float = None,
    senior_mi: float = None,
    junior_mi_full: float = None,
    supplemental: float = None,
    wf_factor: float = None,
    supplemental_rem: float = None,
    next_wy_committed: float = None,
) -> str:
    """Build the observation string for one timestep.

    With ``complexity_mode=False`` the output is byte-identical to the simple
    baseline and the complex kwargs are ignored. With ``complexity_mode=True`` an
    enriched forecast block (mean + 10/25/50/75/90 percentiles), a near-term
    outlook line, and operating-context lines (hydrologic class, min flow, flood
    limit, release structure) are emitted, and the decision request asks for the
    structured multi-part decision.
    """
    _m = int(mowy) if mowy is not None else 0
    observation = OBSERVATION_MONTH.format(
        mowy=_m,
        month_name=_WY_MONTH_NAMES[(_m - 1) % 12] if _m else "the current month",
    )

    # Add cumulative inflow if past month 1
    if mowy > 1:
        observation += OBSERVATION_INFLOW_TO_DATE.format(
            qwyaccum=int(qwyaccum) if qwyaccum is not None else 0
        )

    # Add current storage
    observation += OBSERVATION_STORAGE.format(
        storage=int(st_1) if st_1 is not None else 0
    )

    # Add forecast if available (enriched in complex mode)
    if complexity_mode and qwy_forecast_mean is not None:
        observation += OBSERVATION_FORECAST_COMPLEX.format(
            forecast_mean=int(qwy_forecast_mean),
            forecast_10=int(qwy_forecast_10) if qwy_forecast_10 is not None else 0,
            forecast_25=int(qwy_forecast_25) if qwy_forecast_25 is not None else 0,
            forecast_median=int(qwy_forecast_median) if qwy_forecast_median is not None else 0,
            forecast_75=int(qwy_forecast_75) if qwy_forecast_75 is not None else 0,
            forecast_90=int(qwy_forecast_90) if qwy_forecast_90 is not None else 0,
        )
        if near_term is not None:
            observation += OBSERVATION_NEAR_TERM.format(near_term=int(near_term))
    elif qwy_forecast_mean is not None:
        observation += OBSERVATION_FORECAST.format(
            forecast_mean=int(qwy_forecast_mean),
            forecast_10=int(qwy_forecast_10) if qwy_forecast_10 is not None else 0,
            forecast_90=int(qwy_forecast_90) if qwy_forecast_90 is not None else 0,
        )

    # Complex operating-context lines
    if complexity_mode:
        if wy_index is not None and hydro_class is not None:
            observation += OBSERVATION_HYDRO_CLASS.format(
                wy_index=int(wy_index), hydro_class=hydro_class
            )
        if min_flow is not None:
            observation += OBSERVATION_MIN_FLOW.format(
                min_flow=round(float(min_flow), 1),
                min_flow_month=int(round(float(min_flow_month))) if min_flow_month is not None
                else int(round(float(min_flow) * 30)),
                min_flow_floor=round(float(min_flow_floor), 1) if min_flow_floor is not None
                else round(float(min_flow) * 0.5, 1),
            )
        if tocs is not None:
            observation += OBSERVATION_FLOOD_LIMIT.format(tocs=int(tocs))
        if senior_mi is not None and supplemental is not None:
            observation += OBSERVATION_RELEASE_STRUCTURE.format(
                senior_mi=round(float(senior_mi), 1),
                junior_mi_full=round(float(junior_mi_full), 1) if junior_mi_full is not None else 0,
                wf_pct=int(round(float(wf_factor) * 100)) if wf_factor is not None else 100,
                supplemental=round(float(supplemental), 1),
            )

    # Remaining quantity: complex shows remaining supplemental demand; simple shows
    # remaining demand.
    if complexity_mode and supplemental_rem is not None:
        observation += OBSERVATION_REMAINING_SUPPLEMENTAL.format(supplemental_rem=int(supplemental_rem))
    else:
        observation += OBSERVATION_REMAINING_DEMAND.format(
            d_wy_rem=int(d_wy_rem) if d_wy_rem is not None else 0
        )

    # Next-water-year warning near end of year (committed obligations in complex mode,
    # demand in simple mode).
    if mowy >= 9:
        if complexity_mode and next_wy_committed is not None:
            observation += OBSERVATION_NEXT_WY_COMMITTED.format(
                next_wy_committed=int(next_wy_committed)
            )
        elif not complexity_mode and next_wy_demand is not None:
            observation += OBSERVATION_NEXT_WY_DEMAND.format(
                next_wy_demand=int(next_wy_demand)
            )

    # Add allocation decision request
    if complexity_mode:
        observation += OBSERVATION_ALLOCATION_DECISION_COMPLEX.format(
            alloc_1=int(alloc_1) if alloc_1 is not None else 0
        )
    else:
        observation += OBSERVATION_ALLOCATION_DECISION.format(
            alloc_1=int(alloc_1) if alloc_1 is not None else 0
        )

    return observation
