#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resllm.reservoir

Reservoir simulation classes for the resllm project.
"""

import os

import numpy as np
import pandas as pd
import src.utils as utils


class Reservoir:
    """Basic reservoir simulation."""

    # Day count per water year month (Oct–Sep, no leap day)
    _WY_MONTH_DAYS = (31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30)

    def __init__(
        self,
        characteristics: dict | None = None,
    ):
        """Initialize the reservoir simulation.

        Parameters:
            characteristics (dict): Reservoir characteristics, including:
                - inflow_file (str): Path to the inflow data file.
                - demand_file (str): Path to the demand data file.
                - operable_storage_max (float): Maximum operable storage in TAF.
                - operable_storage_min (float): Minimum operable storage in TAF.
                - max_safe_release (float): Maximum safe release in TAF.
                - sp_to_rp (list): Storage-to-release points.
                - sp_to_ep (list): Storage-to-elevation points.
                - tp_to_tocs (list): Top-of-conservation-storage points.
        """

        if characteristics is None:
            characteristics = {}

        self.tocs = characteristics["tocs"]
        self.demand = np.loadtxt(characteristics["demand_file"]) # demand, TAF
        print(f"Demand data loaded: {characteristics['demand_file']}")
        self.inflows = pd.read_csv(characteristics["inflow_file"])  # inflow, TAF
        if characteristics["wy_forecast_file"] is not False:
            self.forecasted_inflows = pd.read_csv(characteristics["wy_forecast_file"])  # forecasted inflows, TAF
            self.forecasted_inflows["date"] = pd.to_datetime(self.forecasted_inflows["date"])
            print(f"Forecasted inflows data loaded: {characteristics['wy_forecast_file']}")

        # optional near-term monthly inflow outlook (drives the dynamic flood curve)
        self.monthly_forecast = None
        if characteristics.get("wy_monthly_forecast_file"):
            self.monthly_forecast = pd.read_csv(characteristics["wy_monthly_forecast_file"])
            self.monthly_forecast["date"] = pd.to_datetime(self.monthly_forecast["date"])
            print(f"Monthly forecast data loaded: {characteristics['wy_monthly_forecast_file']}")

        # derive date metadata on the inflow data
        self.inflows["date"] = pd.to_datetime(self.inflows["date"])
        self.inflows["year"] = self.inflows["date"].dt.year
        self.inflows["month"] = self.inflows["date"].dt.month
        self.inflows["day"] = self.inflows["date"].dt.day
        self.inflows["doy"] = self.inflows["date"].dt.dayofyear
        self.inflows["dowy"] = self.inflows.apply(
            lambda row: utils.water_day(row["doy"]) + 1, axis=1
        )
        self.inflows["week"] = self.inflows["dowy"].apply(
            lambda x: int((x - 1) / 7) + 1
        )
        self.inflows.loc[
            (self.inflows["month"] == 10) & (self.inflows["day"] == 1), "week"
        ] = 1
        self.inflows["water_year"] = np.where(
            self.inflows["month"] >= 10,
            self.inflows["year"] + 1,
            self.inflows["year"],
        )
        self.inflows["date"] = self.inflows["date"].dt.strftime("%Y-%m-%d")
        print(f"Inflow data loaded: {characteristics['inflow_file']}")

        self.characteristics = characteristics
        self.characteristics["average_water_year_total_demand"] = int(self.demand.sum())
        self.characteristics["average_remaining_demand_by_month"] = (
            self.compute_average_remaining_demand_by_month()
        )
        self.characteristics["average_cumulative_inflow_by_month"] = (
            self.compute_average_cumulative_inflow_by_month()
        )

        # historical mean full-WY inflow; fallback for wy_inflow_index when no
        # forecast file is supplied
        self.average_wy_total_inflow = float(
            self.inflows.groupby("water_year")["inflow"].sum().mean()
        )

        # complexity-mode operating context (optional; default off = simple baseline)
        self.complexity = characteristics.get("complexity")
        self.complexity_mode = bool(characteristics.get("complexity_mode", False))
        if self.complexity_mode:
            if self.complexity is None:
                raise ValueError(
                    "complexity_mode is True but no 'complexity' config block was provided"
                )
            # CalSim-grounded demand stack: additive layers all served by the release
            # ``rt`` (= observed total storage outflow). A supplemental downstream-flow
            # demand (FLOW-ADD-INSTREAM, above the minimum) the LLM governs, plus committed
            # upstream M&I split senior (water-rights, firm) / junior (CVP+MFP, LLM-governed).
            # Lower-American M&I is informational (diverted below the dam, not a storage draw).
            # All arrays are Oct→Sep.
            #
            # ``supplemental_schedule``: per mowy ``[[ARI breakpoints], [TAF/month]]``,
            # interpolated on the continuous ARI. ``_delta_demand_monthly`` is the
            # representative (normal-year) array used for static context and any ARI-less path.
            sched = self.complexity["supplemental_schedule"]
            self._supplemental_schedule = [
                (np.asarray(bp, dtype=float), np.asarray(vals, dtype=float)) for bp, vals in sched
            ]
            if len(self._supplemental_schedule) != 12:
                raise ValueError(
                    "complexity supplemental_schedule must have 12 monthly [bps, vals] entries (Oct→Sep)"
                )
            # Representative ARI = the median breakpoint (normal year); interpolating there
            # gives the normal-year value shown as static context. Midpoint is robust to the
            # number of breakpoints.
            _bps = self._supplemental_schedule[0][0]
            self._supplemental_repr_ari = float(_bps[len(_bps) // 2])
            self._delta_demand_monthly = [
                float(np.interp(self._supplemental_repr_ari, bp, vals))
                for bp, vals in self._supplemental_schedule
            ]
            # Delta-support floor: a per-month minimum supplemental demand (TAF) that does NOT
            # relax with a falling ARI. In dry/critical years the observed Folsom release
            # carries a persistent late-summer downstream/Delta obligation above the ARI-relaxed
            # demand (a falling index pulls np.interp toward the too-low critical column). The
            # floor binds only when the interpolated demand drops below it (low ARI / late
            # summer); 0 elsewhere.
            self._supplemental_floor_monthly = list(
                self.complexity.get("supplemental_floor_monthly_taf", [0.0] * 12)
            )
            if len(self._supplemental_floor_monthly) != 12:
                raise ValueError(
                    "complexity supplemental_floor_monthly_taf must have 12 monthly values (Oct→Sep)"
                )
            self._delta_demand_monthly = [
                max(d, f) for d, f in zip(self._delta_demand_monthly, self._supplemental_floor_monthly)
            ]

            # Low-storage relaxation of the supplemental demand (config
            # `supplemental_low_storage`): in the listed months the demand scales linearly with
            # storage, full at/above `full_above_taf` to zero at/below `zero_below_taf`. Without
            # the block the supplemental demand is storage-independent.
            _ls = self.complexity.get("supplemental_low_storage")
            if _ls:
                self._supp_ls_months = {int(m) for m in _ls.get("months_mowy", [])}
                self._supp_ls_full = float(_ls.get("full_above_taf", 0.0))
                self._supp_ls_zero = float(_ls.get("zero_below_taf", 0.0))
            else:
                self._supp_ls_months = set()
                self._supp_ls_full = self._supp_ls_zero = 0.0

            # Minimum-flow standard (Lower American River Flow Management Standard, ARWA-103):
            # storage-driven Oct–Feb (Four Reservoir Index + Jan/Feb storage triggers),
            # inflow-driven Mar–Sep (Impaired Folsom Inflow Index). Complex mode requires an
            # enabled `fms` block. The month-to-month chaining and held FRI are month state,
            # refreshed once per month by :meth:`update_monthly_min_flow`. The historical
            # Sacramento River Index (``sri_file``) drives the Jan/Feb tiers.
            self._fms = self.complexity.get("fms")
            self._fms_enabled = bool(self._fms and self._fms.get("enabled", False))
            if not self._fms_enabled:
                raise ValueError("complex mode requires an enabled 'fms' block")
            self._fri_oct1_storage = None
            self._prev_mfr_cfs = None
            self._feb_mfr_cfs = None
            self._month_mfr_cfs = None
            self._sri_lookup = {}
            if self._fms_enabled and self._fms.get("sri_file"):
                _data_dir = os.path.dirname(os.path.abspath(characteristics["inflow_file"]))
                _sri_path = os.path.join(_data_dir, self._fms["sri_file"])
                if os.path.exists(_sri_path):
                    _sri = pd.read_csv(_sri_path)
                    self._sri_lookup = {
                        (int(r.wy), int(r.mowy)): float(r.sri_taf)
                        for r in _sri.itertuples(index=False)
                    }
                    print(f"SRI data loaded: {_sri_path}")
                else:
                    print(f"⚠ FMS sri_file not found ({_sri_path}); using Folsom-index SRI proxy")

            umi = self.complexity["upstream_mi"]
            self._upstream_mi_monthly = list(umi["monthly_taf"])
            self._upstream_mi_senior_frac = float(umi["senior_frac"])
            self._upstream_mi_wf_cvp_frac = float(
                umi.get("wf_cvp_frac", 1.0 - float(umi["senior_frac"]))
            )
            self._lower_mi_monthly = list(
                self.complexity.get("lower_mi_monthly_taf", [0.0] * 12)
            )
            for name, arr in (
                ("upstream_mi.monthly_taf", self._upstream_mi_monthly),
                ("lower_mi_monthly_taf", self._lower_mi_monthly),
            ):
                if len(arr) != 12:
                    raise ValueError(
                        f"complexity {name} must have 12 monthly values (Oct→Sep)"
                    )

        self.record = pd.DataFrame()

    def record_timestep(
        self,
        idx: int = 0,
        date: pd.Timestamp = None,
        wy: int = None,
        mowy: int = None,
        dowy: int = None,
        qt: float = None,
        st: float = None,
        rt: float = None,
        dt: float = None,
        uu: float = None,
        min_flow: float | None = None,
        tocs: float | None = None,
        hydro_class: str | None = None,
        delta_demand: float | None = None,
        delta_delivered: float | None = None,
        delta_short: float | None = None,
        committed_mi: float | None = None,
        junior_delivery_pct: float | None = None,
        junior_short: float | None = None,
        near_term: float | None = None,
        release_cap: float | None = None,
        flood_release: float | None = None,
        wf_factor: float | None = None,
        ari: float | None = None,
        min_flow_short: float | None = None,
        evap: float | None = None,
    ):
        """Record the simulation output for one time step.

        The trailing complex-mode columns are written only when supplied; leaving them
        ``None`` keeps the simple-mode output byte-identical. Column meanings:

        - ``min_flow``: regulatory minimum instream flow (the forced floor).
        - ``delta_demand``: supplemental downstream-flow demand offered this step;
          ``delta_delivered`` the portion the agent released; ``delta_short`` the shortfall
          ``max(0, delta_demand − delta_delivered)`` (including the agent's own curtailment).
        - ``committed_mi``: delivered committed upstream M&I (senior + agent-set junior);
          ``junior_delivery_pct`` the agent's junior delivery percent; ``junior_short`` the
          junior M&I shortfall against the full commitment.
        - ``near_term``: held near-term inflow outlook driving the winter ``tocs``.
        - ``ari``: continuous American River Index driving min flow, carryover, and the
          Water-Forum cutback; ``wf_factor`` the hydrology-suggested junior delivery fraction
          shown to the agent.
        - ``release_cap``: carryover-defense cap (``inf`` when inactive).
        - ``flood_release``: the portion of ``rt`` the flood curve (dynamic-TOCS ramped
          evacuation) or gross-pool capacity forced ABOVE the demand target ``uu``
          (``max(rt − uu, 0)``; the only mechanisms that lift ``rt`` above ``uu``). Logs the
          model's own dynamic-curve flood evacuation, distinct from the policy-neutral
          regulatory spill characterized against the fixed WCD curve in the analysis layer.
        - ``min_flow_short``: regulatory minimum-flow shortfall ``max(0, min_flow − rt)``
          (TAF/day) — non-zero only when the reservoir is too depleted to release the required
          instream flow (a regulatory violation, distinct from a water-supply shortage).
        - ``evap``: evaporative loss applied to the mass balance this step (TAF/day; complex
          mode), recorded for transparency.
        """
        self.record.loc[idx, "date"] = date
        self.record.loc[idx, "wy"] = wy
        self.record.loc[idx, "mowy"] = mowy
        self.record.loc[idx, "dowy"] = dowy
        self.record.loc[idx, "qt"] = qt
        self.record.loc[idx, "st"] = st
        self.record.loc[idx, "rt"] = rt
        self.record.loc[idx, "dt"] = dt
        self.record.loc[idx, "uu"] = uu
        if min_flow is not None:
            self.record.loc[idx, "min_flow"] = min_flow
        if tocs is not None:
            self.record.loc[idx, "tocs"] = tocs
        if hydro_class is not None:
            self.record.loc[idx, "hydro_class"] = hydro_class
        if delta_demand is not None:
            self.record.loc[idx, "delta_demand"] = delta_demand
        if delta_delivered is not None:
            self.record.loc[idx, "delta_delivered"] = delta_delivered
        if delta_short is not None:
            self.record.loc[idx, "delta_short"] = delta_short
        if committed_mi is not None:
            self.record.loc[idx, "committed_mi"] = committed_mi
        if junior_delivery_pct is not None:
            self.record.loc[idx, "junior_delivery_pct"] = junior_delivery_pct
        if junior_short is not None:
            self.record.loc[idx, "junior_short"] = junior_short
        if near_term is not None:
            self.record.loc[idx, "near_term"] = near_term
        if release_cap is not None:
            self.record.loc[idx, "release_cap"] = release_cap
        if flood_release is not None:
            self.record.loc[idx, "flood_release"] = flood_release
        if wf_factor is not None:
            self.record.loc[idx, "wf_factor"] = wf_factor
        if ari is not None:
            self.record.loc[idx, "ari"] = ari
        if min_flow_short is not None:
            self.record.loc[idx, "min_flow_short"] = min_flow_short
        if evap is not None:
            self.record.loc[idx, "evap"] = evap

    def evaluate(
        self,
        st_1,
        qt,
        uu,
        tocs,
        *,
        min_flow: float = 0.0,
        release_cap: float = float("inf"),
        senior_floor: float = 0.0,
        cap_protect: float | None = None,
        evaporation: float = 0.0,
    ):
        """Evaluate the release and ending storage given state, inflow, target, and TOCS.

        Every downstream demand (minimum flow, committed M&I, supplemental downstream flow)
        is served by the single release ``rt`` (= observed total storage outflow), so the
        mass balance is ``S(t) = S(t-1) + Q − rt − E`` (``E`` the evaporative loss, zero
        unless supplied) with no separate off-channel draw. The caller composes the demand
        stack into the target ``uu`` and attributes the realized ``rt`` back across the
        priority layers.

        Parameters:
            st_1 (float): Storage at the end of the previous time step.
            qt (float): Inflow during the current time step.
            uu (float): Desired (target) release.
            tocs (float): Top of conservation storage (TAF).
            min_flow (float): Hard minimum downstream-flow floor (TAF/day). Complex mode.
            release_cap (float): Upper release cap defending a carryover target (TAF/day);
                never applied below the flood-forced release, the hard floor, or
                ``cap_protect`` — the flood curve always overrules the cap. Complex mode.
            senior_floor (float): Firm obligation that must be met if water is available
                (TAF/day) — in complex mode the firm minimum-flow floor plus senior committed
                M&I. Complex mode.
            cap_protect (float | None): Release level the carryover cap must not reduce below
                (TAF/day) — the minimum flow at the agent's chosen delivery level
                (``meet_min_flow``) plus senior committed M&I plus the agent's chosen junior
                M&I delivery. Ensures a carryover target throttles ONLY the discretionary
                supplemental/delta layer: the agent's delivery decision overrides the target,
                which is breached rather than clawing back committed supply when even zero
                supplemental cannot hold it. Defaults to the hard floor when None. Complex mode.
            evaporation (float): Evaporative loss from storage this step (TAF/day), applied to
                the mass balance after the release (``S = S_1 + Q − rt − E``) and to the
                available-water and spill constraints. Default 0.0 reproduces the
                no-evaporation result exactly. Complex mode.
        Returns:
            list: A list containing:
                - rt (float): The actual release from the reservoir.
                - st (float): The storage in the reservoir at the end of the current time step.
        Notes:
            - The target release is raised to evacuate flood water per the TOCS constraint.
            - With the default neutral arguments (``min_flow=0``, ``release_cap=inf``,
              ``senior_floor=0``) this reproduces the simple-mode result exactly.
            - Physical limits (max safe release, available water) can still pull the release
              below the hard floor — a rare hard-shortage event, by design.
        """
        K = self.characteristics["operable_storage_max"]
        demand_target = uu
        hard_floor = max(min_flow, senior_floor)
        # Forced flood release (never negative). Ramped evacuation: pass the full inflow and
        # draw down a fraction (1 - 0.8) of the EXISTING above-curve excess each day, so the
        # excess decays geometrically (excess_t = 0.8 * excess_{t-1}) back to TOCS without a
        # standing offset — releasing a fraction of (inflow + excess) instead would under-pass
        # inflow and leave storage perched above the curve. When st_1 > tocs this reduces to
        # qt + 0.2*(st_1 - tocs). Bounded below by the max-safe-release cap downstream, so a
        # flood exceeding channel capacity backs up and is worked off over the next days.
        flood_release = max((qt + st_1 - tocs) - 0.8 * max(st_1 - tocs, 0.0), 0.0)
        # the carryover cap may throttle only the junior + supplemental deliveries; the flood
        # curve always overrules it, so it floors the cap alongside the protected obligations
        # (the chosen min flow + senior, cap_protect, and the hard floor)
        protect = hard_floor if cap_protect is None else max(hard_floor, cap_protect)
        cap_floor = max(protect, flood_release)

        # floor at the forced flood release
        rt = max(flood_release, demand_target)
        # enforce the hard minimum-flow / firm obligation floor
        rt = max(rt, hard_floor)
        # apply the carryover cap, but never below cap_floor (chosen min flow + senior,
        # hard floor, or flood release)
        rt = max(min(rt, release_cap), cap_floor)
        # cap at max safe release
        rt = min(rt, utils.cfs_to_taf(self.compute_max_release(st_1)))
        # cap at available water (net of this step's evaporation)
        rt = min(rt, max(st_1 + qt - evaporation, 0.0))
        # add any spill (storage above capacity after release and evaporation)
        rt += max(st_1 + qt - evaporation - rt - K, 0)
        # ending storage (net of evaporation)
        st = st_1 + qt - rt - evaporation

        return [rt, st]

    def compute_max_release(self, S):
        """Maximum release from the storage-to-release curve at storage ``S``.

        Parameters:
            S (float): Reservoir storage.
        Returns:
            float: Maximum release.
        """
        sp = self.characteristics["sp_to_rp"][0]
        rp = self.characteristics["sp_to_rp"][1]
        return np.interp(S, sp, rp)

    def compute_tocs(self, dowy, date=None, near_term=None):
        """Top of Conservation Storage (TOCS, TAF) for the given day or date.

        Parameters:
            dowy (int): Day of the water year (1-365/366).
            date (str): Date in 'YYYY-MM-DD' format (used by the ``historical`` mode).
            near_term (float): Forecasted near-term inflow outlook (TAF), driving the winter
                regime of the ``"dynamic"`` curve. Complex mode.
        Returns:
            float: Top of Conservation Storage (TAF).
        """
        tp = self.characteristics["tp_to_tocs"][0]
        tocs = self.characteristics["tp_to_tocs"][1]
        tocs = np.interp(dowy, tp, tocs)
        if self.tocs == "historical":
            hist_st = self.inflows.loc[
                (self.inflows["date"] == date), "storage"
            ].values[0]
            return np.max([tocs, hist_st])
        elif self.tocs == "fixed":
            return tocs
        elif self.tocs == "dynamic":
            return self._compute_dynamic_tocs(dowy, float(tocs), near_term)
        else:
            return self.characteristics["operable_storage_max"]

    def _compute_dynamic_tocs(self, dowy, base_tocs, near_term):
        """Forecast-based two-regime flood-control curve (CalSim abstraction).

        Modeled in flood-space terms (``flood_space = K - tocs``), reusing the fixed
        ``tp_to_tocs`` curve as the date-based baseline:

        - **Winter (flood season):** reserve the baseline space, increased when the near-term
          inflow outlook is wet (storm-risk encroachment; mirrors CalSim's ``OctMarRunoffEst``
          threshold). More forecast inflow -> lower TOCS -> more forced evacuation.
        - **Rest of year (fall drawdown / spring refill):** TOCS follows the static WCD rule
          curve directly (``base_tocs``). Per the 2017 WCM Update, no forecast-based flood
          space applies outside the Nov 19–Feb 28 variable-reserve window.

        Returns the TOCS (TAF), clipped to ``[0, K]``.
        """
        K = self.characteristics["operable_storage_max"]
        cfg = self.complexity["dynamic_tocs"]
        base_space = max(0.0, K - base_tocs)
        mowy = self._dowy_to_mowy(dowy)
        lo_m, hi_m = cfg["flood_season_mowy"]

        if lo_m <= mowy <= hi_m:
            nt = near_term if near_term is not None else 0.0
            lo, hi = cfg["winter_nearterm_lo"], cfg["winter_nearterm_hi"]
            frac = float(np.clip((nt - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
            flood_space = base_space + cfg["max_winter_encroachment_taf"] * frac
            tocs = K - flood_space
            floor_curve = cfg.get("aggressive_floor_curve")
            if floor_curve is not None:
                tocs = max(tocs, float(np.interp(dowy, floor_curve[0], floor_curve[1])))
        else:
            # outside the flood season: static WCD rule curve
            tocs = float(base_tocs)

        return float(np.clip(tocs, 0.0, K))

    def _dowy_to_mowy(self, dowy):
        """Map day-of-water-year (1-based, Oct 1 = 1) to month-of-water-year (1-12)."""
        cum = 0
        for i, days in enumerate(self._WY_MONTH_DAYS):
            cum += days
            if dowy <= cum:
                return i + 1
        return 12

    def volume_to_height(self, S):
        """Convert storage (TAF) to height (feet) via the storage-elevation curve.

        Parameters:
            S (float): Storage in TAF.
        Returns:
            float: Height in feet.
        """
        sp = self.characteristics["sp_to_ep"][0]
        ep = self.characteristics["sp_to_ep"][1]
        return np.interp(S, sp, ep)

    def compute_average_cumulative_inflow_by_month(self):
        """Average cumulative inflow at the start of each water-year month.

        Returns:
            np.ndarray: Average cumulative inflow indexed by month of the water year.
        """
        monthly_inflow = (
            self.inflows[["water_year", "month", "inflow"]]
            .groupby(["water_year", "month"], as_index=False)
            .sum()
        )

        # accumulate average monthly inflows in water-year order (Oct–Sep)
        wy_month_order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        cumulative_inflow_by_month = np.zeros(12)
        for i in range(11):
            cumulative_inflow_for_month = (
                monthly_inflow.loc[
                    (monthly_inflow["month"] == wy_month_order[i]),
                    "inflow",
                ]
                .mean()
                .astype(int)
            )
            cumulative_inflow_by_month[i + 1] = (
                cumulative_inflow_by_month[i] + cumulative_inflow_for_month
            )

        return cumulative_inflow_by_month

    def compute_average_remaining_demand_by_month(self):
        """Average demand remaining at the start of each water-year month.

        Returns:
            np.ndarray: Remaining demand indexed by month of the water year.
        """
        total_demand = self.demand.sum()
        remaining_demand_by_month = np.zeros(12)
        remaining_demand_by_month[0] = int(total_demand)
        cum_days = 0
        for i, days in enumerate(self._WY_MONTH_DAYS[:11]):
            total_demand -= self.demand[cum_days : cum_days + days].sum()
            remaining_demand_by_month[i + 1] = int(total_demand)
            cum_days += days

        return remaining_demand_by_month

    # --------------------------------------------------------------------- #
    # Complex-mode operating context (gated on complexity_mode)
    # --------------------------------------------------------------------- #

    def wy_inflow_index(self, date, wy, dowy) -> float:
        """Self-contained full-water-year inflow index (TAF).

        ``cumulative_observed_inflow_to_date + remaining-WY forecast (QCYFHM)`` — leaks no
        future information. Degrades to the historical mean full-WY total when no forecast
        file is loaded.
        """
        qwyaccum = 0
        if dowy > 0:
            qwyaccum = int(
                self.inflows.loc[self.inflows["water_year"] == wy, "inflow"]
                .values[0 : (dowy - 1)]
                .sum()
            )
        if self.characteristics.get("wy_forecast_file") not in (None, False):
            row = self.forecasted_inflows.loc[self.forecasted_inflows["date"] == date]
            if not row.empty:
                return float(qwyaccum + float(row["QCYFHM"].values[0]))
        return float(self.average_wy_total_inflow)

    def classify_water_year(self, index: float) -> str:
        """Coarse ``dry``/``normal``/``wet`` label from the inflow index.

        A legibility hint shown to the agent only. The physics (supplemental demand,
        Water-Forum cutback) responds to the continuous American River Index, not these
        discrete classes.
        """
        thr = self.complexity["wy_index_thresholds"]
        if index <= thr["dry_max"]:
            return "dry"
        if index >= thr["wet_min"]:
            return "wet"
        return "normal"

    def american_river_index(self, wy_index: float, spill_vol: float = 0.0) -> float:
        """American River Index (TAF): forecast WY inflow less cumulative spills.

        Mirrors CalSim ``ARI = max(Amer_B120 - Fol_spill_vol, 0)``. ``wy_index`` is the
        self-contained forecast WY inflow (:meth:`wy_inflow_index`); ``spill_vol`` is the
        cumulative volume of releases above the spill threshold since October.
        """
        return max(float(wy_index) - float(spill_vol), 0.0)

    def near_term_inflow(self, date) -> float:
        """Forecasted near-term inflow outlook (TAF) for the dynamic flood curve.

        Reads the committed monthly-forecast file (``nt3_mean``). Returns ``0.0`` when
        no monthly forecast is available or the date is missing.
        """
        if self.monthly_forecast is None:
            return 0.0
        row = self.monthly_forecast.loc[self.monthly_forecast["date"] == date]
        if row.empty:
            return 0.0
        return float(row["nt3_mean"].values[0])

    def compute_min_flow(self) -> float:
        """Hard minimum downstream-flow floor (TAF/day): the held FMS Minimum Flows
        Requirement (Lower American River Flow Management Standard).

        Refresh the stored value once per month with :meth:`update_monthly_min_flow` before
        reading it here; the result is the held MFR converted from cfs to TAF/day.
        """
        base = self._month_mfr_cfs if self._month_mfr_cfs is not None else 0.0
        return float(utils.cfs_to_taf(base))

    def update_monthly_min_flow(
        self, mowy: int, st_month_start_taf: float, wy_index: float, wy: int | None = None
    ) -> float:
        """Refresh the FMS minimum-flow state for the month (no-op when FMS is disabled).

        Computes and stores this month's Minimum Flows Requirement (cfs), maintaining the held
        Four-Reservoir-Index (set from the Oct 1 storage) and the previous-month MFR used by
        the chaining. Call once on the first day of each water-year month, before reading the
        min flow. ``wy`` selects the historical Sacramento River Index for the Jan/Feb tiers.
        Returns the stored MFR (cfs); 0.0 when FMS is disabled.
        """
        if not getattr(self, "_fms_enabled", False):
            return 0.0
        if mowy == 1:
            self._fri_oct1_storage = float(st_month_start_taf)
            self._prev_mfr_cfs = None
            self._feb_mfr_cfs = None
        self._month_mfr_cfs = self._compute_fms_mfr_cfs(
            mowy, wy_index=float(wy_index), wy=wy,
            prev_mfr_cfs=self._prev_mfr_cfs,
            prev_month_end_storage_taf=float(st_month_start_taf),
            current_storage_taf=float(st_month_start_taf),
        )
        self._prev_mfr_cfs = self._month_mfr_cfs
        if mowy == 5:                       # Feb: hold its MFR for the Mar–May cap
            self._feb_mfr_cfs = self._month_mfr_cfs
        return self._month_mfr_cfs

    def _compute_fms_mfr_cfs(
        self, mowy: int, *, wy_index: float, wy: int | None, prev_mfr_cfs: float | None,
        prev_month_end_storage_taf: float, current_storage_taf: float,
    ) -> float:
        """The FMS Minimum Flows Requirement (cfs) for the month (ARWA-103 §2/§6).

        Oct–Dec: Four Reservoir Index curve on a Folsom-approximated FRI. Jan–Feb: Sacramento
        River Index tiers (historical SRI, with a Folsom-index proxy fallback) plus end-of-
        month Folsom storage triggers. Mar–Sep: Impaired Folsom Inflow Index curve on a
        Folsom-approximated IFII. A year-round off-ramp relaxes the floor at very low storage.
        """
        f = self._fms

        def curve(x, c, below):
            bps, vals = c
            return float(below) if x < bps[0] else float(np.interp(x, bps, vals))

        if mowy in (1, 2, 3):                                  # Oct–Dec: FRI (storage) curve
            a, b = f["fri_from_folsom"]
            fri = a * float(self._fri_oct1_storage) + b
            mfr = curve(fri, f["fri_curve"], f["fri_below_min_cfs"])
            if mowy == 1:
                mfr = min(mfr, float(f["oct_cap_cfs"]))
        elif mowy in (4, 5):                                   # Jan–Feb: SRI tier + storage trigger
            prev = prev_mfr_cfs if prev_mfr_cfs is not None else float(f["critical_floor_cfs"])
            sri = self._sri_lookup.get((int(wy), mowy)) if wy is not None else None
            if sri is not None:
                is_critical = sri < float(f["sri_critical_taf"])
                is_wet = sri >= float(f["sri_wet_taf"])
            else:                                              # fallback: Folsom-index proxy
                lo, hi = f["sri_from_folsom_idx"]
                is_critical = wy_index < float(lo)
                is_wet = wy_index >= float(hi)
            thr = f["eod_dec_storage_taf"] if mowy == 4 else f["eod_jan_storage_taf"]
            if is_critical:                                    # critically dry (SRI rule)
                mfr = (float(f["critical_offramp_cfs"]) if prev <= 800
                       else max(float(f["critical_cut_frac"]) * prev, float(f["critical_floor_cfs"])))
            elif prev_month_end_storage_taf < float(thr):      # end-of-month storage trigger
                mfr = max(float(f["storage_trigger_frac"]) * prev, float(f["storage_trigger_floor_cfs"]))
            elif is_wet:                                       # above normal / wet
                mfr = float(f["janfeb_max_cfs"])
            else:                                              # dry / below normal: hold, capped
                mfr = min(prev, float(f["janfeb_max_cfs"]))
        else:                                                  # Mar–Sep: IFII (inflow) curve
            a, b = f["ifii_from_folsom_idx"]
            ifii = a * float(wy_index) + b
            mfr = curve(ifii, f["ifii_curve"], f["ifii_below_min_cfs"])
            if mowy in (6, 7, 8) and self._feb_mfr_cfs is not None:
                # Mar–May (FMS §6.4.1): MFR is the lesser of the IFII flow and the Feb MFR when
                # end-of-May storage is low. Approximated by always capping at the held Feb MFR
                # — inert in wet years (Feb ≈ IFII), pulls Mar–May down to Feb in dry years,
                # avoiding a storage forecast.
                mfr = min(mfr, float(self._feb_mfr_cfs))
            if mowy == 12:                                     # Sep: post-Labor-Day cap
                mfr = min(mfr, float(f["sep_cap_cfs"]))

        # Year-round off-ramp: very low Folsom storage relaxes the floor.
        if current_storage_taf < float(f["offramp_storage_taf"]):
            mfr = min(mfr, float(f["offramp_cfs_octdec"] if mowy in (1, 2, 3)
                                 else f["offramp_cfs_jansep"]))
        return float(mfr)

    def compute_water_forum_factor(self, wy_index: float) -> float:
        """Water-Forum delivery factor (0–1) for the junior upstream-M&I portion.

        Ports the ``UIFR_MarNov``-scaled Water-Forum cutbacks: full delivery in wet years,
        reduced in dry years per the ``water_forum_cutback`` table (``np.interp`` clamps the
        tails). Returns 1.0 when no cutback table is configured.
        """
        cutback = self.complexity.get("upstream_mi", {}).get("water_forum_cutback")
        if not cutback:
            return 1.0
        return float(np.interp(float(wy_index), cutback[0], cutback[1]))

    def supplemental_storage_factor(self, mowy: int, st: float | None) -> float:
        """Low-storage scaling (0–1) on the supplemental demand for the configured months.

        Linear in storage: 1.0 at/above ``full_above_taf``, 0.0 at/below ``zero_below_taf``,
        interpolated between — applied only in ``supplemental_low_storage.months_mowy`` (Nov–May).
        Returns 1.0 when storage is unknown, no relaxation is configured, or the month is
        outside the window (leaving summer and storage-independent behaviour unchanged).
        """
        if (st is None or mowy not in self._supp_ls_months
                or self._supp_ls_full <= self._supp_ls_zero):
            return 1.0
        span = self._supp_ls_full - self._supp_ls_zero
        return float(min(1.0, max(0.0, (float(st) - self._supp_ls_zero) / span)))

    def _supplemental_taf_month(self, mowy: int, ari: float | None = None,
                                st: float | None = None) -> float:
        """Supplemental downstream-flow demand (TAF) for the month, hydrology- and storage-scaled.

        The monthly FLOW-ADD-INSTREAM demand is interpolated from the ``supplemental_schedule``
        on the continuous ARI (``np.interp`` clamps the tails); ``ari=None`` uses the
        representative (normal-year) index shown as static context. The result is floored at
        the per-month Delta-support minimum (``supplemental_floor_monthly_taf``, 0 by default),
        which binds only when the interpolated demand falls below it (low ARI / late summer).
        Finally, in the low-storage relaxation months (Nov–May) the demand is scaled by
        :meth:`supplemental_storage_factor` when ``st`` is supplied (no effect when ``st`` is None).
        """
        bp, vals = self._supplemental_schedule[mowy - 1]
        a = self._supplemental_repr_ari if ari is None else float(ari)
        base = float(np.interp(a, bp, vals))
        val = max(base, float(self._supplemental_floor_monthly[mowy - 1]))
        return val * self.supplemental_storage_factor(mowy, st)

    def delta_demand_day(self, mowy: int, ari: float | None = None,
                         st: float | None = None) -> float:
        """Supplemental downstream-flow demand (TAF/day) for the month — the LLM's lever.

        The monthly :meth:`_supplemental_taf_month` demand spread evenly across the month's
        days. ``ari=None`` falls back to the representative (normal-year) demand. ``st``
        (current storage, TAF) applies the Nov–May low-storage relaxation; omit it for the
        storage-independent demand.
        """
        return self._supplemental_taf_month(mowy, ari, st=st) / self._WY_MONTH_DAYS[mowy - 1]

    def upstream_mi_day(self, mowy: int, dowy: int | None = None) -> float:
        """Committed upstream M&I delivery (TAF/day) for the month (senior+junior).

        Released from storage (part of ``rt``/the observed outflow), not a separate draw.
        """
        return float(self._upstream_mi_monthly[mowy - 1]) / self._WY_MONTH_DAYS[mowy - 1]

    def lower_mi_day(self, mowy: int, dowy: int | None = None) -> float:
        """Lower-American M&I (TAF/day) for the month — informational (diverted below the dam)."""
        return float(self._lower_mi_monthly[mowy - 1]) / self._WY_MONTH_DAYS[mowy - 1]

    def delta_demand_total(self, ari: float | None = None) -> float:
        """Annual supplemental downstream-flow demand the agent governs (TAF/yr).

        Hydrology-scaled on ``ari``; ``ari=None`` gives the representative (normal-year) total
        used for static system-message context.
        """
        return float(sum(self._supplemental_taf_month(mo, ari) for mo in range(1, 13)))

    def committed_total(self) -> float:
        """Annual committed upstream M&I delivery (TAF/yr; senior + junior).

        Lower-American M&I is diverted below the dam from the released flow, not counted here.
        """
        return float(sum(self._upstream_mi_monthly))

    def delta_demand_remaining(self, mowy: int, ari: float | None = None) -> float:
        """Supplemental downstream-flow demand still ahead this water year, this month→Sep (TAF).

        Hydrology-scaled on ``ari``; ``ari=None`` gives the representative (normal-year)
        remaining demand for static system-message context.
        """
        return float(sum(self._supplemental_taf_month(mo, ari) for mo in range(mowy, 13)))

    def committed_first_months(self, n: int = 3) -> float:
        """Committed M&I over the first ``n`` months of a water year (Oct…) — next-WY hint (TAF)."""
        return float(sum(self._upstream_mi_monthly[:n]) + sum(self._lower_mi_monthly[:n]))

    def expected_window_inflow(self, dowy: int, window_end_dowy: int) -> float:
        """Climatological expected inflow (TAF) from ``dowy`` to ``window_end_dowy`` inclusive.

        Mean daily inflow per day-of-water-year across the loaded record (a normal-year
        expectation, NOT this year's realized inflow — leak-free), summed over the window. Used
        by :meth:`compute_carryover_release_cap` to ration the drawdown NET of the inflow that
        will refill storage over the window.
        """
        clim = getattr(self, "_dowy_inflow_clim", None)
        if clim is None:
            grp = self.inflows.groupby("dowy")["inflow"].mean()
            clim = np.zeros(367, dtype=float)
            for d, v in grp.items():
                di = int(d)
                if 1 <= di <= 366:
                    clim[di] = float(v)
            self._dowy_inflow_clim = clim
        lo, hi = max(1, int(dowy)), min(366, int(window_end_dowy))
        return float(clim[lo:hi + 1].sum()) if hi >= lo else 0.0

    def compute_carryover_release_cap(
        self, mowy: int, st_1: float, ari: float, days_left: int, target,
        expected_inflow: float = 0.0,
    ) -> float:
        """Upper release cap (TAF/day) defending the agent's carryover target.

        Fully agent-governed: a committed target (``>= 0``) is enforced; a ``None``/negative
        target (the agent set none) returns ``inf`` (no cap). When enforced, rations the
        drawdown above ``target`` over ``days_left`` (days to the end of the operating window),
        NET of ``expected_inflow`` — the inflow expected to refill storage over the window.
        Netting that inflow front-loads the throttle LESS: ignoring it under-releases early and
        then ramps up as the unspent inflow reveals itself (the July supplemental ramp); netting
        it gives a roughly flat per-day allowance, so the discretionary supplemental layer is
        delivered evenly. Releasing at this cap lands end-of-window storage near ``target`` GIVEN
        the expected inflow (realized inflow above/below climatology lands storage above/below
        it). The hard obligation floor is re-imposed in :meth:`evaluate`, so this can never force
        a release below the minimum flow or the senior committed deliveries.
        """
        if target is None or float(target) < 0 or days_left <= 0:
            return float("inf")
        return max(0.0, (st_1 - float(target) + float(expected_inflow)) / days_left)
