import numpy as np
from src.reservoirs.baseReservoir import BaseReservoir
import src.utils

class Itezhitezhi(BaseReservoir):
    """
    Itezhitezhi reservoir simulation class.
    Inherits interpolation and integration physics from BaseReservoir.
    Overrides actual_release_daily to implement "No Flow Augmentation" MEF logic.
    """

    def __init__(
        self,
        name,
        config_path,
        prompt_config,
        model_and_version,
        policy,
    ):
        super().__init__(
            name,
            config_path,
            prompt_config,
            model_and_version,
            policy,
        )

    def actual_release_daily(self, desired_release_m3s, S_m3, cmonth, n_sim_m3s, mef_m3s):
        """
        Overrides the base method to apply Itezhitezhi's specific environmental rules.
        Translates the C++ `itezhitezhi::actual_release_MEF` exactly.
        All parameters are in m3/s except S_m3 (m3).
        """
        # 1. Base physical constraints (C++ min_release and max_release)
        qm = self.compute_min_release(S_m3)
        qM = self.compute_max_release(S_m3)

        # 2. Constrain the target decision between the physical min and max
        rr = min(qM, max(qm, desired_release_m3s))

        # 3. Itezhitezhi "No Flow Augmentation" Logic
        base_40 = 40.0  # The hardcoded ecological baseline in m3/s
        
        if mef_m3s <= base_40:
            # Itezhitezhi MUST release a MF of 40 m3/sec all year round
            rr_mef = max(rr, mef_m3s)
        else:
            if n_sim_m3s <= base_40:
                # Natural flow is disastrously low, enforce the baseline
                rr_mef = max(rr, base_40)
            elif base_40 < n_sim_m3s < mef_m3s:
                # Natural flow is lower than target MEF; just pass the natural flow.
                # Do NOT augment the flow using stored water.
                rr_mef = max(rr, n_sim_m3s)
            else:
                # Natural flow is plentiful, meet the MEF
                rr_mef = max(rr, mef_m3s)

        # Safety clamp: Ensure environmental rules don't exceed physical max capacity
        return min(qM, rr_mef)

    def calculate_power(self, release_m3s, storage_m3, days_in_month):
        """
        Calculates hydropower production in TWh/yr equivalent for the timestep.
        Uses Itezhitezhi's specific turbine specs from the EMDOPS main loop.
        """
        level = self.volume_to_height(storage_m3)
        
        # Itezhitezhi specific constraints: max turbine flow is 2 * 306 m3/s
        q_turb = min(release_m3s, 2 * 306)
        
        # Head calculation: Tailwater elevation is ~1030.5, Dam crest height factor ~40.50
        head = 40.50 - (1030.5 - level)
        
        # Power formula: Q * H * rho * g * efficiency * time
        # Efficiency = 0.89
        energy_mwh = (q_turb * head * 1000 * 9.81 * 0.89 * (24 * days_in_month)) / 1000000
        
        # Convert to TWh/yr equivalent to match C++ output format
        energy_twh_yr = energy_mwh * 12 / 1000000
        
        return {
            "hydro": energy_twh_yr,
        }