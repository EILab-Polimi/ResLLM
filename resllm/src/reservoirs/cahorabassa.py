from src.reservoirs.baseReservoir import BaseReservoir

class CahoraBassa(BaseReservoir):
    """
    Derived class specifically for the Cahora Bassa reservoir.
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
        Overrides the base method to exactly match the C++ CahoraBassa logic.
        """
        qm = self.compute_min_release(S_m3)
        qM = self.compute_max_release(S_m3)

        # 1. Constrain target between physical bounds
        # C++: double rr = min(qM, max(qm, uu));
        rr = min(qM, max(qm, desired_release_m3s))

        # 2. Apply MEF Constraint
        # C++: double rr_MEF = fmax(rr, MEF);
        # CRITICAL DIFFERENCE: The C++ code forces the MEF even if it exceeds qM.
        # Your BaseReservoir did `min(qM, max(rr, mef_m3s))`. We override it here 
        # to ensure it behaves exactly like the original EMDOPS C++ physics.
        rr_mef = max(rr, mef_m3s)

        return rr_mef
        
    def calculate_power(self, storage_m3, release_m3s, moy, days_in_month, config):
        """
        Calculates hydropower and solar production.
        Matches the exact logic of _calculate_cb_power.
        """
        # --- 1. SETTINGS & FACTORS ---
        level = self.volume_to_height(storage_m3)
        sp = config['Sp']  # Array of solar capacity multipliers
        sp_pot = config['SpPotential_CB'][moy - 1]
        sys_loss = config['SpSystemLoss']
        
        # Determine solar index. 
        # If config['Sp'][2] is 0.0, sol_idx will be 0.
        sol_idx = int(sp[2] * sp_pot * (1 - sys_loss) / 0.05)
        
        # --- 2. SOLAR PRODUCTION ---
        # Avg daily production lookup * hours * days * units factor
        solar_prod = (config['cb_solar_lookup'][sol_idx] * 24 * days_in_month) * 12 / 1e6
        
        # --- 3. RAW HYDROPOWER PRODUCTION ---
        # Turbine flow capped at 5 turbines * 452 m3/s
        q_turb = min(release_m3s, 5 * 452.0)
        
        # Head calculation based on level
        head = 128.0 - (331.0 - level)
        
        # Potential hydropower (before curtailment) - Twh/yr equivalent
        hyd_temp = ((q_turb * head * 1000.0 * 9.81 * 0.73 * (24 * days_in_month)) / 1e6) * 12 / 1e6
        
        # --- 4. HYDRO-SOLAR CURTAILMENT ---
        # hyd_mult determines the row, sol_idx determines the column
        hyd_mult = (q_turb * head) / (5 * 452.0 * 128.0)
        
        # Look up curtailment factor (percentage of hydro lost to accommodate solar)
        # We cap hyd_mult index at the max matrix size (usually 40)
        row_idx = min(int(hyd_mult / 0.025), 40)
        curtail_penalty = config['cb_hydro_solar_curtailment'][row_idx][sol_idx]
        
        curtail_factor = 1.0 - curtail_penalty
        
        # Final Hydropower
        final_hyd_prod = hyd_temp * curtail_factor

        # Return a dictionary so the Environment can track both
        return {
            "hydro": final_hyd_prod,
            "solar": solar_prod
        }