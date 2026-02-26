from src.reservoirs.baseReservoir import BaseReservoir

class Kariba(BaseReservoir):
    """
    Derived class specifically for the Kariba reservoir.
    Handles standard physical routing, but features a highly complex, 
    bilateral power calculation split between Zambia (North) and Zimbabwe (South).
    """

    def __init__(self, characteristics: dict = {}):
        super().__init__(characteristics)

    def actual_release_daily(self, desired_release_m3s, S_m3, cmonth, n_sim_m3s, mef_m3s):
        """
        Overrides the base method to exactly match C++ kariba::actual_release_MEF.
        Applies MEF constraint even if it exceeds physical bounds (qM).
        """
        qm = self.compute_min_release(S_m3)
        qM = self.compute_max_release(S_m3)

        # 1. Constrain target between physical bounds
        rr = min(qM, max(qm, desired_release_m3s))

        # 2. Apply MEF Constraint (fmax(rr, MEF))
        rr_mef = max(rr, mef_m3s)

        return rr_mef
        
    def calculate_power(self, storage_m3, release_m3s, moy, days_in_month, config):
        """
        Calculates hydropower and solar production, split across North and South banks.
        Matches the exact logic of _calculate_ka_power.
        """
        # --- 1. SETTINGS & FACTORS ---
        level = self.volume_to_height(storage_m3)
        sp = config['Sp']  
        sp_pot = config['SpPotential_KA'][moy - 1]
        sys_loss = config['SpSystemLoss']
        
        # Solar lookups (sp[0] = North/Zambia, sp[1] = South/Zimbabwe)
        sol_idx_n = int(sp[0] * sp_pot * (1 - sys_loss) / 0.05)
        sol_idx_s = int(sp[1] * sp_pot * (1 - sys_loss) / 0.05)
        
        # --- 2. SOLAR PRODUCTION ---
        solar_prod_n = (config['ka_solar_lookup'][sol_idx_n] * 24 * days_in_month) * 12 / 1e6
        solar_prod_s = (config['ka_solar_lookup'][sol_idx_s] * 24 * days_in_month) * 12 / 1e6
        
        # --- 3. NORTH BANK HYDRO (ZAMBIA) ---
        # 48.8% allocation, 48% efficiency
        q_turb_n = min(release_m3s * 0.488, 6 * 200.0)
        head_n = 108.0 - (489.5 - level)
        hyd_temp_n = ((q_turb_n * head_n * 1000.0 * 9.81 * 0.48 * (24 * days_in_month)) / 1e6) * 12 / 1e6
        
        # --- 4. SOUTH BANK HYDRO (ZIMBABWE) ---
        # 51.2% allocation, 51% efficiency
        q_turb_s = min(release_m3s * 0.512, 6 * 140.0)
        head_s = 110.0 - (489.5 - level)
        hyd_temp_s = ((q_turb_s * head_s * 1000.0 * 9.81 * 0.51 * (24 * days_in_month)) / 1e6) * 12 / 1e6

        # --- 5. HYDRO-SOLAR CURTAILMENT ---
        hyd_n_mult = (q_turb_n * head_n) / (6 * 200.0 * 108.0)
        hyd_s_mult = (q_turb_s * head_s) / (6 * 140.0 * 110.0)
        
        # Cap indices at 40 (max size of the lookup matrices)
        row_idx_n = min(int(hyd_n_mult / 0.025), 40)
        row_idx_s = min(int(hyd_s_mult / 0.025), 40)
        
        curtail_n = 1.0 - config['ka_n_hydro_solar_curtailment'][row_idx_n][sol_idx_n]
        curtail_s = 1.0 - config['ka_s_hydro_solar_curtailment'][row_idx_s][sol_idx_s]

        # Final Curtailed Production
        final_hyd_n = hyd_temp_n * curtail_n
        final_hyd_s = hyd_temp_s * curtail_s

        # Return a rich dictionary so the Environment can calculate Atkinson Indexes properly
        return {
            "hydro": final_hyd_n + final_hyd_s, # Total physical output
            "hydropower_N": final_hyd_n,             # Zambia's share
            "hydropower_S": final_hyd_s,             # Zimbabwe's share
            "solar_N": solar_prod_n,
            "solar_S": solar_prod_s,
            "solar": solar_prod_n + solar_prod_s,   # Total solar output
        }