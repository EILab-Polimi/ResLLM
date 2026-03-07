
from src.reservoirs.baseReservoir import BaseReservoir

class KafueGorgeUpper(BaseReservoir):
    """
    Derived class specifically for the Kafue Gorge Upper (KGU) reservoir.
    Introduces a volumetric safety cap to its max release to protect dead storage.
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
        
    def compute_max_release(self, S_m3):
        """
        Overrides the base method.
        Calculates max physical release, but caps it to prevent drawing 
        the reservoir below 5 million cubic meters (5e6) in a single day.
        """
        # 1. Standard physical rating curve max (calls the BaseReservoir method)
        q_rating_max = super().compute_max_release(S_m3)
        
        # 2. Volumetric safety cap (s - 5,000,000 m3) / seconds_in_day
        q_s_max = (S_m3 - 5e6) / 86400.0
        
        # 3. The actual max is the lesser of the two constraints
        return min(q_rating_max, q_s_max)

    def actual_release_daily(self, desired_release_m3s, S_m3, cmonth, n_sim_m3s, mef_m3s):
        """
        Overrides the base method to match C++ kafuegorgeupper::actual_release_MEF.
        Forces the MEF even if it exceeds physical bounds (qM).
        """
        qm = self.compute_min_release(S_m3)
        qM = self.compute_max_release(S_m3)

        # 1. Constrain target between physical bounds
        rr = min(qM, max(qm, desired_release_m3s))

        # 2. Apply MEF Constraint (C++: fmax(rr, MEF))
        rr_mef = max(rr, mef_m3s)

        return rr_mef
        
    def calculate_power(self, storage_m3, release_m3s, days_in_month, config=None):
        """
        Calculates pure hydropower production.
        (KGU does not have floating solar in the EMDOPS model).
        """
        # Step 1: Get current water level (elevation) from storage
        level = self.volume_to_height(storage_m3)
        
        # Step 2: Calculate turbine flow
        # C++: qTurb_Temp = fmin(output->kgu[2][t + 1], 6 * 42);
        # 6 turbines with a maximum discharge capacity of 42 m3/s each
        q_turb = min(release_m3s, 6 * 42.0)
        
        # Step 3: Calculate Head
        # C++: headTemp = (397 - (977.6 - output->kgu[1][t]));
        head = 397.0 - (977.6 - level)
        
        # Step 4: Calculate Energy Production
        # C++: ((qTurb * head * 1000 * 9.81 * 0.61 * (24 * days)) / 1000000) * 12 / 1000000
        # Efficiency = 0.61, Gravity = 9.81, Density = 1000
        hydropower = ((q_turb * head * 1000.0 * 9.81 * 0.61 * (24 * days_in_month)) / 1e6) * 12 / 1e6
        
        return {
            "hydro": hydropower,
            "solar": 0.0
        }