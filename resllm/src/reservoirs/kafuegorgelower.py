from src.reservoirs.baseReservoir import BaseReservoir

class KafueGorgeLower(BaseReservoir):
    """
    Derived class specifically for the Kafue Gorge Lower (KGL) reservoir.
    Overrides standard physical boundaries to use Min/Max step-logic.
    """

    def __init__(self, characteristics: dict = {}):
        super().__init__(characteristics)

    def compute_min_release(self, S_m3):
        """
        Overrides the base method.
        Uses rating_curve_minmax: [min_storage, max_storage, max_spillway_release]
        """
        rc_minmax = self.characteristics["rating_curve_minmax"]
        
        if S_m3 <= rc_minmax[0]:
            return 0.0
        elif S_m3 >= rc_minmax[1]:
            return rc_minmax[2]
        else:
            return 0.0

    def compute_max_release(self, S_m3):
        """
        Overrides the base method.
        Converts excess volume directly into a daily spill rate (m3/s).
        """
        rc_minmax = self.characteristics["rating_curve_minmax"]
        
        if S_m3 <= rc_minmax[0]:
            return 0.0
        else:
            # Replicates: (s - rating_curve_minmax[0]) / (24 * 3600)
            return (S_m3 - rc_minmax[0]) / 86400.0

    def actual_release_daily(self, desired_release_m3s, S_m3, cmonth, n_sim_m3s, mef_m3s):
        """
        Overrides the base method to exactly match the C++ KGL logic.
        Applies the MEF constraint even if it exceeds the physical maximum (qM).
        """
        qm = self.compute_min_release(S_m3)
        qM = self.compute_max_release(S_m3)

        # 1. Constrain target between physical bounds
        rr = min(qM, max(qm, desired_release_m3s))

        # 2. Apply MEF Constraint (fmax(rr, MEF))
        rr_mef = max(rr, mef_m3s)

        return rr_mef
        
    def calculate_power(self, storage_m3, release_m3s, days_in_month, config=None):
        """
        Calculates pure hydropower production.
        (Kafue Gorge Lower does not have floating solar installed in the EMDOPS model).
        """
        # Step 1: Get current water level (elevation) from storage
        level = self.volume_to_height(storage_m3)
        
        # Step 2: Calculate turbine flow
        # C++: qTurb_Temp = fmin(output->kgl[2][t + 1], 97.4 * 5);
        # 5 turbines with a maximum discharge capacity of 97.4 m3/s each
        q_turb = min(release_m3s, 97.4 * 5.0)
        
        # Step 3: Calculate Head
        # C++: headTemp = (182.7 - (586 - output->kgl[1][t]));
        head = 182.7 - (586.0 - level)
        
        # Step 4: Calculate Energy Production
        # C++: ((qTurb * head * 1000 * 9.81 * 0.88 * (24 * days)) / 1000000) * 12 / 1000000
        # Efficiency = 0.88, Gravity = 9.81, Density = 1000
        hydropower = ((q_turb * head * 1000.0 * 9.81 * 0.88 * (24 * days_in_month)) / 1e6) * 12 / 1e6
        
        # Returns a dict to maintain consistency with dams that DO have solar
        return {
            "hydro": hydropower,
            "solar": 0.0 
        }