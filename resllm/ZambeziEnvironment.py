import numpy as np
import math

class IrrigationPolicy:
    """
    Handles the parabolic irrigation diversion policy for the Zambezi basin.
    Translates the C++ irrparam_function logic 1:1.
    """
    def __init__(self, num_irr, min_params, max_params, active=True):
        self.num_irr = num_irr
        self.irr_input_min = np.array(min_params)
        self.irr_input_max = np.array(max_params)
        self.irr_parab_param = np.zeros(num_irr * 2)

    def set_irr_parameters(self, irr_theta):
        """
        Loads the flat array of optimized parameters (hdg and m) 
        from your pre-calculated policy files (e.g., Zambezi_5_res.txt).
        """
        # We need 2 parameters (hdg, m) for each irrigation district
        self.irr_parab_param = np.array(irr_theta[:self.num_irr * 2])

    def clear_irr_parameters(self):
        self.irr_parab_param = np.zeros(self.num_irr * 2)

    def get_irr_output(self, input_inflow, input_w, irr_district, irr_district_idx, apply_hdg):
        """
        Calculates the actual water delivered to the irrigation district.
        
        :param input_inflow: Available water in the river segment
        :param input_w: The actual crop demand (irr_demand)
        :param irr_district: The ID of the district (2 through 9)
        :param irr_district_idx: Array mapping districts to their param starting indices
        :param apply_hdg: Boolean/Int flag (1 = use hedging, 0 = standard delivery)
        """
        # C++ uses district - 2 to map IDs (2-9) to 0-indexed arrays
        idx = irr_district - 2
        
        # Get the starting index in the flat parameter array
        start_param_idx = int(irr_district_idx[idx])
        
        hdg = self.irr_parab_param[start_param_idx]      # Threshold parameter
        m = self.irr_parab_param[start_param_idx + 1]    # Curvature parameter
        
        # De-normalize the hedging threshold
        hdg_dn = hdg * (self.irr_input_max[idx] - self.irr_input_min[idx]) + self.irr_input_min[idx]
        
        # Apply the Parabolic Hedging Logic
        if input_inflow <= hdg_dn and apply_hdg == 1:
            # Restricted delivery based on the curve: w * (inflow / hdg_dn)^m
            delivery = input_w * (math.pow(input_inflow / hdg_dn, m))
            y = max(min(input_inflow, delivery), 0.0)
        else:
            # Standard delivery: give them what they ask for, bounded by what's available
            y = max(min(input_inflow, input_w), 0.0)
            
        return y

class ZambeziEnvironment:
    """
    Exact Python translation of the C++ EMDOPS model for the 5 core Zambezi dams.
    Acts as the physical 'Blackbox' for the LLM multi-agent system.
    """
    def __init__(self, H, integration_steps, integration_steps_delay, config):
        self.H = H
        self.integration_steps = integration_steps  # Days in each month (size 12)
        self.integration_steps_delay = integration_steps_delay # Delay days (size 12)
        self.config = config # Contains curtailment tables, Sp values, etc.
        
        # State tracking equivalent to C++ vectors
        self.r_itt_delay = np.zeros(self.H + 3)
        self.r_itt_delay[1] = 56.183290322580640 # October 1985
        self.r_itt_delay[2] = 59.670678571428570 # November 1985
        self.r_itt_delay[3] = 101.7307419354839 # December 1985
        self.storage = {dam: np.zeros(self.H + 1) for dam in ['ITT', 'KGU', 'KGL', 'KA', 'CB']}
        self.release = {dam: np.zeros(self.H + 1) for dam in ['ITT', 'KGU', 'KGL', 'KA', 'CB']}
        self.evaporation = {dam: np.zeros(self.H) for dam in ['ITT', 'KGU', 'KGL', 'KA', 'CB']}
        #self.level = {dam: np.zeros(self.H) for dam in ['ITT', 'KGU', 'KGL', 'KA', 'CB']}
        
        # Output tracking
        self.prod_hyd = {dam: np.zeros(self.H) for dam in ['ITT', 'KGU', 'KGL', 'KA', 'CB']}
        self.prod_sol = {dam: np.zeros(self.H) for dam in ['KA_N', 'KA_S', 'CB']}
        
        # Irrigation and Deficits
        self.irr_delivery = {i: np.zeros(self.H + 1) for i in [2, 3, 4, 5, 6, 7, 8, 9]}
        self.irr_deficit_sq = {i: np.zeros(self.H) for i in [2, 3, 4, 5, 6, 7, 8, 9]}

        # Initialize the policy based on your config file arrays
        self.irr_policy = IrrigationPolicy(
            num_irr=config['irr_param']['num_irr'],
            min_params=config['irr_param']['mParam'],
            max_params=config['irr_param']['MParam']
        )

        # Load the indices that map district IDs to the flat parameter array
        self.irr_district_idx = config['irr_district_idx']
        self.apply_hdg = config['apply_hdg']
        
        # Country production metrics
        self.zam_prod = np.zeros(self.H)
        self.zim_prod = np.zeros(self.H)
        self.moz_prod = np.zeros(self.H)

    def _get_irrigation(self, available_water, demand):
        """Mock of irrPolicy->get_irr_output"""
        return min(available_water, demand)
    
    def erase_execution(self, t):
        """Resets the state of the environment to just before executing month t."""
        for dam in ['ITT', 'KGU', 'KGL', 'KA', 'CB']:
            self.storage[dam][t+1] = 0.0
            self.release[dam][t+1] = 0.0
            self.evaporation[dam][t] = 0.0
            #self.level[dam][t] = 0.0
            self.prod_hyd[dam][t] = 0.0
            self.prod_sol[dam][t] = 0.0
        for i in [2, 3, 4, 5, 6, 7, 8, 9]:
            self.irr_delivery[i][t+1] = 0.0
            self.irr_deficit_sq[i][t] = 0.0

    def execute_timestep(self, t, moy, inflows, actions, irr_demands, reservoirs):
        """
        Executes exactly one month of physics.
        actions: dict containing u_itt, u_kgu, u_kgl, u_ka, u_cb (0 to 1 scaling factors)
        reservoirs: dict containing the Python equivalent of your C++ Reservoir objects
        """
        days = self.integration_steps[moy - 1]
        
        # ---------------------------------------------------------
        # 1. KAFUE RIVER CASCADE (ITT -> KGU -> KGL)
        # ---------------------------------------------------------
        
        # ITEZHITEZHI (ITT)
        # C++: sd_rd = Itezhitezhi->integration_daily(...)
        st_itt, rt_itt, evap_itt = reservoirs['ITT'].integration_daily(
            days, self.storage['ITT'][t], actions['ITT'], inflows['q_Itt'], moy
        )
        self.storage['ITT'][t+1] = st_itt
        self.release['ITT'][t+1] = rt_itt
        self.evaporation['ITT'][t] = evap_itt
        
        # 2-month delay buffer
        # C++: r_itt_delay[t + 3] = output->itt[2][t + 1] * (integrationStep) / (integrationStep_delay)
        delay_factor = days / self.integration_steps_delay[moy - 1]
        self.r_itt_delay[t + 2] = rt_itt * delay_factor  # Note: 0-indexed adjustment
        
        # Irrigation 4 (Kafue Flats)
        q_kf_avail = inflows['q_KafueFlats'] + self.r_itt_delay[t]
        self.irr_delivery[4][t+1] = self.irr_policy.get_irr_output(
            input_inflow=q_kf_avail, 
            input_w=irr_demands[4][t], 
            irr_district=4, 
            irr_district_idx=self.irr_district_idx, 
            apply_hdg=self.apply_hdg
        )
        
        # KAFUE GORGE UPPER (KGU)
        q_kgu_in = q_kf_avail - self.irr_delivery[4][t+1]
        st_kgu, rt_kgu, evap_kgu = reservoirs['KGU'].integration_daily(
            days, self.storage['KGU'][t], actions['KGU'], q_kgu_in, moy
        )
        self.storage['KGU'][t+1] = st_kgu
        self.release['KGU'][t+1] = rt_kgu
        self.evaporation['KGU'][t] = evap_kgu
        
        # KAFUE GORGE LOWER (KGL)
        q_kgl_in = rt_kgu
        st_kgl, rt_kgl, evap_kgl = reservoirs['KGL'].integration_daily(
            days, self.storage['KGL'][t], actions['KGL'], q_kgl_in, moy
        )
        self.storage['KGL'][t+1] = st_kgl
        self.release['KGL'][t+1] = rt_kgl
        self.evaporation['KGL'][t] = evap_kgl

        # ---------------------------------------------------------
        # 2. MAIN STEM & LOWER ZAMBEZI (KA -> CB)
        # ---------------------------------------------------------
        
        # Bypass BG and DG inflows directly to Kariba area
        q_upper_zambezi = inflows['q_Bg'] + inflows['q_Cuando'] + inflows['q_KaLat']
        self.irr_delivery[2][t+1] = self.irr_policy.get_irr_output(
            input_inflow=q_upper_zambezi,
            input_w=irr_demands[2][t],
            irr_district=2,
            irr_district_idx=self.irr_district_idx,
            apply_hdg=self.apply_hdg
        )
        
        # KARIBA (KA)
        q_ka_in = q_upper_zambezi - self.irr_delivery[2][t+1]
        st_ka, rt_ka, evap_ka = reservoirs['KA'].integration_daily(
            days, self.storage['KA'][t], actions['KA'], q_ka_in, moy
        )
        self.storage['KA'][t+1] = st_ka
        self.release['KA'][t+1] = rt_ka
        self.evaporation['KA'][t] = evap_ka
        
        # Mid-Zambezi Irrigation (3, 5, 6)
        self.irr_delivery[3][t+1] = self.irr_policy.get_irr_output(
                input_inflow=rt_ka, 
                input_w=irr_demands[3][t], 
                irr_district=3, 
                irr_district_idx=self.irr_district_idx, 
                apply_hdg=self.apply_hdg
            )
        self.irr_delivery[5][t+1] = self.irr_policy.get_irr_output(
                input_inflow=rt_kgl, 
                input_w=irr_demands[5][t], 
                irr_district=5, 
                irr_district_idx=self.irr_district_idx, 
                apply_hdg=self.apply_hdg
            )
        
        q_irr6_avail = rt_ka - self.irr_delivery[3][t+1] + rt_kgl - self.irr_delivery[5][t+1]
        self.irr_delivery[6][t+1] = self.irr_policy.get_irr_output(
                input_inflow=q_irr6_avail, 
                input_w=irr_demands[6][t], 
                irr_district=6, 
                irr_district_idx=self.irr_district_idx, 
                apply_hdg=self.apply_hdg
            )

        # CAHORA BASSA (CB)
        q_cb_in = inflows['q_Cb'] + q_irr6_avail - self.irr_delivery[6][t+1]
        st_cb, rt_cb, evap_cb = reservoirs['CB'].integration_daily(
            days, self.storage['CB'][t], actions['CB'], q_cb_in, moy
        )
        self.storage['CB'][t+1] = st_cb
        self.release['CB'][t+1] = rt_cb
        self.evaporation['CB'][t] = evap_cb
        
        self.irr_delivery[7][t+1] = self._get_irrigation(rt_cb, irr_demands[7])

        q_8_avail = rt_cb - self.irr_delivery[7][t+1]
        self.irr_delivery[8][t+1] = self.irr_policy.get_irr_output(
                input_inflow=q_8_avail, 
                input_w=irr_demands[8][t], 
                irr_district=8, 
                irr_district_idx=self.irr_district_idx, 
                apply_hdg=self.apply_hdg
            )
        
        q_9_avail = q_8_avail + inflows['q_Shire'] - self.irr_delivery[8][t+1] 
        self.irr_delivery[9][t+1] = self.irr_policy.get_irr_output(
                input_inflow=q_9_avail, 
                input_w=irr_demands[9][t], 
                irr_district=9, 
                irr_district_idx=self.irr_district_idx,
                apply_hdg=self.apply_hdg,
        )

        # HYDROPOWER & SOLAR CURTAILMENT PHYSICS
        self.prod_hyd['ITT'][t] = reservoirs['ITT'].calculate_power(st_itt, rt_itt, moy, days, self.config)['hydro']
        self.prod_hyd['KGU'][t] = reservoirs['KGU'].calculate_power(st_kgu, rt_kgu, moy, days, self.config)['hydro']
        self.prod_hyd['KGL'][t] = reservoirs['KGL'].calculate_power(st_kgl, rt_kgl, moy, days, self.config)['hydro']
        self.prod_hyd['KA'][t] = reservoirs['KA'].calculate_power(st_ka, rt_ka, moy, days, self.config)['hydro']
        self.prod_hyd['CB'][t] = reservoirs['CB'].calculate_power(st_cb, rt_cb, moy, days, self.config)['hydro']

        self.prod_sol['KA_N'][t] = reservoirs['KA'].calculate_power(st_ka, rt_ka, moy, days, self.config)['solar_N']
        self.prod_sol['KA_S'][t] = reservoirs['KA'].calculate_power(st_ka, rt_ka, moy, days, self.config)['solar_S']
        self.prod_sol['CB'][t] = reservoirs['CB'].calculate_power(st_cb, rt_cb, moy, days, self.config)['solar']
        
        # Squared Irrigation Deficits (g_deficit_norm)
        for i in [2, 3, 4, 5, 6, 7, 8, 9]:
            deficit = max(irr_demands[i] - self.irr_delivery[i][t+1], 0)
            self.irr_deficit_sq[i][t] = (deficit / irr_demands[i]) ** 2 if irr_demands[i] > 0 else 0
        