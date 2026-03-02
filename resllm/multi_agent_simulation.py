import autogen
import copy
import re
import numpy as np
import pandas as pd
import os
import json
import random
from src.agents import ReservoirAgent
from src.ZambeziEnvironment import ZambeziEnvironment
import networkx as nx

class MultiAgentZambeziSimulationRunner:
    def __init__(self, config, simulation_number, data_dir, policy='self-interest', hydro_goal = True, irrigation_goal = False):
        self.config = config['multi_agent_system']
        self.max_rounds = self.config['max_rounds']
        self.simulation_number = simulation_number

        self.current_wy = self.config['start_year']
        self.next_resume_date = None
        self.number_years = self.config['end_year'] - self.config['start_year'] + 1
        self.end_year = self.config['end_year']
        self.steps_per_year = self.config['steps_per_year']

        self.t = 0
        self.policy = policy
        self.hydro_goal = hydro_goal
        self.irrigation_goal = irrigation_goal

        self.inflows = pd.read_csv(os.path.join(data_dir, self.config['inflow_file']))
        self.records = pd.DataFrame(columns=['time_step', 'date', 'dam_name', 'inflow', 'observation', 'reply', 'commitment'])

        self.define_topology()
        
        if policy != 'self-interest':
            self.table_agent = autogen.UserProxyAgent(
                name="Global_Environment",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1
            )

        self.environment = ZambeziEnvironment()
        
        # Register tools to the table
        # This is where you map the operators' local tools to the AutoGen environment
        for op in self.operators:
            autogen.agentchat.register_function(
                op.commit_action, caller=op.autogen_agent, executor=self.table_agent, name=f"commit_dam_{op.name}"
            )

    def define_topology(self):
        self.topology = nx.DiGraph(
            [
                ("itezhitezhi", "kafuegorgeupper"),
                ("kafuegorgeupper", "kafuegorgelower"),
                ("kafuegorgelower", "kariba"),
                ("kariba", "cahorabassa"),
            ]
        )
        self.dam_order = list(nx.topological_sort(self.topology))
        self.operators = {}

        self.operators['kariba'] = ReservoirAgent(
            name='kariba',
            reservoir_config=self.config['system']['kariba'],
                llm_config=self.config['system']['llm-config'],
            )
        self.operators['cahorabassa'] = ReservoirAgent(
            name='cahorabassa',
            reservoir_config=self.config['system']['cahorabassa'],
                llm_config=self.config['system']['llm-config'],
            )
        self.operators['itezhitezhi'] = ReservoirAgent(
            name='itezhitezhi',
            reservoir_config=self.config['system']['itezhitezhi'],
                llm_config=self.config['system']['llm-config'],
            )
        self.operators['kafuegorgelower'] = ReservoirAgent(
            name='kafuegorgelower',
            reservoir_config=self.config['system']['kafuegorgelower'],
                llm_config=self.config['system']['llm-config'],
            )
        self.operators['kafuegorgeupper'] = ReservoirAgent(
            name='kafuegorgeupper',
            reservoir_config=self.config['system']['kafuegorgeupper'],
                llm_config=self.config['system']['llm-config'],
            )

    def get_inflows_for_date(self, date):
        inflows_for_date = self.inflows[self.inflows['date'] == date]
        if inflows_for_date.empty:
            return {}
        else:
            return inflows_for_date.to_dict(orient='records')[0]

    def _step(self, ty, d, wy):
        mowy = d.month
       
        # --- PHASE 2: THE NEGOTIATION (THE TABLE) ---
        print(f"[Sample {self.simulation_number}] Step {self.t}: Initiating negotiation for {d.strftime('%B %Y')}")
        
        for op in self.operators:
            op.current_commitment = None

        if self.policy == 'self-interest' and self.irrigation == 'None':
            self._isolated_step(self, ty, d, wy, mowy)
        elif self.policy == 'collaborative-information-exchange':
            self._collaborative_information_exchange_step(self, ty, d, wy, mowy)
        elif self.policy == 'collaborative-global':
            self._global_optimization_step(self, ty, d, wy, mowy)
        else:
            raise ValueError(f"Unknown policy: {self.policy}")
        
        self.t += 1
        
    def _isolated_step(self, ty, d, wy, mowy):  

        natural_inflows = self.get_inflows_for_date(d)  # Get natural inflow for this dam and date
        target_releases = {key: None for key in self.operators.keys()}

        for dam_name in self.dam_order:
            op = self.operators[dam_name]
            
            if dam_name == 'itezhitezhi':
                inflow = natural_inflows['Inflow_qInfItt']
            elif dam_name == 'kariba':
                inflow = natural_inflows['Inflow_qInfKaLat'] + natural_inflows['Inflow_qCuando'] + natural_inflows['Inflow_qInfBg']
            elif dam_name == 'kafuegorgeupper':  # Kafue Gorge Upper
                delayed_itt_water = self.environment.r_itt_delay[self.t] 
                inflow = natural_inflows['Inflow_qKafueFlats'] + delayed_itt_water
            elif dam_name == 'kafuegorgelower':  # Kafue Gorge Lower
                inflow = target_releases.get('kafuegorgeupper', 0.0) # Gets water directly from KGU's commitment
            elif dam_name == 'cahorabassa':   # Cahora Bassa
                inflow = natural_inflows['Inflow_qInfCb'] + target_releases.get('kariba', 0.0) + target_releases.get('kafuegorgelower', 0.0)
            else:
                raise ValueError(f"Unknown dam name: {dam_name}")

            obs_msg = op.generate_observation(mowy=mowy, wy=wy, inflow=inflow)
            reply = op.autogen_agent.generate_reply(messages=obs_msg)

            if op.current_commitment is not None:
                target_releases[dam_name] = op.current_commitment
            else:
                raise ValueError(f"Operator for {dam_name} did not commit to a release. Check the tool execution and response parsing logic.")
            self.record_timestamp(dam_name = dam_name,inflow=inflow, observation=obs_msg, reply=reply, commitment=op.current_commitment)

        self.physical_step(mowy, natural_inflows, target_releases, irr_demands=None)  # Assuming no irrigation demands for now

    def _collaborative_information_exchange_step(self, ty, d, wy, mowy):
        raise NotImplementedError("Collaborative information exchange step logic goes here. Agents share their local information (e.g., current storage, forecasted inflows) with each other before making their decisions. You can implement a simple information sharing protocol followed by independent decision-making, or a more complex negotiation process where agents iteratively share information and update their decisions.")
    
    def _global_optimization_step(self, ty, d, wy, mowy):
        raise NotImplementedError("Global optimization step logic goes here. Agents collaborate to find a joint solution that optimizes a global objective (e.g., maximizing total energy production, minimizing shortages). This could involve a more complex negotiation process where agents propose and evaluate different joint actions, or the use of a centralized optimization algorithm that takes into account the preferences and constraints of all agents.")

    def record_timestamp(self, dam_name, date, inflow, observation, reply, commitment):
        self.records = self.records.append({
            'time_step': self.t,
            'date': date,
            'dam_name': dam_name,
            'inflow': inflow,
            'observation': observation,
            'reply': reply,
            'commitment': commitment
        }, ignore_index=True)


    def physical_step(self, mowy, inflows, actions, irr_demands):
        self.environment.execute_timestep(
            t=self.t, 
            moy=mowy, 
            inflows=inflows, 
            actions=actions, 
            irr_demands=irr_demands,
        )

    def run(self):
        for wy in np.arange(self.current_wy, self.end_year + 1):
            self.current_wy = wy 
            print(f"[Sample {self.n}] Simulating water year {wy}")

            date_range = pd.date_range(start=f"{wy}-01-01", end=f"{wy}-12-31", freq='M')
            
            if self.next_resume_date is not None:
                resume_ts = pd.Timestamp(self.next_resume_date)
                if resume_ts.year == wy:
                    date_range = date_range[date_range >= resume_ts]
                if not date_range.empty:
                    self.next_resume_date = None

            for ty, d in enumerate(date_range):
                self._step(ty, d, wy)
                
                # self._save_csvs(d)
                self.next_resume_date = d + pd.DateOffset(months=1)
                # self.save_checkpoint()

        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        print(f"[Sample {self.n}] Simulation complete.")

    def custom_router(self, last_speaker, groupchat):
        if self.policy == 'self-interest':
            # Find the last speaker's index in the order
            if last_speaker in self.dam_order:
                idx = self.dam_order.index(last_speaker)
                # If not the last dam, pick the next downstream dam
                if idx + 1 < len(self.dam_order):
                    return self.dam_order[idx + 1]
                else:
                    # If last dam, return table agent to close negotiation
                    return self.table_agent.name
            else:
                # If last speaker is not recognized (e.g., first message), start with the first dam
                return self.dam_order[0]
            
        else:
            raise NotImplementedError("Custom routing logic goes here. You can parse the last_message to determine the next speaker based on the negotiation context and the topology of the dams.")
    


