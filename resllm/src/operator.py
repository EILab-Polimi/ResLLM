#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resllm.operator

Defines the Operator agent for reservoir management decisions. Uses a language
model to compute monthly percent-allocation release decisions based on
current storage, inflows, and remaining demand.

"""
import pandas as pd
from agno.agent import Agent, RunOutput
from src.utils import prompt_procedure_text, prompt_closure_text, concept_map
from agno.db.sqlite import SqliteDb
from src.utils import forecast_columns_definitions
from textwrap import dedent
from pydantic import BaseModel, Field
from typing import Optional
from typing_extensions import TypedDict
import json
import re
import time


class OperationalConcepts(TypedDict):
    """
    Dictionary for importance of operational concepts related to reservoir management.
    """

    environment_setting: int
    goal: int
    operational_limits: int
    average_cumulative_inflow_by_month: int
    average_remaining_demand_by_month: int
    previous_allocation: int
    current_month: int
    current_storage: int
    current_cumulative_observed_inflow: int
    current_water_year_remaining_demand: int
    next_water_year_demand: int
    mean_forecast: int
    percentile_forecast_10th: int
    percentile_forecast_90th: int
    puppies: int


class AllocationDecision(BaseModel):
    """
    Pydantic model for the demand release decision.
    """

    allocation_reasoning: str = Field(
        ...,
        description=dedent("A brief justification of the percent allocation decision."),
    )
    allocation_percent: float = Field(
        ...,
        description=dedent("The percent allocation decision (from 0-100 percent) which continues or updates the allocation and release from the reservoir."),
    )

    allocation_concept_importance: OperationalConcepts


class ReservoirAllocationOperator:
    """
    Water reservoir agent that uses a language model to determine reservoir operating decisions.
    """

    def __init__(
        self,
        model_and_version: list = [],
        reservoir=None,
        include_double_check: bool = False,
        include_num_history: int = 0,
        include_red_herring: bool = False,
        agent_kwargs: dict = {},
        model_kwargs: dict = {},
    ):
        """
        Initializes the water reservoir agent.
        Parameters:
            model_and_version (list): A list containing the language model type and version.
            reservoir: The reservoir instance to be used.
            agent_kwargs (dict): Additional arguments for the agent.
            model_kwargs (dict): Additional arguments for the model.
        """
        # - reservoir
        if reservoir is None:
            raise ValueError("reservoir cannot be None")
        self.reservoir = reservoir
        self.model_and_version = model_and_version

        # - double check
        self.include_double_check = include_double_check

        # - num history
        self.include_num_history = include_num_history + 1 if include_double_check else include_num_history

        # - include red herring
        self.include_red_herring = include_red_herring

        # - model
        if model_and_version[0] == "OpenAI":
            from agno.models.openai.responses import OpenAIResponses
            self.model = OpenAIResponses(id=model_and_version[1], **model_kwargs)
        
        elif model_and_version[0] == "xAI":
            from agno.models.xai import xAI
            self.model = xAI(id=model_and_version[1], **model_kwargs)

        elif model_and_version[0] == "Google":
            from agno.models.google import Gemini
            self.model = Gemini(id=model_and_version[1], **model_kwargs)

        elif model_and_version[0] == "Ollama":
            from agno.models.ollama import Ollama
            self.model = Ollama(id=model_and_version[1], options=model_kwargs)

        elif model_and_version[0] == "Mistral":
            from agno.models.mistral import MistralChat
            self.model = MistralChat(id=model_and_version[1], **model_kwargs)
        elif model_and_version[0] == "DeepSeek_Local":
            # We use OpenAIResponses because vLLM creates an OpenAI-compatible API
            from agno.models.openai.responses import OpenAIResponses
            self.model = OpenAIResponses(id=f"DeepSeek-R1-Distill-{model_and_version[1]}", **model_kwargs)

        # - system message
        self.system_message = ""
        if "gpt-oss" in model_and_version[1]:
            self.system_message += dedent(
                """\
                Reasoning: high
                """
            )

        self.system_message += dedent(
            """\
            You are a water reservoir operator.
            Your goal is to meet the hydropower production request by releasing water from the reservoir through the turbines, minimizing any hydropower shortfall.
            The reservoir is located in a subtropical region characterized by a distinct wet season and a pronounced dry season, with highly variable inflows between years.
            The reservoir is operated primarily to satisfy hydropower generation needs; all release decisions should be made only to meet the hydropower production request while respecting basic physical and operational limits (e.g., storage bounds, turbine/spill capacity).
            The hydropower system has two turbine groups: North turbines with a total capacity of 6 × 200 m3/s and efficiency 0.48 with and head of 108m, and South turbines with a total capacity of 6 × 140 m3/s and efficiency 0.51 and an head of 110m; turbine releases must not exceed these capacities for each group.
            
            The water year is defined as the period from January through December.
            """
        )

        # - instruction message
        self.instructions = dedent(
            """\
            - You are tasked with determining the percent allocation of water demand to release from the reservoir.
            - Each month, you will be asked to update the percent allocation decision based on your current observations.
            - In your determination, consider the volume currently in storage, inflow to date compared to expected inflows, and the need to balance meeting current demands against conserving water for future demands.
            - Note that a shortage is calculated by current demand x (100 - percent allocation).
            You have the following information about the reservoir:
            - The maximum operable storage level is {:,} cubic meters.
            - The minimum operable storage level is {:,} cubic meters.
            """
        ).format(
            reservoir.characteristics["operable_storage_max"],
            reservoir.characteristics["operable_storage_min"],
        )

        self.instructions += f"- The average total water year demand: {reservoir.characteristics['average_water_year_total_demand']:,} cubic meters\n"

        self.instructions += """- The average cumulative inflow by beginning of month of the water year: """
        for month in range(12):
            self.instructions += f"Month {month+1}: {reservoir.characteristics['average_cumulative_inflow_by_month'][month]:,} cubic meters | "
        self.instructions += "\n"

        self.instructions += """- The average remaining demand by beginning of month of the water year: """
        for month in range(12):
            self.instructions += f"Month {month+1}: {reservoir.characteristics['average_remaining_demand_by_month'][month]:,} cubic meters | "
        self.instructions += "\n"

        # if self.reservoir.characteristics["wy_forecast_file"] is not False:
        #     self.instructions += dedent(
        #         """\
        #         - You have access to a probabilistic forecast of inflows for the remainder of the water year.
        #         - The probabilistic forecast includes the ensemble mean, and 10th and 90th percentile expected water year inflow.
        #         - Use this forecast to inform your allocation decision.
        #         """
        #     )

        if self.include_red_herring:
            self.instructions += dedent(
                """\
                - Puppies like to play, explore their surroundings with boundless curiosity, and chew on just about everything they can get their teeth on. They also love to sleep deeply after their bursts of energy, often curling up in the coziest spots they can find.
                """
            )


        forecast_columns = []
        if self.reservoir.characteristics["wy_forecast_file"] is not False:
            forecast_columns = list(self.reservoir.forecasted_inflows.columns)
            forecast_columns.remove("date")
        for col in forecast_columns:
            concept_map[col] = "<rank 0-4>"

        json_structure = json.dumps({
            "allocation_reasoning": "<string justification>",
            "allocation_percent": "<number 0-100>",
            "allocation_concept_importance": concept_map
        }, indent=4)
        json_structure = json_structure.replace('"<number 0-100>"', '<number 0-100>')
        json_structure = json_structure.replace('"<rank 0-4>"', '<rank 0-4>')

        if self.model_and_version[0] == "DeepSeek_Local":
            self.instructions += dedent(
                f"""\
                {prompt_procedure_text}
                {json_structure}
                {prompt_closure_text}
                """
            )

        else:
            self.instructions += dedent(
                """\
                - Assign an importance ranking ("very high"=4, "high"=3, "medium"=2, "low"=1, or "no importance"=0) to the reservoir management concepts supporting your decision.
                """
            )

        # = response model
        self.response_model = AllocationDecision if not model_and_version[0] == "DeepSeek_Local" else None
        use_json_mode = True if model_and_version[0] == "Ollama" else False
        
        # - agent
        self.agent: Agent = Agent(
            model=self.model,
            description=self.system_message,
            instructions=self.instructions,
            output_schema=self.response_model,
            use_json_mode=use_json_mode,
            db=SqliteDb(db_file=("./agent.db")) if self.include_num_history > 0 else None,
            add_history_to_context=True if self.include_num_history > 0 else False,
            num_history_runs=self.include_num_history if self.include_num_history > 0 else None,
            **agent_kwargs,
        )

        self.record = pd.DataFrame()

    def set_observation(
        self,
        idx: int = 0,
        date: Optional[pd.Timestamp] = None,
        wy: Optional[int] = None,
        mowy: Optional[int] = None,
        dowy: Optional[int] = None,
        alloc_1: Optional[float] = None,
        st_1: Optional[float] = None,
        qwyaccum: Optional[float] = None,
        d_wy_rem: Optional[float] = None,
        tocs: Optional[float] = None,
        max_safe_release: Optional[float] = None,
    ):
        """
        Generates a monthly water supply allocation decision based on the current state of the reservoir and hydropower demand.
        """
        # get inflow and demand data if date specific
        if date is not None:
            inflows = self.reservoir.inflows.copy()

            # get the remaining demand for the water year
            d_wy_rem = int(self.reservoir.demand[(dowy - 1) :].sum())

            # get cumulative water year inflow
            if dowy == 0:
                qwyaccum = 0
            else:
                qwyaccum = int(
                    inflows.loc[(inflows["water_year"] == wy), "inflow"]
                    .values[0 : (dowy - 1)]
                    .sum()
                )

            # get the forecasted inflows
            forecast_dict = {}
            if self.reservoir.characteristics["wy_forecast_file"] is not False:
                self.forecast_lookup = self.reservoir.forecasted_inflows.set_index("date").to_dict("index")
            else:
                self.forecast_lookup = {}
            forecast_updates = self.forecast_lookup.get(date, {})
            if forecast_updates:
                forecast_dict.update(forecast_updates)
            print("Forecast Dict:", forecast_dict)


        # set the observation string
        self.observation = dedent(
            """\n
            - It is the beginning of month {} of the water year.\n
            """.format(
                int(mowy) if mowy is not None else 0
            )
        )

        # add the current state of the reservoir
        if mowy > 1:
            self.observation += dedent(
                """
                    So far this water year, {:,} cubic meters of reservoir inflow has been observed.\n
                """
            ).format(int(qwyaccum) if qwyaccum is not None else 0)

        # add the current state of the reservoir
        self.observation += dedent(
            """
            - There is currently {:,} cubic meters in storage.\n
            """
        ).format(int(st_1) if st_1 is not None else 0)

        # add the top of conservation storage if available
        # if tocs is not None:
        #     self.observation += dedent(
        #         """\n
        #         The top of conservation storage is {:,} cubic meters. Consider that the actual release is calculated as max((0.2 * (Current Month Inflow + Storage level - Top of conservation storage value)), Demand * (Percent allocation / 100))
        #         """
        #     ).format(int(tocs) if tocs is not None else 0)

        # add max safe release if available
        if max_safe_release is not None:
            self.observation += dedent(
                """
                - The maximum safe release for this month is {:,} cubic meters.\n
                """
            ).format(int(max_safe_release) if max_safe_release is not None else 0)

        # add forecasted inflows if available
        if forecast_dict:
            for forecast_key, forecast_value in forecast_dict.items():
                self.observation += dedent(
                    f"""
                    - {forecast_key}: {forecast_columns_definitions[forecast_key][0]} {{:,}} {forecast_columns_definitions[forecast_key][1]}\n
                    """
                ).format(float(forecast_value) if forecast_value is not None else 0)

        # add the current month demand
        self.observation += dedent(
            """
            - The demand for this month is {:,} cubic meters.\n
            """
        ).format(
            int(self.reservoir.demand[(dowy - 1)]) if dowy is not None else 0
        )

        # add the remaining demand
        self.observation += dedent(
            """
            - There is approximately {:,} cubic meters of hydropower demand to meet over the remainder of the water year.\n
            """
        ).format(int(d_wy_rem) if d_wy_rem is not None else 0)
        if mowy >= 9:
            self.observation += dedent(
                """
                Also, note that next water year is approaching and the first three months have a demand of {:,} cubic meters.\n
                """
            ).format(int(self.reservoir.demand[0:90].sum()))

        # add the instruction to provide a percent allocation decision
        self.observation += dedent(
            """
            - The previous percent allocation decision was {} percent.\n\n
            Provide a percent allocation decision (from 0-100 percent) which continues or updates the allocation.
            """
        ).format(
            int(alloc_1) if alloc_1 is not None else 0,
        )

        # record the observation in the decision output
        self.record.loc[idx, "date"] = date
        self.record.loc[idx, "wy"] = wy
        self.record.loc[idx, "mowy"] = mowy
        self.record.loc[idx, "dowy"] = dowy
        self.record.loc[idx, "qwyaccum"] = qwyaccum
        self.record.loc[idx, "d_wy_rem"] = d_wy_rem
        self.record.loc[idx, "st_1"] = st_1

    def make_allocation_decision(self, idx: int = 0):
        """
        Makes a demand release decision based on the current state.
        Handles both Structured Agents (OpenAI) and Reasoning Agents (DeepSeek).
        """
        # Define a wrapper class to mimic Pydantic object for DeepSeek results
        class DecisionWrapper:
            def __init__(self, data):
                self.allocation_percent = float(data.get("allocation_percent", 100))
                self.allocation_reasoning = data.get("allocation_reasoning", "No justification provided.")
                self.allocation_concept_importance = data.get("allocation_concept_importance", {})

        # Prepare Prompt
        if self.model_and_version[0] == "DeepSeek_Local":
            prompt_content = f"SYSTEM INSTRUCTIONS:\n{self.system_message}\nCORE INSTRUCTIONS:\n{self.instructions}\n\nOBSERVATION:\n{self.observation}"
        else:
            prompt_content = self.observation

        # Run the Agent
        content = None
        attempts = 0
        while attempts < 3:
            try:
                time_start = time.time()
                self.response: RunOutput = self.agent.run(prompt_content)
                
                # --- HANDLING DEEPSEEK (Raw Text Parsing) ---
                if self.model_and_version[0] == "DeepSeek_Local":
                    raw_text = self.response.content
                    time_end = time.time()
                    print(f"DeepSeek response time: {time_end - time_start:.2f} seconds")
                    
                    # 1. Extract Thinking (Robust Version)
                    # Try standard <think>...</think>
                    think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
                    
                    if think_match:
                        thinking = think_match.group(1).strip()
                    else:
                        # Fallback: Check if it ends with </think> but has no opening tag
                        # This captures everything up to the closing tag
                        end_match = re.search(r"(.*?)</think>", raw_text, re.DOTALL)
                        if end_match:
                            thinking = end_match.group(1).strip()
                        else:
                            thinking = "No thoughts found (Check logs for raw output)."

                    # Save thinking
                    # self.record.loc[idx, "thinking"] = str(thinking)
                    self.record.loc[idx, "raw_response"] = str(raw_text)
                    time_end2 = time.time()
                    print(f"DeepSeek thinking extraction time: {time_end2 - time_start:.2f} seconds")
                    print(f"\n🧠 [DeepSeek Thought]: {thinking[:200]}...\n")

                    # 2. Extract JSON
                    clean_text = raw_text.replace(think_match.group(0), "") if think_match else raw_text
                    json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
                    
                    if json_match:
                        json_str = json_match.group(0)
                        try:
                            # Clean up potential double-double quotes if they occur (e.g. ""key"")
                            json_str = json_str.replace('""', '"')
                            
                            data = json.loads(json_str)
                            
                            # Wrap it so the rest of the code works unchanged
                            self.response.content = DecisionWrapper(data)
                            content = self.response.content
                            break # Success!
                        except json.JSONDecodeError as e:
                            print(f"JSON Parse Error on attempt {attempts}: {e}")
                            print(f"Faulty JSON string: {json_str[:500]}...")
                    else:
                        print(f"No JSON found on attempt {attempts}")
                        
                # --- HANDLING STANDARD MODELS (OpenAI/Google) ---
                else:
                    content = self.response.content
                    if content is not None:
                        break

            except Exception as e:
                print(f"Agent run failed: {e}")
            
            attempts += 1

        if attempts == 3 and content is None:
            raise Exception(f"{self.model.provider} API call failed after 3 attempts")

        # --- EXISTING LOGIC REMAINS UNCHANGED BELOW ---

        # Double check logic (Optional: you might want to skip this for DeepSeek to save time)
        if self.include_double_check and not self.is_deepseek_local:
             # (Keep your existing double check logic here if desired)
             pass

        allocation_percent = self.response.content.allocation_percent
        
        # keep decision within bounds
        if allocation_percent < 0:
            allocation_percent = 0
        elif allocation_percent > 0 and allocation_percent < 1:
            allocation_percent = allocation_percent * 100
        elif allocation_percent > 100:
            allocation_percent = 100

        # record the observation
        if self.model_and_version[0] == "DeepSeek_Local":
            self.record.loc[idx, "observation"] = prompt_content
        else:
            self.record.loc[idx, "observation"] = self.system_message + self.instructions + prompt_content

        # record the allocation percent
        self.record.loc[idx, "allocation_percent"] = allocation_percent

        # record the allocation justification
        allocation_justification = str(self.response.content.allocation_reasoning)
        self.record.loc[idx, "allocation_justification"] = allocation_justification
        
        allocation_concept_importance = self.response.content.allocation_concept_importance
        print("Allocation Concept Importance:", allocation_concept_importance)
        
        # Safe Loop for dictionary
        for k, v in allocation_concept_importance.items():
            self.record.loc[idx, k] = v

        return (
            allocation_percent,
            allocation_justification,
            allocation_concept_importance,
        )