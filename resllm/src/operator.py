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
from agno.db.sqlite import SqliteDb
from textwrap import dedent
from pydantic import BaseModel, Field
from typing import Optional
from typing_extensions import TypedDict



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
            Your goal is to minimize shortages to downstream water supply by releasing water from the reservoir.
            The reservoir is located in a region with a Mediterranean climate, characterized by hot, dry summers and highly variable wet winters.
            The reservoir is operated to meet the municipal and agricultural water supply needs of the region while also maintaining flood control and environmental flow requirements.
            The water year is defined as the period from October through September.
            """
        )

        # - instruction message
        self.instructions = dedent(
            """\
            - You are tasked with determining the percent allocation of water demand to release from the reservoir.
            - At the beginning of each month, you will be asked to update the percent allocation decision based on your current observations.
            - In your determination, consider the volume currently in storage, inflow to date compared to expected inflows, and the need to balance meeting current demands against conserving water for future demands.
            - Note that a shortage is calculated by demand x (100 - percent allocation).
            You have the following information about the reservoir:
            - The maximum operable storage level is {} TAF.
            - The minimum operable storage level is {} TAF.
            """
        ).format(
            reservoir.characteristics["operable_storage_max"],
            reservoir.characteristics["operable_storage_min"],
        )

        self.instructions += f"- The average total water year demand: {reservoir.characteristics['average_water_year_total_demand']}\n"

        self.instructions += """- The average cumulative inflow by beginning of month of the water year: """
        for month in range(12):
            self.instructions += f"Month {month+1}: {reservoir.characteristics['average_cumulative_inflow_by_month'][month]} TAF | "
        self.instructions += "\n"

        self.instructions += """- The average remaining demand by beginning of month of the water year: """
        for month in range(12):
            self.instructions += f"Month {month+1}: {reservoir.characteristics['average_remaining_demand_by_month'][month]} TAF | "
        self.instructions += "\n"

        if self.reservoir.characteristics["wy_forecast_file"] is not False:
            self.instructions += dedent(
                """\
                - You have access to a probabilistic forecast of inflows for the remainder of the water year.
                - The probabilistic forecast includes the ensemble mean, and 10th and 90th percentile expected water year inflow.
                - Use this forecast to inform your allocation decision.
                """
            )

        if self.include_red_herring:
            self.instructions += dedent(
                """\
                - Puppies like to play, explore their surroundings with boundless curiosity, and chew on just about everything they can get their teeth on. They also love to sleep deeply after their bursts of energy, often curling up in the coziest spots they can find.
                """
            )

        self.instructions += dedent(
            """\
            - Assign an importance ranking ("very high"=1, "high"=2, "medium"=3, "low"=4, or "no importance"=0) to the reservoir management concepts supporting your decision.
            """
        )

        # = response model
        self.response_model = AllocationDecision
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
        qwy_forecast_mean: Optional[float] = None,
        qwy_forecast_10: Optional[float] = None,
        qwy_forecast_90: Optional[float] = None,
    ):
        """
        Generates a monthly water suuply allocation decision based on the current state of the reservoir and water demand.
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
            if self.reservoir.characteristics["wy_forecast_file"] is not False:
                qwy_forecast_mean = int(
                    self.reservoir.forecasted_inflows.loc[
                        (self.reservoir.forecasted_inflows["date"] == date), "QCYFHM"
                    ].values[0]
                )
                qwy_forecast_10 = int(
                    self.reservoir.forecasted_inflows.loc[
                        (self.reservoir.forecasted_inflows["date"] == date), "QCYFH1"
                    ].values[0]
                )
                qwy_forecast_90 = int(
                    self.reservoir.forecasted_inflows.loc[
                        (self.reservoir.forecasted_inflows["date"] == date), "QCYFH9"
                    ].values[0]
                )

        # set the observation string
        self.observation = dedent(
            """\
            It is the beginning of month {} of the water year.
            """.format(
                int(mowy) if mowy is not None else 0
            )
        )

        # add the current state of the reservoir
        if mowy > 1:
            self.observation += dedent(
                """\
                    So far this water year, {} TAF of reservoir inflow has been observed.
                """
            ).format(int(qwyaccum) if qwyaccum is not None else 0)

        # add the current state of the reservoir
        self.observation += dedent(
            """\
            There is currently {} TAF in storage.
            """
        ).format(int(st_1) if st_1 is not None else 0)

        # add forecasted inflows if available
        if qwy_forecast_mean is not None:
            self.observation += dedent(
                """\
                The probabilistic forecasted inflows for the remainder of the water year are:
                - Mean (expected): {} TAF
                - 10th percentile: {} TAF
                - 90th percentile: {} TAF
                """
            ).format(
                int(qwy_forecast_mean) if qwy_forecast_mean is not None else 0,
                int(qwy_forecast_10) if qwy_forecast_10 is not None else 0,
                int(qwy_forecast_90) if qwy_forecast_90 is not None else 0,
            )

        # add the remaining demand
        self.observation += dedent(
            """\
            There is approximately {} TAF of water demand to meet over the remainder of the water year.
            """
        ).format(int(d_wy_rem) if d_wy_rem is not None else 0)
        if mowy >= 9:
            self.observation += dedent(
                """\
                Also, note that next water year is approaching and the first three months have a demand of {} TAF.
                """
            ).format(int(self.reservoir.demand[0:90].sum()))

        # add the instruction to provide a percent allocation decision
        self.observation += dedent(
            """\
            The previous percent allocation decision was {} percent.
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
        Makes a demand release decision based on the current state of the reservoir and water demand.
        """
        # run the agent
        content = None
        attempts = 0
        while (content is None or isinstance(content, str)) and attempts < 3:
            self.response: RunOutput = self.agent.run(self.observation)
            content = self.response.content
            attempts += 1
            if attempts == 3:
                raise Exception(f"{self.model.provider} API call failed after 3 attempts")
        
        if self.include_double_check:
            self.check_answer = dedent(
                """\
                Double check your response and make sure you are confident in the percent allocation decision. Comment on any changes to your decision in the justification.
                """
            )
            self.response: RunOutput = self.agent.run(
                self.check_answer
            )
            content = self.response.content

        # get the decision output
        if self.response is None or self.response.content is None:
            raise ValueError("No response received from agent")

        allocation_percent = self.response.content.allocation_percent
        # keep decision within bounds
        if allocation_percent < 0:
            allocation_percent = 0
        elif allocation_percent > 0 and allocation_percent < 1:
            allocation_percent = allocation_percent * 100
        elif allocation_percent > 100:
            allocation_percent = 100

        # record the observation
        self.record.loc[idx, "observation"] = self.system_message + self.instructions + self.observation

        # record the allocation percent
        self.record.loc[idx, "allocation_percent"] = allocation_percent

        # record the allocation justification
        allocation_justification = dedent(
            self.response.content.allocation_reasoning
        )
        self.record.loc[idx, "allocation_justification"] = allocation_justification
        allocation_concept_importance = (
            self.response.content.allocation_concept_importance
        )
        for k, v in allocation_concept_importance.items():
            self.record.loc[idx, k] = v

        return (
            allocation_percent,
            allocation_justification,
            allocation_concept_importance,
        )
