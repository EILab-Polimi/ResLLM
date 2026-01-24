#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resllm.operator

Defines the Operator agent for reservoir management decisions. Uses a language
model to compute monthly percent-allocation release decisions based on
current storage, inflows, and remaining demand.

"""
import json
import pandas as pd
from types import SimpleNamespace
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
        debug_response: bool = False,
        use_ollama_native: bool | None = None,
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

        # - debug response
        self.debug_response = debug_response

        # - model identifiers
        self.model_server = model_and_version[0] if model_and_version else None
        self.model_id = model_and_version[1] if len(model_and_version) > 1 else None
        self.model_kwargs = model_kwargs
        if use_ollama_native is None:
            self.use_ollama_native = (
                self.model_server == "Ollama" and bool(self.model_kwargs.get("think"))
            )
        else:
            self.use_ollama_native = use_ollama_native

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
        if self.model_server == "Ollama" and self.use_ollama_native:
            content_obj, thinking_text, raw_text = self._run_ollama_native(self.observation)
            self.response = SimpleNamespace(
                content=content_obj,
                reasoning_content=thinking_text,
                raw_text=raw_text,
                model_provider="Ollama",
            )
        else:
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

        # capture reasoning content if provided by agno
        self.last_reasoning = getattr(self.response, "reasoning_content", None)

        # optionally capture the raw response for inspection
        if self.debug_response:
            self.record.loc[idx, "response_debug"] = self._serialize_response(self.response)

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

        self.record.loc[idx, "model_reasoning"] = self.last_reasoning
        for k, v in allocation_concept_importance.items():
            self.record.loc[idx, k] = v

        return (
            allocation_percent,
            allocation_justification,
            allocation_concept_importance,
        )

    def _run_ollama_native(self, observation: str):
        """
        Use native Ollama chat to capture thinking traces when available.
        Returns: (AllocationDecision, thinking_text, raw_text)
        """
        from ollama import chat

        system_prompt = (
            f"{self.system_message}{self.instructions}"
            "\nRespond with valid JSON for the AllocationDecision schema."
            "\nUse exact keys: allocation_reasoning, allocation_percent, allocation_concept_importance."
            "\nThe allocation_concept_importance object MUST include these exact keys: "
            "environment_setting, goal, operational_limits, average_cumulative_inflow_by_month, "
            "average_remaining_demand_by_month, previous_allocation, current_month, current_storage, "
            "current_cumulative_observed_inflow, current_water_year_remaining_demand, "
            "next_water_year_demand, mean_forecast, percentile_forecast_10th, "
            "percentile_forecast_90th, puppies."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation},
        ]

        think = bool(self.model_kwargs.get("think", False))
        options = {k: v for k, v in self.model_kwargs.items() if k in ("temperature",)}

        def stream_response(msgs):
            """Stream chat response and collect thinking/content separately."""
            thinking, content = [], []
            for chunk in chat(
                model=self.model_id,
                messages=msgs,
                think=think,
                stream=True,
                options=options or None,
                format="json",
            ):
                msg = getattr(chunk, "message", None)
                if msg is None:
                    continue
                if t := getattr(msg, "thinking", None):
                    thinking.append(t)
                if c := getattr(msg, "content", None):
                    content.append(c)
            return "".join(thinking).strip(), "".join(content).strip()

        thinking_text, raw_text = stream_response(messages)

        if self.include_double_check:
            messages += [
                {"role": "assistant", "content": raw_text},
                {"role": "user", "content": "Double check your response and make sure you are confident in the percent allocation decision. Comment on any changes to your decision in the justification."},
            ]
            thinking_2, raw_text = stream_response(messages)
            if thinking_2:
                thinking_text = f"{thinking_text}\n{thinking_2}".strip()

        payload = self._parse_json_response(raw_text)
        payload = self._normalize_ollama_payload(payload)
        return AllocationDecision(**payload), thinking_text, raw_text

    def _parse_json_response(self, raw_text: str) -> dict:
        """Parse JSON from model response, handling common formatting issues."""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # Try cleaning up common issues: markdown code blocks, extra text
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            lines = lines[1:] if lines[0].startswith("```") else lines
            lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
            cleaned = "\n".join(lines).strip()

        # Extract JSON object if surrounded by other text
        if not cleaned.lstrip().startswith("{"):
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start != -1 and end > start:
                cleaned = cleaned[start:end + 1]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ollama returned non-JSON content: {raw_text[:500]}") from exc

    def _normalize_ollama_payload(self, payload: dict) -> dict:
        """Normalize Ollama JSON keys to expected AllocationDecision fields."""
        if not isinstance(payload, dict):
            return payload

        # Handle camelCase to snake_case conversion
        key_map = {
            "allocationReasoning": "allocation_reasoning",
            "allocationPercent": "allocation_percent",
            "allocationConceptImportance": "allocation_concept_importance",
        }
        normalized = {key_map.get(k, k): v for k, v in payload.items()}

        # Unwrap if model wrapped response in a single key
        if not any(k in normalized for k in ("allocation_reasoning", "allocation_percent", "allocation_concept_importance")):
            if len(normalized) == 1:
                inner = next(iter(normalized.values()))
                if isinstance(inner, dict):
                    return self._normalize_ollama_payload(inner)

        # Normalize allocation_concept_importance keys
        aci = normalized.get("allocation_concept_importance")
        if isinstance(aci, dict):
            normalized["allocation_concept_importance"] = self._normalize_concept_keys(aci)

        return normalized

    def _normalize_concept_keys(self, aci: dict) -> dict:
        """Fuzzy-match concept importance keys to expected field names."""
        expected_keys = {
            "environment_setting", "goal", "operational_limits",
            "average_cumulative_inflow_by_month", "average_remaining_demand_by_month",
            "previous_allocation", "current_month", "current_storage",
            "current_cumulative_observed_inflow", "current_water_year_remaining_demand",
            "next_water_year_demand", "mean_forecast", "percentile_forecast_10th",
            "percentile_forecast_90th", "puppies",
        }

        # If keys already match, return as-is
        if set(aci.keys()) == expected_keys:
            return aci

        # Fuzzy matching patterns: (keywords_to_match, target_key)
        patterns = [
            (("environment",), "environment_setting"),
            (("goal",), "goal"),
            (("operational", "limit"), "operational_limits"),
            (("average", "cumulative", "inflow"), "average_cumulative_inflow_by_month"),
            (("average", "remaining", "demand"), "average_remaining_demand_by_month"),
            (("previous", "allocation"), "previous_allocation"),
            (("current", "month"), "current_month"),
            (("current", "storage"), "current_storage"),
            (("current", "cumulative", "inflow"), "current_cumulative_observed_inflow"),
            (("current", "remaining", "demand"), "current_water_year_remaining_demand"),
            (("next", "water", "demand"), "next_water_year_demand"),
            (("mean", "forecast"), "mean_forecast"),
            (("10", "percent"), "percentile_forecast_10th"),
            (("90", "percent"), "percentile_forecast_90th"),
            (("pupp",), "puppies"),
        ]

        def normalize_key(k: str) -> str:
            return " ".join(str(k).strip().lower().replace("_", " ").split())

        mapping = {}
        for k, v in aci.items():
            nk = normalize_key(k)
            for keywords, target in patterns:
                if all(kw in nk for kw in keywords):
                    mapping[target] = v
                    break

        # Fill missing keys with default value
        for key in expected_keys:
            mapping.setdefault(key, 0)

        return mapping

    def _serialize_response(self, response: RunOutput) -> str:
        """
        Best-effort serialization of RunOutput for debugging.
        """
        def to_serializable(obj):
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_serializable(v) for v in obj]
            if hasattr(obj, "model_dump"):
                return to_serializable(obj.model_dump())
            if hasattr(obj, "__dict__"):
                return to_serializable(obj.__dict__)
            return str(obj)

        try:
            payload = to_serializable(response)
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return str(response)
