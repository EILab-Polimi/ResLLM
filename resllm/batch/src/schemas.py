#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resllm.batch.schemas

Shared JSON response schemas for the batch request builders. The batch
ablation pipeline uses a reduced decision schema (reasoning + percent only,
no concept-importance rankings), so it cannot reuse the full
``AllocationDecision`` model from ``src.operator``.
"""

# Reduced allocation-decision schema used by the historical and ablation
# batch request builders.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "allocation_reasoning": {
            "type": "string",
            "description": "A brief justification of the percent allocation decision."
        },
        "allocation_percent": {
            "type": "number",
            "description": "The percent allocation to release from the reservoir."
        },
    },
    "required": ["allocation_reasoning", "allocation_percent"],
    "additionalProperties": False,
}
