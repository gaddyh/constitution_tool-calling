from __future__ import annotations

import dspy
from pydantic import BaseModel, Field

from models import DatasetBatch


class GenerationPlan(BaseModel):
    complete_tool_call_count: int = 4
    missing_required_field_count: int = 3
    context_resolves_field_count: int = 2
    ambiguous_request_count: int = 1

    def to_prompt(self) -> str:
        return self.model_dump_json(indent=2)


class GenerateDatasetBatch(dspy.Signature):
    """Generate a complete scenario-partitioned dataset batch.

    Important:
    - Fill each output list with exactly the requested count.
    - Do not place a row in the wrong scenario list.
    - missing_required_field_rows must always ask_followup.
    - context_resolves_field_rows must have missing information in input and explicit resolving evidence in context.
    - complete_tool_call_rows must have all required tool args in the input.
    - ambiguous_request_rows must ask followup and include at least two candidate tools.
    - Every row must have non-empty metadata.source_rules.
    - Inputs must be diverse and non-duplicate.
    - For context_resolves_field rows, copy the resolved missing value exactly from context into args.
    """

    constitution: str = dspy.InputField(desc="Behavioral constitution YAML.")
    generation_plan: str = dspy.InputField(desc="Exact counts for each scenario type.")

    batch: DatasetBatch = dspy.OutputField(desc="Scenario-partitioned structured dataset batch.")


class ConstitutionalBatchGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use Predict first. CoT can make structured output fallback more fragile.
        self.generate = dspy.Predict(GenerateDatasetBatch)

    def forward(self, constitution: str, plan: GenerationPlan) -> DatasetBatch:
        pred = self.generate(
            constitution=constitution,
            generation_plan=plan.to_prompt(),
        )
        return pred.batch
