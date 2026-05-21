from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import dspy
import yaml

from models import DatasetRow, validate_constitutional_rules


CONSTITUTION_PATH = Path("constitution.yaml")
OUTPUT_PATH = Path("data/generated.jsonl")


class GenerateConstitutionalRow(dspy.Signature):
    """Generate one coherent dataset row for a tool-calling agent.

    The row must obey the constitution.
    Do not invent fields that are not supported by input or context.
    """

    constitution: str = dspy.InputField(desc="Behavioral constitution YAML.")
    scenario_type: str = dspy.InputField(
        desc="One of: complete_tool_call, missing_required_field, context_resolves_field, ambiguous_request."
    )
    tool: str = dspy.InputField(desc="One of: create_reminder, create_task.")
    language: str = dspy.InputField(desc="One of: english, hebrew.")

    row: DatasetRow = dspy.OutputField(desc="A valid DatasetRow object.")


class ConstitutionalDatasetGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateConstitutionalRow)

    def forward(
        self,
        constitution: str,
        scenario_type: str,
        tool: str,
        language: str,
    ) -> DatasetRow:
        prediction = self.generate(
            constitution=constitution,
            scenario_type=scenario_type,
            tool=tool,
            language=language,
        )
        return prediction.row


def weighted_choice(distribution: Dict[str, float]) -> str:
    keys = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(keys, weights=weights, k=1)[0]


def main() -> None:
    # Configure your LM.
    # Example:
    #   export OPENAI_API_KEY=...
    #
    # You can replace this with another DSPy-supported model.
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    config = yaml.safe_load(CONSTITUTION_PATH.read_text(encoding="utf-8"))
    constitution_text = CONSTITUTION_PATH.read_text(encoding="utf-8")

    random.seed(config["generation"]["random_seed"])

    generator = ConstitutionalDatasetGenerator()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    attempts = 0
    target_rows = config["generation"]["rows"]
    max_attempts = target_rows * 5

    while len(rows) < target_rows and attempts < max_attempts:
        attempts += 1

        scenario_type = weighted_choice(config["distributions"]["scenarios"])
        tool = weighted_choice(config["distributions"]["tools"])
        language = weighted_choice(config["distributions"]["languages"])

        try:
            row = generator(
                constitution=constitution_text,
                scenario_type=scenario_type,
                tool=tool,
                language=language,
            )

            # Force stable ID outside the LLM.
            row.id = f"row_{len(rows) + 1:03d}"

            # Pydantic already validated structure.
            # This validates semantic/constitutional rules.
            validate_constitutional_rules(row)

            rows.append(row)

        except Exception as e:
            print(f"Rejected generated row: {e}")

    if len(rows) < target_rows:
        raise RuntimeError(
            f"Only generated {len(rows)} valid rows after {attempts} attempts."
        )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json(indent=None) + "\n")

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
