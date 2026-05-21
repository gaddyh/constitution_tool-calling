from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List

import dspy
import yaml

from models import BaseDatasetRow, validate_constitutional_rules
from program import ConstitutionalBatchGenerator, GenerationPlan


CONSTITUTION_PATH = Path("constitution.yaml")
OUTPUT_PATH = Path("data/generated.jsonl")


def validate_dataset_level(rows: List[BaseDatasetRow], plan: GenerationPlan) -> None:
    expected_total = (
        plan.complete_tool_call_count
        + plan.missing_required_field_count
        + plan.context_resolves_field_count
        + plan.ambiguous_request_count
    )

    if len(rows) != expected_total:
        raise ValueError(f"Expected {expected_total} rows, got {len(rows)}")

    inputs = [row.input.strip().lower() for row in rows]
    duplicates = [x for x, c in Counter(inputs).items() if c > 1]
    if duplicates:
        raise ValueError(f"Duplicate inputs detected: {duplicates}")

    counts = Counter(row.scenario_type for row in rows)
    expected = {
        "complete_tool_call": plan.complete_tool_call_count,
        "missing_required_field": plan.missing_required_field_count,
        "context_resolves_field": plan.context_resolves_field_count,
        "ambiguous_request": plan.ambiguous_request_count,
    }

    for scenario, expected_count in expected.items():
        actual = counts.get(scenario, 0)
        if actual != expected_count:
            raise ValueError(f"{scenario}: expected {expected_count}, got {actual}")

    for row in rows:
        validate_constitutional_rules(row)


def write_jsonl(rows: List[BaseDatasetRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            row.id = f"row_{idx:03d}"
            f.write(row.model_dump_json() + "\n")


def main() -> None:
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    constitution_text = CONSTITUTION_PATH.read_text(encoding="utf-8")

    # MVP fixed plan. Later this can be derived from constitution distributions.
    plan = GenerationPlan(
        complete_tool_call_count=4,
        missing_required_field_count=3,
        context_resolves_field_count=2,
        ambiguous_request_count=1,
    )

    generator = ConstitutionalBatchGenerator()

    max_attempts = 3
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            batch = generator(constitution=constitution_text, plan=plan)
            rows = batch.all_rows()

            validate_dataset_level(rows, plan)
            write_jsonl(rows, OUTPUT_PATH)

            print(f"Wrote {len(rows)} valid rows to {OUTPUT_PATH}")
            return

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt} rejected: {e}")

    raise RuntimeError(f"Failed to generate valid dataset. Last error: {last_error}")


if __name__ == "__main__":
    main()
