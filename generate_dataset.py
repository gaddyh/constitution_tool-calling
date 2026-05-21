from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List

import dspy
import yaml

from models import DatasetRow, validate_constitutional_rules
from program import ConstitutionalBatchGenerator, GenerationPlan

from dotenv import load_dotenv

load_dotenv()
CONSTITUTION_PATH = Path("constitution.yaml")
OUTPUT_PATH = Path("data/generated.jsonl")


def build_plan_from_constitution(config: dict) -> GenerationPlan:
    rows = int(config["generation"].get("rows", 10))

    # MVP: fixed small quotas for 10 rows.
    # Later this should be computed from distributions.
    if rows != 10:
        raise ValueError("This MVP currently expects generation.rows = 10")

    return GenerationPlan(
        total_rows=10,
        complete_tool_call=4,
        missing_required_field=3,
        context_resolves_field=2,
        ambiguous_request=1,
        languages=list(config["distributions"]["languages"].keys()),
        tools=list(config["distributions"]["tools"].keys()),
    )


def validate_dataset_level(rows: List[DatasetRow], plan: GenerationPlan) -> None:
    if len(rows) != plan.total_rows:
        raise ValueError(f"Expected {plan.total_rows} rows, got {len(rows)}")

    inputs = [row.input.strip().lower() for row in rows]
    duplicates = [x for x, count in Counter(inputs).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate inputs detected: {duplicates}")

    scenario_counts = Counter(row.metadata.scenario_type for row in rows)

    expected = {
        "complete_tool_call": plan.complete_tool_call,
        "missing_required_field": plan.missing_required_field,
        "context_resolves_field": plan.context_resolves_field,
        "ambiguous_request": plan.ambiguous_request,
    }

    for scenario, expected_count in expected.items():
        actual_count = scenario_counts.get(scenario, 0)
        if actual_count != expected_count:
            raise ValueError(
                f"Scenario quota failed for {scenario}: expected {expected_count}, got {actual_count}"
            )

    for row in rows:
        if not row.metadata.source_rules:
            raise ValueError(f"{row.id}: metadata.source_rules must be non-empty")

        validate_constitutional_rules(row)

        scenario = row.metadata.scenario_type
        decision = row.expected_decision

        if scenario == "context_resolves_field":
            if not row.context:
                raise ValueError(f"{row.id}: context_resolves_field requires non-empty context")

            if decision.type != "call_tool":
                raise ValueError(f"{row.id}: context_resolves_field must call a tool")

            # MVP-specific check: context must contain at least one value that appears in args.
            context_text = json.dumps(row.context, ensure_ascii=False).lower()
            arg_values = [str(v).lower() for v in decision.args.values()]
            if not any(value and value in context_text for value in arg_values):
                raise ValueError(
                    f"{row.id}: context_resolves_field must include an explicit arg value in context"
                )

        if scenario == "ambiguous_request":
            if decision.type != "ask_followup":
                raise ValueError(f"{row.id}: ambiguous_request must ask followup")
            if len(decision.candidate_tools) < 2:
                raise ValueError(f"{row.id}: ambiguous_request must include at least 2 candidate_tools")


def write_jsonl(rows: List[DatasetRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            row.id = f"row_{idx:03d}"
            f.write(row.model_dump_json() + "\n")


def main() -> None:
    # Configure your LM.
    # Example:
    #   export OPENAI_API_KEY=...
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    config = yaml.safe_load(CONSTITUTION_PATH.read_text(encoding="utf-8"))
    constitution_text = CONSTITUTION_PATH.read_text(encoding="utf-8")

    plan = build_plan_from_constitution(config)

    generator = ConstitutionalBatchGenerator()

    max_attempts = 3
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            batch = generator(constitution=constitution_text, plan=plan)
            rows = batch.rows

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
