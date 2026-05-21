from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import ValidationError

from models import (
    AmbiguousRequestRow,
    BaseDatasetRow,
    CompleteToolCallRow,
    ContextResolvesFieldRow,
    MissingRequiredFieldRow,
    validate_constitutional_rules,
)


SCENARIO_MODELS = {
    "complete_tool_call": CompleteToolCallRow,
    "missing_required_field": MissingRequiredFieldRow,
    "context_resolves_field": ContextResolvesFieldRow,
    "ambiguous_request": AmbiguousRequestRow,
}


@dataclass
class Violation:
    row_id: str
    severity: str
    category: str
    message: str


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\u0590-\u05FF ]+", "", text)
    return text


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                rows.append(
                    {
                        "__invalid_json__": True,
                        "__line_number__": line_number,
                        "__error__": str(e),
                    }
                )
    return rows


def parse_row(raw: Dict[str, Any]) -> Tuple[Optional[BaseDatasetRow], Optional[Violation]]:
    if raw.get("__invalid_json__"):
        return None, Violation(
            row_id=f"line_{raw.get('__line_number__')}",
            severity="error",
            category="invalid_json",
            message=raw.get("__error__", "Invalid JSON"),
        )

    scenario = raw.get("scenario_type") or raw.get("metadata", {}).get("scenario_type")
    row_id = raw.get("id", "unknown")

    model = SCENARIO_MODELS.get(scenario)
    if not model:
        return None, Violation(
            row_id=row_id,
            severity="error",
            category="unknown_scenario",
            message=f"Unknown or missing scenario_type: {scenario}",
        )

    # Backward compatibility: older rows put scenario_type inside metadata.
    if "scenario_type" not in raw and "metadata" in raw and "scenario_type" in raw["metadata"]:
        raw = dict(raw)
        raw["scenario_type"] = raw["metadata"]["scenario_type"]

    try:
        parsed = model.model_validate(raw)
        return parsed, None
    except ValidationError as e:
        return None, Violation(
            row_id=row_id,
            severity="error",
            category="schema_validation",
            message=str(e),
        )


def get_target_scenario_distribution(constitution: Dict[str, Any]) -> Dict[str, float]:
    scenarios = constitution.get("distributions", {}).get("scenarios")
    if scenarios:
        return {k: float(v) for k, v in scenarios.items()}

    scenario_types = constitution.get("distributions", {}).get("scenario_types")
    if scenario_types:
        return {k: float(v) for k, v in scenario_types.items()}

    return {}


def get_target_language_distribution(constitution: Dict[str, Any]) -> Dict[str, float]:
    return {
        k: float(v)
        for k, v in constitution.get("distributions", {}).get("languages", {}).items()
    }


def get_target_tool_distribution(constitution: Dict[str, Any]) -> Dict[str, float]:
    return {
        k: float(v)
        for k, v in constitution.get("distributions", {}).get("tools", {}).items()
    }


def normalized_distribution(counter: Counter, total: int) -> Dict[str, float]:
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def distribution_error(target: Dict[str, float], observed: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    keys = sorted(set(target) | set(observed))
    return {
        key: {
            "target": target.get(key, 0.0),
            "observed": observed.get(key, 0.0),
            "error": observed.get(key, 0.0) - target.get(key, 0.0),
            "absolute_error": abs(observed.get(key, 0.0) - target.get(key, 0.0)),
        }
        for key in keys
    }


def lexical_diversity(rows: List[BaseDatasetRow]) -> float:
    tokens: List[str] = []
    for row in rows:
        tokens.extend(normalize_text(row.input).split())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def evaluate_dataset(dataset_path: Path, constitution_path: Path, reports_dir: Path) -> None:
    constitution = yaml.safe_load(constitution_path.read_text(encoding="utf-8"))
    raw_rows = load_jsonl(dataset_path)

    parsed_rows: List[BaseDatasetRow] = []
    violations: List[Violation] = []

    for raw in raw_rows:
        parsed, violation = parse_row(raw)
        if violation:
            violations.append(violation)
        if parsed:
            parsed_rows.append(parsed)

    for row in parsed_rows:
        try:
            validate_constitutional_rules(row)
        except Exception as e:
            violations.append(
                Violation(
                    row_id=row.id,
                    severity="error",
                    category="constitutional_violation",
                    message=str(e),
                )
            )

    total_rows = len(raw_rows)
    valid_rows = len(parsed_rows)
    valid_row_rate = valid_rows / total_rows if total_rows else 0.0

    normalized_inputs = [normalize_text(row.input) for row in parsed_rows]
    duplicate_counts = Counter(normalized_inputs)
    duplicate_inputs = {k: v for k, v in duplicate_counts.items() if v > 1}
    duplicate_row_count = sum(v - 1 for v in duplicate_inputs.values())
    duplicate_input_rate = duplicate_row_count / valid_rows if valid_rows else 0.0

    scenario_counter = Counter(row.scenario_type for row in parsed_rows)
    language_counter = Counter(row.metadata.language for row in parsed_rows)

    tool_counter: Counter[str] = Counter()
    for row in parsed_rows:
        decision = row.expected_decision
        if decision.type == "call_tool":
            tool_counter[decision.tool] += 1
        elif decision.type == "ask_followup":
            if decision.candidate_tool:
                tool_counter[decision.candidate_tool] += 1
            for tool in decision.candidate_tools:
                tool_counter[tool] += 1

    constitutional_violations = [
        v for v in violations if v.category == "constitutional_violation"
    ]
    schema_violations = [
        v for v in violations if v.category in {"schema_validation", "invalid_json", "unknown_scenario"}
    ]

    context_resolution_failures = [
        v for v in constitutional_violations
        if "context_resolves_field" in v.message or "context" in v.message.lower()
    ]

    invalid_followup_failures = [
        v for v in violations
        if "ask_followup" in v.message or "missing_required_field" in v.message
    ]

    ambiguity_rows = [
        row for row in parsed_rows if row.scenario_type == "ambiguous_request"
    ]

    scenario_observed = normalized_distribution(scenario_counter, valid_rows)
    language_observed = normalized_distribution(language_counter, valid_rows)
    tool_total = sum(tool_counter.values())
    tool_observed = normalized_distribution(tool_counter, tool_total)

    scenario_errors = distribution_error(
        get_target_scenario_distribution(constitution), scenario_observed
    )
    language_errors = distribution_error(
        get_target_language_distribution(constitution), language_observed
    )
    tool_errors = distribution_error(
        get_target_tool_distribution(constitution), tool_observed
    )

    summary = {
        "dataset_path": str(dataset_path),
        "constitution_path": str(constitution_path),
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": total_rows - valid_rows,
        "valid_row_rate": valid_row_rate,
        "schema_error_rate": len(schema_violations) / total_rows if total_rows else 0.0,
        "constitutional_violation_rate": len(constitutional_violations) / valid_rows if valid_rows else 0.0,
        "duplicate_input_rate": duplicate_input_rate,
        "duplicate_input_groups": len(duplicate_inputs),
        "ambiguity_density": len(ambiguity_rows) / valid_rows if valid_rows else 0.0,
        "context_resolution_failure_rate": len(context_resolution_failures) / valid_rows if valid_rows else 0.0,
        "invalid_followup_rate": len(invalid_followup_failures) / valid_rows if valid_rows else 0.0,
        "lexical_diversity": lexical_diversity(parsed_rows),
        "scenario_counts": dict(scenario_counter),
        "language_counts": dict(language_counter),
        "tool_counts": dict(tool_counter),
        "scenario_distribution_error": scenario_errors,
        "language_distribution_error": language_errors,
        "tool_distribution_error": tool_errors,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (reports_dir / "violations.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "severity", "category", "message"])
        writer.writeheader()
        for violation in violations:
            writer.writerow(asdict(violation))

    with (reports_dir / "scenario_distribution.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario", "target", "observed", "error", "absolute_error", "count"],
        )
        writer.writeheader()
        for scenario, values in scenario_errors.items():
            writer.writerow(
                {
                    "scenario": scenario,
                    **values,
                    "count": scenario_counter.get(scenario, 0),
                }
            )

    with (reports_dir / "language_distribution.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["language", "target", "observed", "error", "absolute_error", "count"],
        )
        writer.writeheader()
        for language, values in language_errors.items():
            writer.writerow(
                {
                    "language": language,
                    **values,
                    "count": language_counter.get(language, 0),
                }
            )

    with (reports_dir / "tool_distribution.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tool", "target", "observed", "error", "absolute_error", "count"],
        )
        writer.writeheader()
        for tool, values in tool_errors.items():
            writer.writerow(
                {
                    "tool": tool,
                    **values,
                    "count": tool_counter.get(tool, 0),
                }
            )

    md = build_markdown_report(summary, violations)
    (reports_dir / "summary.md").write_text(md, encoding="utf-8")

    print(f"Wrote reports to {reports_dir}")


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_markdown_report(summary: Dict[str, Any], violations: List[Violation]) -> str:
    lines = []
    lines.append("# Constitutional Dataset Evaluation Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    core_metrics = [
        "total_rows",
        "valid_rows",
        "valid_row_rate",
        "schema_error_rate",
        "constitutional_violation_rate",
        "duplicate_input_rate",
        "ambiguity_density",
        "context_resolution_failure_rate",
        "invalid_followup_rate",
        "lexical_diversity",
    ]

    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for metric in core_metrics:
        lines.append(f"| {metric} | {fmt(summary.get(metric))} |")

    lines.append("")
    lines.append("## Scenario Counts")
    lines.append("")
    lines.append("| Scenario | Count |")
    lines.append("|---|---:|")
    for scenario, count in summary["scenario_counts"].items():
        lines.append(f"| {scenario} | {count} |")

    lines.append("")
    lines.append("## Scenario Distribution Error")
    lines.append("")
    lines.append("| Scenario | Target | Observed | Error | Abs Error |")
    lines.append("|---|---:|---:|---:|---:|")
    for scenario, values in summary["scenario_distribution_error"].items():
        lines.append(
            f"| {scenario} | {fmt(values['target'])} | {fmt(values['observed'])} | "
            f"{fmt(values['error'])} | {fmt(values['absolute_error'])} |"
        )

    lines.append("")
    lines.append("## Violations")
    lines.append("")
    if not violations:
        lines.append("No violations detected.")
    else:
        lines.append("| Row ID | Severity | Category | Message |")
        lines.append("|---|---|---|---|")
        for violation in violations[:50]:
            safe_message = violation.message.replace("\n", "<br>")
            lines.append(
                f"| {violation.row_id} | {violation.severity} | {violation.category} | {safe_message} |"
            )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `valid_row_rate` measures structural success.")
    lines.append("- `constitutional_violation_rate` measures behavioral law compliance.")
    lines.append("- `duplicate_input_rate` measures generator collapse.")
    lines.append("- `scenario_distribution_error` measures whether the generated world matches the intended world.")
    lines.append("- `ambiguity_density` measures how much of the dataset stresses behavioral boundaries.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/generated.jsonl"))
    parser.add_argument("--constitution", type=Path, default=Path("constitution.yaml"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/latest"))
    args = parser.parse_args()

    evaluate_dataset(args.dataset, args.constitution, args.reports_dir)


if __name__ == "__main__":
    main()
