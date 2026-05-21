# Constitutional Dataset Evaluation Report

## Summary

| Metric | Value |
|---|---:|
| total_rows | 10 |
| valid_rows | 10 |
| valid_row_rate | 1.000 |
| schema_error_rate | 0.000 |
| constitutional_violation_rate | 0.000 |
| duplicate_input_rate | 0.000 |
| ambiguity_density | 0.100 |
| context_resolution_failure_rate | 0.000 |
| invalid_followup_rate | 0.000 |
| lexical_diversity | 0.493 |

## Scenario Counts

| Scenario | Count |
|---|---:|
| complete_tool_call | 4 |
| missing_required_field | 3 |
| context_resolves_field | 2 |
| ambiguous_request | 1 |

## Scenario Distribution Error

| Scenario | Target | Observed | Error | Abs Error |
|---|---:|---:|---:|---:|
| ambiguous_request | 0.100 | 0.100 | 0.000 | 0.000 |
| complete_tool_call | 0.400 | 0.400 | 0.000 | 0.000 |
| context_resolves_field | 0.200 | 0.200 | 0.000 | 0.000 |
| missing_required_field | 0.300 | 0.300 | 0.000 | 0.000 |

## Violations

No violations detected.

## Interpretation

- `valid_row_rate` measures structural success.
- `constitutional_violation_rate` measures behavioral law compliance.
- `duplicate_input_rate` measures generator collapse.
- `scenario_distribution_error` measures whether the generated world matches the intended world.
- `ambiguity_density` measures how much of the dataset stresses behavioral boundaries.
