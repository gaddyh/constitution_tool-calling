# Constitutional Tool Calling

Constitutional Tool Calling is an experiment in behavioral dataset engineering.

The project explores a simple but powerful idea:

> AI datasets should not be collected first.
> They should be generated from explicit behavioral laws.

Instead of starting with:

```text
examples
    ->
optimization
```

the project starts with:

```text
behavioral constitution
    ->
synthetic world generation
        ->
dataset
            ->
evaluation
                ->
optimization
```

---

# Core Idea

Modern AI systems increasingly rely on behavioral datasets:

- tool calling
- agent routing
- workflow execution
- assistant actions
- follow-up decisions
- memory systems

But these datasets usually contain hidden assumptions:

- inconsistent behavior
- ambiguous supervision
- ontology drift
- hallucinated fields
- unstable boundaries
- contradictory labels

This project makes those assumptions explicit.

---

# Behavioral Constitution

The system begins with a structured `constitution.yaml`.

Example:

```yaml
decision_space:
  allowed_decisions:
    - call_tool
    - ask_followup
    - no_action

tools:
  create_reminder:
    required_fields:
      - text
      - time

    missing_required_policy: ask_followup

policies:
  ambiguity:
    ambiguous_tool: ask_followup

distributions:
  scenarios:
    complete_tool_call: 0.40
    missing_required_field: 0.30
    context_resolves_field: 0.20
    ambiguous_request: 0.10
```

The constitution defines:

- allowed behaviors
- ontology
- ambiguity policies
- hallucination rules
- required fields
- distributions
- behavioral constraints

The constitution becomes the source of truth.

---

# Architecture

```text
constitution.yaml
    ->
DSPy batch generator
    ->
Pydantic validation
    ->
constitutional validation
    ->
dataset metrics
    ->
evaluation reports
```

---

# Structured Behavioral Generation

The generator uses:

- DSPy for structured generation
- Pydantic for behavioral typing
- constitutional validators for semantic rules
- dataset-level metrics for observability

The system does not merely generate JSON.

It generates:

```text
typed behavioral worlds
```

---

# Example Dataset Row

```json
{
  "id": "row_001",
  "input": "Remind me to call Yossi tomorrow at 9",
  "context": {},
  "expected_decision": {
    "type": "call_tool",
    "tool": "create_reminder",
    "args": {
      "text": "call Yossi",
      "time": "tomorrow at 9"
    }
  },
  "metadata": {
    "source_rules": [
      "create_reminder.required_fields_present"
    ],
    "language": "english"
  },
  "scenario_type": "complete_tool_call"
}
```

---

# Dataset Evaluation

The project evaluates the generated dataset itself before any optimization begins.

This introduces a new layer:

```text
behavioral dataset observability
```

Current evaluation layers:

## Structural Metrics

- valid_row_rate
- schema_error_rate
- malformed_decision_rate

## Constitutional Metrics

- constitutional_violation_rate
- invalid_followup_rate
- context_resolution_failure_rate

## Distribution Metrics

- scenario_distribution_error
- language_distribution_error
- tool_distribution_error

## Diversity Metrics

- duplicate_input_rate
- lexical_diversity
- ambiguity_density

---

# Example Evaluation Report

Generated report:

| Metric | Value |
|---|---:|
| valid_row_rate | 1.000 |
| constitutional_violation_rate | 0.000 |
| duplicate_input_rate | 0.000 |
| ambiguity_density | 0.100 |
| lexical_diversity | 0.493 |

Observed scenario quotas perfectly matched the constitutional distributions:

| Scenario | Target | Observed |
|---|---:|---:|
| complete_tool_call | 0.40 | 0.40 |
| missing_required_field | 0.30 | 0.30 |
| context_resolves_field | 0.20 | 0.20 |
| ambiguous_request | 0.10 | 0.10 |

However, evaluation also exposed generator drift:

| Distribution | Problem |
|---|---|
| language_distribution | generator collapsed to English |
| tool_distribution | create_task overrepresented |

This is important:

The system is already capable of detecting:

- behavioral drift
- distribution collapse
- ontology instability
- generator bias

before any model optimization begins.

---

# Why This Matters

Most AI systems optimize models directly.

This project asks a different question:

```text
Can we evaluate and stabilize the behavioral world itself
before optimization begins?
```

The project treats datasets as:

```text
sampled realizations of behavioral laws
```

not merely collections of examples.

---

# Current Status

Implemented:

- constitutional DSL
- DSPy batch generator
- Pydantic behavioral typing
- constitutional validators
- dataset-level metrics
- automatic markdown/json/csv reports
- behavioral distribution analysis

Current focus:

```text
behavioral dataset observability
```

Future directions:

- optimization readiness scoring
- ontology overlap detection
- shortcut-risk analysis
- behavioral geometry metrics
- constitution-driven optimization loops
- synthetic behavioral simulation environments

---

# Project Philosophy

The project follows a dataset-first AI engineering approach:

```text
constitution
    ->
dataset
        ->
metrics
            ->
evaluation
                ->
optimization
```

NOT:

```text
prompt
    ->
hope
```
