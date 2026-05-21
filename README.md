# Simple DSPy Constitutional Dataset Generator

This is the smallest real DSPy version of the idea.

## Concept

```text
constitution.yaml
    ->
DSPy generator
        ->
synthetic dataset rows
            ->
deterministic validation
```

The constitution is the source of truth.

DSPy generates natural language examples and expected structured decisions.

## Install

```bash
pip install dspy pyyaml
```

## Run

```bash
export OPENAI_API_KEY=...
python dspy_generate_dataset.py
```

Output:

```text
data/generated.jsonl
```

## Why this matters

This is not prompt hacking.

The DSPy program has a signature:

```text
constitution + scenario_type + tool + language -> row_json
```

Later, this generator can be optimized using examples of high-quality rows.
