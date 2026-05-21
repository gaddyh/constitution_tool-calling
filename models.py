from __future__ import annotations

from typing import Any, Dict, List, Literal, Union, Annotated

from pydantic import BaseModel, Field, model_validator


ToolName = Literal["create_reminder", "create_task"]
Language = Literal["english", "hebrew"]


class CallToolDecision(BaseModel):
    type: Literal["call_tool"] = "call_tool"
    tool: ToolName
    args: Dict[str, Any]


class AskFollowupDecision(BaseModel):
    type: Literal["ask_followup"] = "ask_followup"
    reason: Literal["missing_required_tool_field", "ambiguous_tool"]
    missing_fields: List[str] = Field(default_factory=list)
    candidate_tool: ToolName | None = None
    candidate_tools: List[ToolName] = Field(default_factory=list)


class NoActionDecision(BaseModel):
    type: Literal["no_action"] = "no_action"
    reason: str


ExpectedDecision = Annotated[
    Union[CallToolDecision, AskFollowupDecision, NoActionDecision],
    Field(discriminator="type"),
]


class RowMetadata(BaseModel):
    source_rules: List[str] = Field(min_length=1)
    language: Language


class BaseDatasetRow(BaseModel):
    id: str
    input: str
    context: Dict[str, Any] = Field(default_factory=dict)
    expected_decision: ExpectedDecision
    metadata: RowMetadata


class CompleteToolCallRow(BaseDatasetRow):
    scenario_type: Literal["complete_tool_call"] = "complete_tool_call"

    @model_validator(mode="after")
    def validate_decision(self) -> "CompleteToolCallRow":
        if self.expected_decision.type != "call_tool":
            raise ValueError("complete_tool_call must produce call_tool")
        return self


class MissingRequiredFieldRow(BaseDatasetRow):
    scenario_type: Literal["missing_required_field"] = "missing_required_field"

    @model_validator(mode="after")
    def validate_decision(self) -> "MissingRequiredFieldRow":
        if self.expected_decision.type != "ask_followup":
            raise ValueError("missing_required_field must produce ask_followup")
        if self.expected_decision.reason != "missing_required_tool_field":
            raise ValueError("missing_required_field must use reason=missing_required_tool_field")
        return self


class ContextResolvesFieldRow(BaseDatasetRow):
    scenario_type: Literal["context_resolves_field"] = "context_resolves_field"

    @model_validator(mode="after")
    def validate_decision(self) -> "ContextResolvesFieldRow":
        if self.expected_decision.type != "call_tool":
            raise ValueError("context_resolves_field must produce call_tool")
        if not self.context:
            raise ValueError("context_resolves_field requires non-empty context")
        return self


class AmbiguousRequestRow(BaseDatasetRow):
    scenario_type: Literal["ambiguous_request"] = "ambiguous_request"

    @model_validator(mode="after")
    def validate_decision(self) -> "AmbiguousRequestRow":
        if self.expected_decision.type != "ask_followup":
            raise ValueError("ambiguous_request must produce ask_followup")
        if self.expected_decision.reason != "ambiguous_tool":
            raise ValueError("ambiguous_request must use reason=ambiguous_tool")
        if len(self.expected_decision.candidate_tools) < 2:
            raise ValueError("ambiguous_request must include at least two candidate_tools")
        return self


DatasetRow = Annotated[
    Union[
        CompleteToolCallRow,
        MissingRequiredFieldRow,
        ContextResolvesFieldRow,
        AmbiguousRequestRow,
    ],
    Field(discriminator="scenario_type"),
]


class DatasetBatch(BaseModel):
    complete_tool_call_rows: List[CompleteToolCallRow]
    missing_required_field_rows: List[MissingRequiredFieldRow]
    context_resolves_field_rows: List[ContextResolvesFieldRow]
    ambiguous_request_rows: List[AmbiguousRequestRow]

    def all_rows(self) -> List[BaseDatasetRow]:
        return [
            *self.complete_tool_call_rows,
            *self.missing_required_field_rows,
            *self.context_resolves_field_rows,
            *self.ambiguous_request_rows,
        ]


def validate_constitutional_rules(row: BaseDatasetRow) -> None:
    decision = row.expected_decision

    if decision.type == "call_tool" and decision.tool == "create_reminder":
        required = {"text", "time"}
        missing = required - set(decision.args)
        if missing:
            raise ValueError(f"create_reminder missing required args: {sorted(missing)}")

    if decision.type == "call_tool" and decision.tool == "create_task":
        required = {"text"}
        missing = required - set(decision.args)
        if missing:
            raise ValueError(f"create_task missing required args: {sorted(missing)}")

