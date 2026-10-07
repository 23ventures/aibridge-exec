from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ACTION_NEED_INPUT = "need_input"
ACTION_CALL_CAPABILITY = "call_capability"
ACTION_PROPOSE_SCHEDULE = "propose_schedule"
ACTION_PROPOSE_MONITOR = "propose_monitor"
ACTION_COMPLETE = "complete"
ACTION_FAIL = "fail"
ACTION_WAITING_FOR_CREDITS = "waiting_for_credits"

TERMINAL_ACTIONS = {ACTION_COMPLETE, ACTION_FAIL}

ALLOWED_ACTIONS = {
    ACTION_NEED_INPUT,
    ACTION_CALL_CAPABILITY,
    ACTION_PROPOSE_SCHEDULE,
    ACTION_PROPOSE_MONITOR,
    ACTION_COMPLETE,
    ACTION_FAIL,
    ACTION_WAITING_FOR_CREDITS,
}


@dataclass(frozen=True)
class CapabilitySpec:
    key: str
    name: str
    kind: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    side_effect_class: str = "none"
    requires_approval: bool = False
    mode: str = "sync"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "input_schema": dict(self.input_schema or {}),
            "output_schema": dict(self.output_schema or {}),
            "consumes": list(self.consumes or ()),
            "produces": list(self.produces or ()),
            "side_effect_class": self.side_effect_class,
            "requires_approval": self.requires_approval,
            "mode": self.mode,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class PersonaSpec:
    id: str
    name: str
    description: str = ""
    scope: str = "user"
    tags: tuple[str, ...] = ()
    instructions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "scope": self.scope,
            "tags": list(self.tags or ()),
            "instructions": dict(self.instructions or {}),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class Artifact:
    artifact_type: str
    key: str
    data: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    source_capability_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.artifact_type}:{self.key}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "key": self.key,
            "ref": self.ref,
            "data": dict(self.data or {}),
            "summary": self.summary,
            "source_capability_key": self.source_capability_key,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class AgentDecision:
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "payload": dict(self.payload or {}),
            "rationale": self.rationale,
            "plan": dict(self.plan or {}),
        }


@dataclass(frozen=True)
class AgentState:
    goal: str
    conversation: tuple[dict[str, Any], ...] = ()
    capabilities: tuple[CapabilitySpec, ...] = ()
    personas: tuple[PersonaSpec, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def capability_map(self) -> dict[str, CapabilitySpec]:
        mapping: dict[str, CapabilitySpec] = {}
        for capability in self.capabilities:
            mapping[capability.key] = capability
            mapping[capability.name] = capability
            metadata = dict(capability.metadata or {})
            for alias in list(metadata.get("aliases") or ()):
                alias_text = str(alias or "").strip()
                if alias_text:
                    mapping[alias_text] = capability
        return mapping
