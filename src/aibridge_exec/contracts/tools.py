from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class CapabilityDescriptor:
    key: str
    type: str
    name: str
    display_name: str
    description: str
    mode: str
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "type": self.type,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "mode": self.mode,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            **(self.metadata or {}),
        }


@dataclass(frozen=True)
class ToolExecutionContext:
    run_id: str
    session_id: str
    team_id: str
    user_id: str
    step_id: str | None = None
    execution_id: str | None = None
    attempt_index: int | None = None
    artifact_key: str | None = None
    external_capability_job_id: str | None = None


@dataclass(frozen=True)
class EffectDescriptor:
    kind: str = "read"
    channel: str | None = None
    requires_approval: bool = False
    supports_implicit_grant: bool = False
    risk: str = "low"
    metadata: dict = field(default_factory=dict)

    def is_side_effect(self) -> bool:
        return str(self.kind or "").strip().lower() not in {
            "",
            "none",
            "read",
            "read_only",
        }

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "channel": self.channel,
            "requires_approval": self.requires_approval,
            "supports_implicit_grant": self.supports_implicit_grant,
            "risk": self.risk,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class DeliveryDescriptor:
    modes: tuple[str, ...] = ("immediate",)
    metadata: dict = field(default_factory=dict)

    def supports(self, mode: str) -> bool:
        return str(mode or "").strip().lower() in {
            str(item or "").strip().lower()
            for item in self.modes
            if str(item or "").strip()
        }

    def to_dict(self) -> dict:
        return {
            "modes": list(self.modes or ()),
            "metadata": dict(self.metadata or {}),
        }


ToolNormalizeFn = Callable[[ToolExecutionContext, dict], dict]
ToolValidateFn = Callable[[ToolExecutionContext, dict], None]
ToolPreviewFn = Callable[[ToolExecutionContext, dict], dict]
ToolGrantScopeFn = Callable[[ToolExecutionContext, dict], dict]
ToolFingerprintFn = Callable[[ToolExecutionContext, dict], str]
ToolInputSuggestionsFn = Callable[[ToolExecutionContext], dict[str, Any]]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolDefinition:
    capability: CapabilityDescriptor
    execute: Callable[[ToolExecutionContext, dict], dict]
    transport: str = "builtin"
    side_effect_class: str = "read"
    requires_confirmation: bool = False
    requires_grounding: bool = False
    grounding_kind: str | None = None
    required_integrations: tuple[str, ...] = ()
    auth_requirements: dict = field(default_factory=dict)
    effect: EffectDescriptor | None = None
    delivery: DeliveryDescriptor = field(default_factory=DeliveryDescriptor)
    normalize: ToolNormalizeFn | None = None
    validate: ToolValidateFn | None = None
    render_approval_preview: ToolPreviewFn | None = None
    build_grant_scope: ToolGrantScopeFn | None = None
    fingerprint: ToolFingerprintFn | None = None
    suggest_inputs: ToolInputSuggestionsFn | None = None

    availability_check: Callable[[], bool] | None = None

    def is_available(self, connected_integrations: set[str] | None = None) -> bool:
        if self.availability_check is not None and not self.availability_check():
            return False
        if not self.required_integrations:
            return True
        connected = connected_integrations or set()
        return any(name in connected for name in self.required_integrations)

    def effect_descriptor(self) -> EffectDescriptor:
        if self.effect is not None:
            return self.effect
        kind = str(self.side_effect_class or "read").strip() or "read"
        normalized_kind = kind.lower()
        is_side_effect = normalized_kind not in {"", "none", "read", "read_only"}
        return EffectDescriptor(
            kind=kind,
            requires_approval=bool(self.requires_confirmation or is_side_effect),
            risk="medium" if is_side_effect else "low",
            metadata={"source": "legacy_tool_definition"},
        )

    def normalize_arguments(self, context: ToolExecutionContext, arguments: dict) -> dict:
        payload = dict(arguments or {})
        if self.normalize is None:
            return payload
        return dict(self.normalize(context, payload) or {})

    def validate_arguments(self, context: ToolExecutionContext, arguments: dict):
        if self.validate is not None:
            self.validate(context, dict(arguments or {}))

    def input_suggestions(self, context: ToolExecutionContext) -> dict[str, Any]:
        if self.suggest_inputs is None:
            return {}
        return dict(self.suggest_inputs(context) or {})

    def approval_preview(self, context: ToolExecutionContext, arguments: dict) -> dict:
        if self.render_approval_preview is not None:
            return dict(self.render_approval_preview(context, dict(arguments or {})) or {})
        return {
            "tool": self.capability.display_name or self.capability.name,
            "effect": self.effect_descriptor().to_dict(),
            "arguments": dict(arguments or {}),
        }

    def grant_scope_for_arguments(
        self,
        context: ToolExecutionContext,
        arguments: dict,
    ) -> dict:
        payload = dict(arguments or {})
        if self.build_grant_scope is not None:
            return dict(self.build_grant_scope(context, payload) or {})
        return {
            "scope_type": "exact_call",
            "tool_name": self.capability.name,
            "capability_key": self.capability.key,
            "arguments_hash": _stable_hash(payload),
            "effect": self.effect_descriptor().to_dict(),
        }

    def side_effect_fingerprint(
        self,
        context: ToolExecutionContext,
        arguments: dict,
    ) -> str:
        payload = dict(arguments or {})
        if self.fingerprint is not None:
            return str(self.fingerprint(context, payload))
        return _stable_hash(
            {
                "capability_key": self.capability.key,
                "tool_name": self.capability.name,
                "effect_kind": self.effect_descriptor().kind,
                "arguments": payload,
            }
        )

    def to_catalog_item(
        self,
        *,
        authorized: bool,
        available: bool,
    ) -> dict:
        return {
            **self.capability.to_dict(),
            "transport": self.transport,
            "side_effect_class": self.side_effect_class,
            "requires_confirmation": self.requires_confirmation,
            "requires_grounding": self.requires_grounding,
            "grounding_kind": self.grounding_kind,
            "required_integrations": list(self.required_integrations or ()),
            "auth_requirements": dict(self.auth_requirements or {}),
            "effect": self.effect_descriptor().to_dict(),
            "delivery": self.delivery.to_dict(),
            "available": available,
            "authorized": authorized,
        }


# orchestrator -> planner -> executor -> [input]
# skills (pdf to csv) use
# tools (internet, mcp tools (slack, github))
# subagents (Complex)
