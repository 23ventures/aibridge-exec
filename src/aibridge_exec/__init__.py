"""AIBridge execution layer.

Stateless building blocks shared with aibridge-platform:

- ``aibridge_exec.contracts.tools``: tool/capability contracts (``ToolDefinition``,
  ``CapabilityDescriptor``, ``ToolExecutionContext``).
- ``aibridge_exec.contracts.agent``: agent decision and state contracts.
- ``aibridge_exec.code_runtime``: local stateful Python execution runtime.

This package must not import Flask, SQLAlchemy, Redis, Kafka, Celery or the
platform (enforced by ``lint-imports``).
"""

__version__ = "0.1.0"
