# aibridge-exec

The stateless execution layer of AIBridge. It has no database, no Flask and no queues
(enforced by `lint-imports`), so it can be used, tested and eventually open-sourced on its own.

## Contents (v0.1.0)

| Module | What it is |
| --- | --- |
| `aibridge_exec.contracts.tools` | Tool contracts: `ToolDefinition`, `CapabilityDescriptor`, `ToolExecutionContext` |
| `aibridge_exec.contracts.agent` | Agent contracts: `AgentDecision`, `AgentState`, `CapabilitySpec`, actions |
| `aibridge_exec.code_runtime` | Stateful local Python execution runtime (sessions, packages, artifacts) |

More moves here as the platform's `aibridge/execution` package loses its database and
framework dependencies: provider adapters, the agent loop, pure tools, file transforms.

## Install

```bash
pip install -e ".[dev]"       # development
pip install -e ../aibridge-exec   # from aibridge-platform
```

## Configuration

The code runtime reads settings from the environment. A host application can supply its own
configuration mapping as well:

```python
from aibridge_exec.code_runtime.settings import set_config_provider

set_config_provider(lambda: my_app_config)   # zero-argument callable returning a mapping
```

`aibridge-platform` registers a provider that returns the Flask app config inside an app
context, which keeps its previous behaviour.

## Checks

```bash
lint-imports    # no Flask, Werkzeug, SQLAlchemy, Redis, Kafka, Celery or platform imports
pytest --cov
```

## License

To be decided before the repository is made public.
