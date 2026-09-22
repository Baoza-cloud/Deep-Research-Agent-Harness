"""Minimal harness contracts for agents, tools, context and compatibility."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from .schemas import utc_now


class RetrievalTool(Protocol):
    """Evidence retrieval contract shared by local, Web and paper backends."""

    async def search(
        self,
        query: str,
        limit: int = 5,
    ) -> Sequence[Mapping[str, Any]]: ...


class RetrievalToolAdapter:
    """Adapt a legacy backend or sync/async callable to ``RetrievalTool``."""

    def __init__(self, backend: Any | Callable[..., Any]):
        if not callable(backend) and not callable(getattr(backend, "search", None)):
            raise TypeError("Retrieval backend must be callable or expose search()")
        self.backend = backend

    async def search(
        self,
        query: str,
        limit: int = 5,
    ) -> Sequence[Mapping[str, Any]]:
        target = getattr(self.backend, "search", self.backend)
        if inspect.iscoroutinefunction(target):
            result = await target(query, limit)
        else:
            result = await asyncio.to_thread(target, query, limit)
            if inspect.isawaitable(result):
                result = await result
        if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
            raise TypeError("Retrieval backend must return a sequence of mappings")
        rows: list[Mapping[str, Any]] = []
        for row in result:
            if not isinstance(row, Mapping):
                raise TypeError("Retrieval result rows must be mappings")
            rows.append(row)
        return rows


class ToolRegistry:
    """Named capability registry; tools stay replaceable without agent changes."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, name: str, tool: Any, *, replace: bool = False) -> None:
        normalized = name.strip()
        if not normalized:
            raise ValueError("Tool name cannot be empty")
        if normalized in self._tools and not replace:
            raise ValueError(f"Tool already registered: {normalized}")
        self._tools[normalized] = tool

    def get(self, name: str) -> Any:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def contains(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))


@dataclass(frozen=True)
class AgentSpec:
    """Declarative identity and capability requirements for one agent role."""

    agent_id: str
    role: str
    description: str = ""
    tool_names: tuple[str, ...] = ()
    model: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("agent_id cannot be empty")
        if not self.role.strip():
            raise ValueError("role cannot be empty")
        if any(not item.strip() for item in self.tool_names):
            raise ValueError("tool_names cannot contain empty names")


@dataclass
class RunContext:
    """Isolated per-run state shared through explicit artifacts and events."""

    run_id: str
    objective: str
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    tools: ToolRegistry | None = None

    def emit(self, event: str, **payload: Any) -> None:
        self.events.append({"at": utc_now(), "event": event, **payload})

    def put_artifact(self, key: str, value: Any) -> None:
        if not key.strip():
            raise ValueError("Artifact key cannot be empty")
        self.artifacts[key] = value

    def get_artifact(self, key: str, default: Any = None) -> Any:
        return self.artifacts.get(key, default)

    def tool(self, name: str) -> Any:
        """Resolve a run-scoped tool for native Harness agents."""

        if self.tools is None:
            raise RuntimeError("No tool registry is bound to this run context")
        return self.tools.get(name)


class AgentHandler(Protocol):
    async def run(self, payload: Any, context: RunContext) -> Any: ...


class AgentRuntime(Protocol):
    async def invoke(
        self,
        spec: AgentSpec,
        handler: AgentHandler,
        payload: Any,
        context: RunContext,
    ) -> "AgentExecutionResult": ...


@dataclass
class AgentExecutionResult:
    agent_id: str
    output: Any
    started_at: str
    finished_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


class LegacyAgentAdapter:
    """Expose an existing ``run(payload)`` object through the Harness contract."""

    def __init__(self, agent: Any):
        if not callable(getattr(agent, "run", None)):
            raise TypeError("Legacy agent must expose run(payload)")
        self.agent = agent

    async def run(self, payload: Any, context: RunContext) -> Any:
        del context
        result = self.agent.run(payload)
        if inspect.isawaitable(result):
            return await result
        return result


class RoleAwareAgentAdapter(LegacyAgentAdapter):
    """Use an optional role-aware entrypoint while preserving legacy agents."""

    def __init__(self, agent: Any, spec: AgentSpec):
        super().__init__(agent)
        self.spec = spec

    async def run(self, payload: Any, context: RunContext) -> Any:
        target = getattr(self.agent, "run_for_agent", None)
        if not callable(target):
            return await super().run(payload, context)
        result = target(payload, self.spec, context)
        if inspect.isawaitable(result):
            return await result
        return result


class LocalAgentRuntime:
    """In-process runtime with lifecycle events and tool capability checks."""

    def __init__(self, tools: ToolRegistry | None = None):
        self.tools = tools or ToolRegistry()

    async def invoke(
        self,
        spec: AgentSpec,
        handler: AgentHandler,
        payload: Any,
        context: RunContext,
    ) -> AgentExecutionResult:
        if context.tools is None:
            context.tools = self.tools
        elif context.tools is not self.tools:
            raise RuntimeError("RunContext is bound to a different tool registry")
        missing = [name for name in spec.tool_names if not self.tools.contains(name)]
        if missing:
            raise RuntimeError(
                f"Agent {spec.agent_id} requires unregistered tools: {missing}"
            )
        started_at = utc_now()
        invocation_id = getattr(payload, "subtask_id", None)
        context.emit(
            "agent_started",
            agent_id=spec.agent_id,
            role=spec.role,
            invocation_id=invocation_id,
        )
        try:
            output = await handler.run(payload, context)
        except BaseException as exc:
            context.emit(
                "agent_failed",
                agent_id=spec.agent_id,
                role=spec.role,
                invocation_id=invocation_id,
                error_type=type(exc).__name__,
            )
            raise
        finished_at = utc_now()
        context.emit(
            "agent_completed",
            agent_id=spec.agent_id,
            role=spec.role,
            invocation_id=invocation_id,
        )
        return AgentExecutionResult(
            agent_id=spec.agent_id,
            output=output,
            started_at=started_at,
            finished_at=finished_at,
            metadata={"role": spec.role, "tools": list(spec.tool_names)},
        )
