"""
Agent API for SoloLLM — Phase 5.

Provides endpoints for listing tools, executing tools,
running the ReAct agent loop (streaming SSE), and managing agent memory.
"""

import json
import logging
from fastapi import APIRouter
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from core.agents import tool_registry, react_agent
from storage.database import (
    save_agent_run, list_agent_runs,
    list_agent_memories, save_agent_memory, delete_agent_memory, clear_agent_memories,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["Agent"])


# ── Request / Response Models ──────────────────────────────

class ToolExecuteRequest(BaseModel):
    tool_name: str
    arguments: dict = {}


class AgentRunRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model: str | None = None
    max_steps: int = 10
    reasoning_model: str | None = None


class AgentMemoryCreate(BaseModel):
    content: str = Field(..., min_length=1)
    category: str = "general"


# ── Tool Endpoints ─────────────────────────────────────────

@router.get("/tools")
async def list_tools():
    """List all available agent tools."""
    tools = tool_registry.list_tools()
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "is_dangerous": t.is_dangerous,
                "parameters": [
                    {"name": p.name, "type": p.type, "description": p.description, "required": p.required}
                    for p in t.parameters
                ],
            }
            for t in tools
        ]
    }


@router.post("/execute")
async def execute_tool(request: ToolExecuteRequest):
    """Execute a single tool directly."""
    result = await tool_registry.execute_tool_async(request.tool_name, **request.arguments)
    return {
        "tool_name": result.tool_name,
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "execution_time_ms": result.execution_time_ms,
    }


@router.get("/schemas")
async def get_tool_schemas():
    """Get JSON schemas for all tools (for LLM function calling)."""
    return {"schemas": tool_registry.get_schemas()}


@router.get("/system-prompt")
async def get_agent_system_prompt():
    """Get the agent's system prompt with tool descriptions."""
    return {"prompt": react_agent.build_system_prompt()}


# ── Agent Run Endpoints ────────────────────────────────────

@router.post("/run")
async def run_agent(request: AgentRunRequest):
    """Run the agent (non-streaming). Returns the full result."""
    from core.config import settings
    model = request.model or settings.default_model

    result = await react_agent.run(
        query=request.query,
        model=model,
        max_steps=request.max_steps,
    )

    # Persist the run
    await save_agent_run(
        query=request.query,
        answer=result.answer,
        model=model,
        total_steps=result.total_steps,
        tools_used=result.tools_used,
        steps_json=json.dumps([
            {"step": s.step_number, "thought": s.thought, "action": s.action,
             "action_input": s.action_input, "observation": s.observation,
             "is_final": s.is_final, "final_answer": s.final_answer}
            for s in result.steps
        ]),
        success=result.success,
        error=result.error,
    )

    return {
        "answer": result.answer,
        "success": result.success,
        "error": result.error,
        "total_steps": result.total_steps,
        "tools_used": result.tools_used,
        "steps": [
            {"step": s.step_number, "thought": s.thought, "action": s.action,
             "action_input": s.action_input, "observation": s.observation,
             "is_final": s.is_final}
            for s in result.steps
        ],
    }


@router.post("/run/stream")
async def run_agent_stream(request: AgentRunRequest):
    """Run the agent with streaming SSE events."""
    from core.config import settings
    import httpx

    model = request.model or settings.default_model

    # Verify model exists in Ollama before starting the agent run
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            if resp.status_code == 200:
                available = [m["name"] for m in resp.json().get("models", [])]
                if model not in available and available:
                    # Requested model not found — fall back to first available
                    model = available[0]
    except Exception:
        pass  # If we can't check, proceed and let the agent report the error

    async def event_generator():
        steps_collected = []
        tools_used = []
        final_answer = ""

        try:
            async for event in react_agent.run_stream(
                query=request.query,
                model=model,
                max_steps=request.max_steps,
            ):
                event_type = event.get("type", "")

                if event_type == "thinking":
                    yield {
                        "event": "thinking",
                        "data": json.dumps({"step": event["step"], "content": event["content"]}),
                    }

                elif event_type == "thought":
                    yield {
                        "event": "thought",
                        "data": json.dumps({"step": event["step"], "content": event["content"]}),
                    }

                elif event_type == "action":
                    yield {
                        "event": "action",
                        "data": json.dumps({
                            "step": event["step"],
                            "tool": event["tool"],
                            "input": event["input"],
                        }),
                    }

                elif event_type == "observation":
                    yield {
                        "event": "observation",
                        "data": json.dumps({"step": event["step"], "content": event["content"]}),
                    }

                elif event_type == "answer":
                    final_answer = event["content"]
                    tools_used = event.get("tools_used", [])
                    steps_collected = event.get("steps", [])
                    yield {
                        "event": "answer",
                        "data": json.dumps({
                            "content": final_answer,
                            "total_steps": event.get("total_steps", 0),
                            "tools_used": tools_used,
                        }),
                    }

                elif event_type == "error":
                    yield {
                        "event": "error",
                        "data": json.dumps({"content": event["content"]}),
                    }

            # Persist the run
            await save_agent_run(
                query=request.query,
                answer=final_answer,
                model=model,
                total_steps=len(steps_collected),
                tools_used=tools_used,
                steps_json=json.dumps(steps_collected),
                success=bool(final_answer),
                error=None,
            )

        except Exception as e:
            logger.error(f"Agent stream error: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"content": str(e)}),
            }

    return EventSourceResponse(event_generator())


@router.get("/runs")
async def get_agent_runs(limit: int = 20):
    """List recent agent runs."""
    runs = await list_agent_runs(limit=limit)
    return {"runs": runs}


# ── Agent Memory Endpoints ─────────────────────────────────

@router.get("/memory")
async def get_agent_memory(category: str | None = None, limit: int = 50):
    """List agent memories."""
    memories = await list_agent_memories(limit=limit, category=category)
    return {"memories": memories, "count": len(memories)}


@router.post("/memory")
async def add_agent_memory(request: AgentMemoryCreate):
    """Add a new agent memory."""
    memory = await save_agent_memory(content=request.content, category=request.category)
    return {"memory": memory}


@router.delete("/memory/{memory_id}")
async def remove_agent_memory(memory_id: str):
    """Delete a specific agent memory."""
    deleted = await delete_agent_memory(memory_id)
    if not deleted:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True}


@router.delete("/memory")
async def clear_all_agent_memory():
    """Clear all agent memories."""
    count = await clear_agent_memories()
    return {"cleared": count}
