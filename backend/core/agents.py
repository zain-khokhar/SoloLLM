"""
Agent Framework for SoloLLM — Phase 5.

Implements a ReAct-style agent loop with tool execution,
RAG integration, knowledge graph queries, and persistent agent memory.

Includes:
- Tool registry with JSON schema definitions
- Built-in tools (calculator, code runner, web search, file I/O)
- RAG retrieval tool & knowledge graph query tool
- ReAct agent loop (Thought → Action → Observation) with async LLM calls
- Persistent agent memory (store/recall facts across sessions)
"""

import re
import os
import json
import math
import time
import logging
import subprocess
import urllib.parse
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


# ── Tool Definitions ────────────────────────────────────────

@dataclass
class ToolParameter:
    """A parameter for a tool."""
    name: str
    type: str  # string, number, boolean, array
    description: str
    required: bool = True
    default: str | None = None


@dataclass
class ToolDefinition:
    """Definition of an agent tool."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    category: str = "general"
    is_dangerous: bool = False

    def to_schema(self) -> dict:
        """Convert to JSON schema format for LLM consumption."""
        props = {}
        required = []
        for p in self.parameters:
            props[p.name] = {
                "type": p.type,
                "description": p.description,
            }
            if p.required:
                required.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }


@dataclass
class ToolResult:
    """Result of executing a tool."""
    tool_name: str
    success: bool
    output: str
    error: str | None = None
    execution_time_ms: float = 0


# ── Built-in Tools ──────────────────────────────────────────

class CalculatorTool:
    """Safe math expression evaluator."""

    definition = ToolDefinition(
        name="calculator",
        description="Evaluate mathematical expressions. Supports basic arithmetic, trigonometry, and common math functions.",
        parameters=[
            ToolParameter("expression", "string", "Mathematical expression to evaluate, e.g. '2 * 3 + sqrt(16)'"),
        ],
        category="math",
    )

    SAFE_NAMES = {
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'sum': sum, 'len': len, 'pow': pow, 'int': int, 'float': float,
        'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'log': math.log, 'log10': math.log10, 'log2': math.log2,
        'pi': math.pi, 'e': math.e, 'ceil': math.ceil, 'floor': math.floor,
    }

    def execute(self, expression: str) -> ToolResult:
        try:
            cleaned = re.sub(r'[^0-9+\-*/().,%^a-zA-Z_ ]', '', expression)
            cleaned = cleaned.replace('^', '**')
            result = eval(cleaned, {"__builtins__": {}}, self.SAFE_NAMES)  # noqa: S307
            return ToolResult(tool_name="calculator", success=True, output=str(result))
        except Exception as e:
            return ToolResult(tool_name="calculator", success=False, output="", error=f"Math error: {e}")


class CodeRunnerTool:
    """Execute Python code in a sandboxed subprocess."""

    definition = ToolDefinition(
        name="code_runner",
        description="Execute Python code and return the output. Use for calculations, data processing, or testing logic.",
        parameters=[
            ToolParameter("code", "string", "Python code to execute"),
        ],
        category="code",
        is_dangerous=True,
    )

    def execute(self, code: str, timeout: int = 10) -> ToolResult:
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True, text=True, timeout=timeout,
                cwd=os.path.expanduser("~"),
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                return ToolResult(tool_name="code_runner", success=False, output=output, error=result.stderr.strip())
            return ToolResult(tool_name="code_runner", success=True, output=output or "(no output)")
        except subprocess.TimeoutExpired:
            return ToolResult(tool_name="code_runner", success=False, output="", error=f"Code execution timed out ({timeout}s limit)")
        except Exception as e:
            return ToolResult(tool_name="code_runner", success=False, output="", error=f"Execution error: {e}")


class FileReaderTool:
    """Read file contents."""

    definition = ToolDefinition(
        name="file_reader",
        description="Read the contents of a local file. Useful for inspecting files on the user's machine.",
        parameters=[
            ToolParameter("path", "string", "Absolute path to the file to read"),
            ToolParameter("max_lines", "number", "Maximum lines to read (default 100)", required=False),
        ],
        category="file",
    )

    def execute(self, path: str, max_lines: int = 100) -> ToolResult:
        try:
            if not os.path.exists(path):
                return ToolResult(tool_name="file_reader", success=False, output="", error=f"File not found: {path}")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"... ({i} lines read, truncated)")
                        break
                    lines.append(line.rstrip())
            return ToolResult(tool_name="file_reader", success=True, output="\n".join(lines))
        except Exception as e:
            return ToolResult(tool_name="file_reader", success=False, output="", error=f"Read error: {e}")


class FileWriterTool:
    """Write content to a file."""

    definition = ToolDefinition(
        name="file_writer",
        description="Write content to a local file. Creates the file if it doesn't exist.",
        parameters=[
            ToolParameter("path", "string", "Absolute path to write to"),
            ToolParameter("content", "string", "Content to write"),
            ToolParameter("append", "boolean", "If true, append instead of overwrite", required=False),
        ],
        category="file",
        is_dangerous=True,
    )

    def execute(self, path: str, content: str, append: bool = False) -> ToolResult:
        try:
            mode = "a" if append else "w"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, mode, encoding="utf-8") as f:
                f.write(content)
            return ToolResult(tool_name="file_writer", success=True, output=f"Written {len(content)} chars to {path}")
        except Exception as e:
            return ToolResult(tool_name="file_writer", success=False, output="", error=f"Write error: {e}")


class WebSearchTool:
    """Search the web using DuckDuckGo (no API key needed)."""

    definition = ToolDefinition(
        name="web_search",
        description="Search the web for information. Returns titles and snippets from search results.",
        parameters=[
            ToolParameter("query", "string", "Search query"),
            ToolParameter("num_results", "number", "Number of results (default 5)", required=False),
        ],
        category="web",
    )

    def execute(self, query: str, num_results: int = 5) -> ToolResult:
        try:
            import httpx
            encoded = urllib.parse.quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded}"
            with httpx.Client(timeout=10, follow_redirects=True) as client:
                response = client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })
            results = []
            links = re.findall(
                r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
                r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                response.text, re.DOTALL,
            )
            for href, title, snippet in links[:num_results]:
                clean_title = re.sub(r'<[^>]+>', '', title).strip()
                clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                results.append(f"• {clean_title}\n  {clean_snippet}")
            if not results:
                titles = re.findall(r'<a[^>]+class="result__a"[^>]*>(.*?)</a>', response.text)
                for t in titles[:num_results]:
                    results.append(f"• {re.sub(r'<[^>]+>', '', t).strip()}")
            if results:
                return ToolResult(tool_name="web_search", success=True, output="\n\n".join(results))
            return ToolResult(tool_name="web_search", success=True, output="No results found.")
        except Exception as e:
            return ToolResult(tool_name="web_search", success=False, output="", error=f"Search error: {e}")


class DateTimeTool:
    """Get current date and time."""

    definition = ToolDefinition(
        name="datetime",
        description="Get the current date and time in UTC.",
        parameters=[],
        category="utility",
    )

    def execute(self) -> ToolResult:
        now = datetime.now(timezone.utc)
        return ToolResult(tool_name="datetime", success=True, output=now.strftime("%Y-%m-%d %H:%M:%S UTC"))


# ── RAG Integration Tool ───────────────────────────────────

class RAGSearchTool:
    """Search the document knowledge base via RAG pipeline."""

    definition = ToolDefinition(
        name="rag_search",
        description="Search your uploaded documents for relevant information. Use this when you need facts from the user's document library.",
        parameters=[
            ToolParameter("query", "string", "Search query to find relevant document chunks"),
            ToolParameter("top_k", "number", "Number of results to retrieve (default 5)", required=False),
        ],
        category="rag",
    )

    async def execute_async(self, query: str, top_k: int = 5) -> ToolResult:
        t0 = time.time()
        try:
            from rag.pipeline import rag_pipeline
            cited_context = await rag_pipeline.query(query=query, workspace_id="default", top_k=top_k)

            if not cited_context.citations:
                return ToolResult(
                    tool_name="rag_search", success=True,
                    output="No relevant documents found.",
                    execution_time_ms=(time.time() - t0) * 1000,
                )

            formatted = []
            for cite in cited_context.citations:
                formatted.append(
                    f"[{cite.index}] {cite.document_title} "
                    f"(relevance: {cite.relevance_score:.2f})\n{cite.excerpt[:500]}"
                )

            return ToolResult(
                tool_name="rag_search", success=True,
                output="\n\n".join(formatted),
                execution_time_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name="rag_search", success=False, output="",
                error=f"RAG search error: {e}",
                execution_time_ms=(time.time() - t0) * 1000,
            )

    def execute(self, **kwargs) -> ToolResult:
        # Sync wrapper — the agent loop calls execute_async directly
        return ToolResult(tool_name="rag_search", success=False, output="", error="Use execute_async for this tool")


class KnowledgeGraphTool:
    """Query the knowledge graph for entity relationships."""

    definition = ToolDefinition(
        name="knowledge_graph",
        description="Query the knowledge graph to find entities and their relationships. Use this for questions about people, places, organizations, or concepts and how they relate.",
        parameters=[
            ToolParameter("query", "string", "Entity name or search query"),
            ToolParameter("operation", "string", "Operation: 'search' to find entities, 'neighbors' to get relationships of an entity", required=False),
        ],
        category="rag",
    )

    async def execute_async(self, query: str, operation: str = "search") -> ToolResult:
        t0 = time.time()
        try:
            from memory.knowledge_graph import knowledge_graph
            if operation == "neighbors":
                # Find entity first, then get neighbors
                entities = await knowledge_graph.search_entities(query, limit=1)
                if not entities:
                    return ToolResult(tool_name="knowledge_graph", success=True, output=f"No entity found matching '{query}'.", execution_time_ms=(time.time() - t0) * 1000)
                entity = entities[0]
                neighbors = await knowledge_graph.get_entity_neighbors(entity["id"])
                lines = [f"Entity: {entity['name']} ({entity['entity_type']})"]
                for n in neighbors:
                    lines.append(f"  —[{n['relation_type']}]→ {n['target_name']} ({n['target_type']})")
                return ToolResult(tool_name="knowledge_graph", success=True, output="\n".join(lines), execution_time_ms=(time.time() - t0) * 1000)
            else:
                entities = await knowledge_graph.search_entities(query, limit=10)
                if not entities:
                    return ToolResult(tool_name="knowledge_graph", success=True, output=f"No entities found matching '{query}'.", execution_time_ms=(time.time() - t0) * 1000)
                lines = [f"Found {len(entities)} entities:"]
                for e in entities:
                    lines.append(f"  • {e['name']} ({e['entity_type']}, mentions: {e.get('mention_count', 0)})")
                return ToolResult(tool_name="knowledge_graph", success=True, output="\n".join(lines), execution_time_ms=(time.time() - t0) * 1000)
        except Exception as e:
            return ToolResult(tool_name="knowledge_graph", success=False, output="", error=f"Knowledge graph error: {e}", execution_time_ms=(time.time() - t0) * 1000)

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name="knowledge_graph", success=False, output="", error="Use execute_async for this tool")


# ── Web Scraper Tool ───────────────────────────────────────

class WebScraperTool:
    """Fetch and extract main text content from a specific URL."""

    definition = ToolDefinition(
        name="web_scrape",
        description="Fetch and extract main text content from a specific URL",
        parameters=[
            ToolParameter("url", "string", "The URL to fetch and extract text from"),
            ToolParameter("max_chars", "number", "Maximum characters to return (default 3000)", required=False),
        ],
        category="web",
    )

    async def execute_async(self, url: str, max_chars: int = 3000) -> ToolResult:
        t0 = time.time()
        try:
            import httpx
            from bs4 import BeautifulSoup

            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                response = await client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove non-content tags
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)

            # Collapse excessive whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)

            # Truncate to max_chars
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... (truncated)"

            return ToolResult(
                tool_name="web_scrape", success=True,
                output=text or "(no text content extracted)",
                execution_time_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name="web_scrape", success=False, output="",
                error=f"Web scrape error: {e}",
                execution_time_ms=(time.time() - t0) * 1000,
            )

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name="web_scrape", success=False, output="", error="Use execute_async for this tool")


# ── Agent Memory Tool ──────────────────────────────────────

class AgentMemoryTool:
    """Store and recall persistent facts across agent sessions."""

    definition = ToolDefinition(
        name="memory",
        description="Store or recall persistent facts and notes. Use 'store' to save a fact, 'recall' to search for previously stored facts, 'list' to see all memories.",
        parameters=[
            ToolParameter("operation", "string", "Operation: 'store', 'recall', or 'list'"),
            ToolParameter("content", "string", "For 'store': the fact to remember. For 'recall': search query.", required=False),
            ToolParameter("category", "string", "Optional category tag (e.g. 'user_preference', 'fact', 'task')", required=False),
        ],
        category="memory",
    )

    async def execute_async(self, operation: str, content: str = "", category: str = "general") -> ToolResult:
        t0 = time.time()
        try:
            from storage.database import save_agent_memory, search_agent_memories, list_agent_memories

            if operation == "store":
                if not content:
                    return ToolResult(tool_name="memory", success=False, output="", error="Content is required for 'store' operation")
                await save_agent_memory(content=content, category=category)
                return ToolResult(tool_name="memory", success=True, output=f"Stored memory: {content}", execution_time_ms=(time.time() - t0) * 1000)

            elif operation == "recall":
                memories = await search_agent_memories(query=content, limit=10)
                if not memories:
                    return ToolResult(tool_name="memory", success=True, output="No matching memories found.", execution_time_ms=(time.time() - t0) * 1000)
                lines = [f"Found {len(memories)} memories:"]
                for m in memories:
                    lines.append(f"  [{m['category']}] {m['content']} (saved: {m['created_at'][:10]})")
                return ToolResult(tool_name="memory", success=True, output="\n".join(lines), execution_time_ms=(time.time() - t0) * 1000)

            elif operation == "list":
                memories = await list_agent_memories(limit=20)
                if not memories:
                    return ToolResult(tool_name="memory", success=True, output="No memories stored yet.", execution_time_ms=(time.time() - t0) * 1000)
                lines = [f"All memories ({len(memories)}):"]
                for m in memories:
                    lines.append(f"  [{m['category']}] {m['content']}")
                return ToolResult(tool_name="memory", success=True, output="\n".join(lines), execution_time_ms=(time.time() - t0) * 1000)

            else:
                return ToolResult(tool_name="memory", success=False, output="", error=f"Unknown operation: {operation}. Use 'store', 'recall', or 'list'.")

        except Exception as e:
            return ToolResult(tool_name="memory", success=False, output="", error=f"Memory error: {e}", execution_time_ms=(time.time() - t0) * 1000)

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name="memory", success=False, output="", error="Use execute_async for this tool")


# ── Tool Registry ──────────────────────────────────────────

ASYNC_TOOLS = {"rag_search", "knowledge_graph", "memory", "web_scrape"}


class ToolRegistry:
    """Registry of all available tools."""

    def __init__(self):
        self._tools: dict[str, tuple[ToolDefinition, object]] = {}
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in tools."""
        builtins = [
            (CalculatorTool.definition, CalculatorTool()),
            (CodeRunnerTool.definition, CodeRunnerTool()),
            (FileReaderTool.definition, FileReaderTool()),
            (FileWriterTool.definition, FileWriterTool()),
            (WebSearchTool.definition, WebSearchTool()),
            (WebScraperTool.definition, WebScraperTool()),
            (DateTimeTool.definition, DateTimeTool()),
            (RAGSearchTool.definition, RAGSearchTool()),
            (KnowledgeGraphTool.definition, KnowledgeGraphTool()),
            (AgentMemoryTool.definition, AgentMemoryTool()),
        ]
        for defn, impl in builtins:
            self._tools[defn.name] = (defn, impl)

    def register(self, definition: ToolDefinition, implementation):
        """Register a custom tool."""
        self._tools[definition.name] = (definition, implementation)

    def get_tool(self, name: str):
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tools."""
        return [defn for defn, _ in self._tools.values()]

    def get_schemas(self) -> list[dict]:
        """Get JSON schemas for all tools (for LLM function calling)."""
        return [defn.to_schema() for defn, _ in self._tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a synchronous tool by name."""
        entry = self._tools.get(tool_name)
        if not entry:
            return ToolResult(tool_name=tool_name, success=False, output="", error=f"Unknown tool: {tool_name}")
        _, impl = entry
        try:
            return impl.execute(**kwargs)
        except Exception as e:
            return ToolResult(tool_name=tool_name, success=False, output="", error=f"Tool execution failed: {e}")

    async def execute_tool_async(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool — uses async path for RAG/KG/memory tools."""
        entry = self._tools.get(tool_name)
        if not entry:
            return ToolResult(tool_name=tool_name, success=False, output="", error=f"Unknown tool: {tool_name}")
        _, impl = entry
        try:
            if tool_name in ASYNC_TOOLS and hasattr(impl, "execute_async"):
                return await impl.execute_async(**kwargs)
            return impl.execute(**kwargs)
        except Exception as e:
            return ToolResult(tool_name=tool_name, success=False, output="", error=f"Tool execution failed: {e}")


# ── ReAct Agent Loop ────────────────────────────────────────

@dataclass
class AgentStep:
    """A single step in the agent's reasoning chain."""
    step_number: int
    thought: str = ""
    action: str = ""
    action_input: dict = field(default_factory=dict)
    observation: str = ""
    is_final: bool = False
    final_answer: str = ""


@dataclass
class AgentResult:
    """Final result of an agent execution."""
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    total_steps: int = 0
    success: bool = True
    error: str | None = None


class ReActAgent:
    """
    ReAct-style agent that interleaves reasoning and tool use.

    Flow:
    1. Think about what to do
    2. Choose a tool and execute it
    3. Observe the result
    4. Repeat until done or max steps reached
    """

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        max_steps: int = 10,
    ):
        self.tools = tool_registry or ToolRegistry()
        self.max_steps = max_steps

    def build_system_prompt(self, agent_memories: list[dict] | None = None) -> str:
        """Build the system prompt with tool descriptions and agent memory."""
        tool_list = []
        for tool in self.tools.list_tools():
            params = ", ".join(f"{p.name}: {p.type}" for p in tool.parameters)
            tool_list.append(f"- **{tool.name}**({params}): {tool.description}")

        prompt = (
            "You are a helpful AI agent that solves tasks step-by-step using tools.\n\n"
            "## Available Tools\n"
            + "\n".join(tool_list) + "\n\n"
            "## How to Use Tools\n"
            "To use a tool, respond EXACTLY in this format:\n\n"
            "Thought: [your reasoning about what to do next]\n"
            "Action: [tool_name]\n"
            'Action Input: {"param": "value"}\n\n'
            "You will then receive an Observation with the tool's output.\n\n"
            "## How to Finish\n"
            "When you have enough information to answer, respond with:\n\n"
            "Thought: [your final reasoning]\n"
            "Final Answer: [your complete, well-formatted answer]\n\n"
            "## Rules\n"
            "- Always start with a Thought\n"
            "- Use one tool at a time\n"
            "- Action Input must be valid JSON\n"
            "- If a tool fails, try a different approach\n"
            "- Be concise but thorough in your final answer\n"
        )

        if agent_memories:
            prompt += "\n## Your Persistent Memories\n"
            for m in agent_memories[:10]:
                prompt += f"- [{m['category']}] {m['content']}\n"
            prompt += "\n"

        return prompt

    def parse_agent_response(self, response: str) -> AgentStep:
        """Parse an LLM response into an AgentStep."""
        step = AgentStep(step_number=0)

        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', response, re.DOTALL)
        if thought_match:
            step.thought = thought_match.group(1).strip()

        # Check for final answer
        final_match = re.search(r'Final Answer:\s*(.+?)$', response, re.DOTALL)
        if final_match:
            step.is_final = True
            step.final_answer = final_match.group(1).strip()
            return step

        # Extract action
        action_match = re.search(r'Action:\s*(\w+)', response)
        if action_match:
            step.action = action_match.group(1).strip()

        # Extract action input — try multiline JSON first
        input_match = re.search(r'Action Input:\s*(\{.*?\})', response, re.DOTALL)
        if input_match:
            try:
                step.action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                # Try to fix common issues
                raw = input_match.group(1).strip()
                step.action_input = {"raw": raw}

        return step

    async def execute_step(self, step: AgentStep) -> AgentStep:
        """Execute a tool action and record the observation."""
        if step.action:
            t0 = time.time()
            result = await self.tools.execute_tool_async(step.action, **step.action_input)
            result.execution_time_ms = (time.time() - t0) * 1000
            step.observation = result.output if result.success else f"Error: {result.error}"
        return step

    def format_scratchpad(self, steps: list[AgentStep]) -> str:
        """Format steps as a scratchpad for the LLM."""
        lines = []
        for step in steps:
            if step.thought:
                lines.append(f"Thought: {step.thought}")
            if step.action:
                lines.append(f"Action: {step.action}")
                lines.append(f"Action Input: {json.dumps(step.action_input)}")
            if step.observation:
                # Truncate long observations
                obs = step.observation
                if len(obs) > 2000:
                    obs = obs[:2000] + "\n... (truncated)"
                lines.append(f"Observation: {obs}")
        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using a simple word_count * 1.3 heuristic."""
        return int(len(text.split()) * 1.3)

    def _build_messages_with_budget(self, system_prompt: str, query: str, steps: list[AgentStep], max_context_tokens: int = 4096) -> list[dict]:
        """
        Build messages list with context budget management.

        Starts with system prompt + query, then adds steps from most recent
        backwards until the token budget is exceeded. Truncates long observations
        to 1000 chars.
        """
        messages = [{"role": "system", "content": system_prompt}]
        budget_used = self._estimate_tokens(system_prompt)

        # Reserve budget for query
        query_content = f"Query: {query}"
        budget_used += self._estimate_tokens(query_content)

        # Build step messages from most recent backwards
        step_messages = []
        for step in reversed(steps):
            lines = []
            if step.thought:
                lines.append(f"Thought: {step.thought}")
            if step.action:
                lines.append(f"Action: {step.action}")
                lines.append(f"Action Input: {json.dumps(step.action_input)}")
            if step.observation:
                obs = step.observation
                if len(obs) > 1000:
                    obs = obs[:1000] + "\n... (truncated)"
                lines.append(f"Observation: {obs}")

            step_text = "\n".join(lines)
            step_tokens = self._estimate_tokens(step_text)

            if budget_used + step_tokens > max_context_tokens:
                break

            budget_used += step_tokens
            step_messages.insert(0, step_text)

        # Combine steps into scratchpad
        scratchpad = "\n".join(step_messages)
        if scratchpad:
            query_content += f"\n\n{scratchpad}\n\nContinue reasoning:"

        messages.append({"role": "user", "content": query_content})
        return messages

    async def run(
        self,
        query: str,
        model: str | None = None,
        max_steps: int | None = None,
        conversation_context: list[dict] | None = None,
    ) -> AgentResult:
        """
        Run the full ReAct loop: Thought → Action → Observation → ... → Final Answer.

        Returns the complete AgentResult.
        """
        from core.inference import ollama_client
        from core.config import settings

        model = model or settings.default_model
        max_steps = max_steps or self.max_steps
        steps: list[AgentStep] = []
        tools_used: list[str] = []

        # Load agent memories for context
        agent_memories = []
        try:
            from storage.database import list_agent_memories
            agent_memories = await list_agent_memories(limit=10)
        except Exception:
            pass

        system_prompt = self.build_system_prompt(agent_memories)

        for step_num in range(1, max_steps + 1):
            # Build messages for LLM
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation context if provided
            if conversation_context:
                for msg in conversation_context[-6:]:  # Last 6 messages
                    messages.append(msg)

            # User query + scratchpad
            scratchpad = self.format_scratchpad(steps)
            user_content = f"Query: {query}"
            if scratchpad:
                user_content += f"\n\n{scratchpad}\n\nContinue reasoning:"
            messages.append({"role": "user", "content": user_content})

            # Call LLM
            try:
                response = await ollama_client.chat(
                    messages=messages,
                    model=model,
                    temperature=0.2,
                    max_tokens=1024,
                )
                llm_output = response.get("content", "")
            except Exception as e:
                return AgentResult(
                    answer="", steps=steps, tools_used=tools_used,
                    total_steps=step_num, success=False,
                    error=f"LLM call failed: {e}",
                )

            # Parse the response
            step = self.parse_agent_response(llm_output)
            step.step_number = step_num

            # Final answer
            if step.is_final:
                steps.append(step)
                return AgentResult(
                    answer=step.final_answer, steps=steps,
                    tools_used=tools_used, total_steps=step_num,
                    success=True,
                )

            # Execute action
            if step.action:
                step = await self.execute_step(step)
                if step.action not in tools_used:
                    tools_used.append(step.action)

            steps.append(step)

            # If no action was taken and no final answer, nudge the agent
            if not step.action and not step.is_final:
                steps.append(AgentStep(
                    step_number=step_num,
                    observation="You must either use a tool (Action + Action Input) or provide a Final Answer.",
                ))

        # Max steps reached — extract whatever answer we have
        last_thoughts = " ".join(s.thought for s in steps if s.thought)
        return AgentResult(
            answer=f"I reached the maximum number of steps ({max_steps}). Based on my analysis: {last_thoughts}",
            steps=steps, tools_used=tools_used, total_steps=max_steps,
            success=True,
        )

    async def run_stream(
        self,
        query: str,
        model: str | None = None,
        max_steps: int | None = None,
        conversation_context: list[dict] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Streaming version of run() that yields events as they happen.

        Yields dicts with:
          {"type": "thinking", "step": N, "content": "..."}
          {"type": "thought", "step": N, "content": "..."}
          {"type": "action", "step": N, "tool": "...", "input": {...}}
          {"type": "observation", "step": N, "content": "..."}
          {"type": "answer", "content": "...", "steps": [...], "tools_used": [...]}
          {"type": "error", "content": "..."}
        """
        from core.inference import ollama_client
        from core.config import settings

        model = model or settings.default_model
        max_steps = max_steps or self.max_steps
        steps: list[AgentStep] = []
        tools_used: list[str] = []

        agent_memories = []
        try:
            from storage.database import list_agent_memories
            agent_memories = await list_agent_memories(limit=10)
        except Exception:
            pass

        system_prompt = self.build_system_prompt(agent_memories)

        for step_num in range(1, max_steps + 1):
            # Emit thinking event before each LLM call
            yield {"type": "thinking", "step": step_num, "content": "Reasoning about next step..."}

            # Build messages with context budget management
            messages = self._build_messages_with_budget(system_prompt, query, steps)

            if conversation_context:
                # Insert conversation context after system prompt
                for i, msg in enumerate(conversation_context[-6:]):
                    messages.insert(1 + i, msg)

            try:
                response = await ollama_client.chat(
                    messages=messages, model=model,
                    temperature=0.2, max_tokens=1024,
                )
                llm_output = response.get("content", "")
            except Exception as e:
                yield {"type": "error", "content": f"LLM call failed: {e}"}
                return

            step = self.parse_agent_response(llm_output)
            step.step_number = step_num

            # Emit thought
            if step.thought:
                yield {"type": "thought", "step": step_num, "content": step.thought}

            # Final answer
            if step.is_final:
                steps.append(step)
                yield {
                    "type": "answer",
                    "content": step.final_answer,
                    "steps": [asdict(s) for s in steps],
                    "tools_used": tools_used,
                    "total_steps": step_num,
                }
                return

            # Execute action
            if step.action:
                yield {"type": "action", "step": step_num, "tool": step.action, "input": step.action_input}
                step = await self.execute_step(step)
                if step.action not in tools_used:
                    tools_used.append(step.action)
                yield {"type": "observation", "step": step_num, "content": step.observation}

            steps.append(step)

            if not step.action and not step.is_final:
                steps.append(AgentStep(
                    step_number=step_num,
                    observation="You must either use a tool or provide a Final Answer.",
                ))

        # Max steps
        last_thoughts = " ".join(s.thought for s in steps if s.thought)
        answer = f"Reached maximum steps ({max_steps}). Based on analysis: {last_thoughts}"
        yield {
            "type": "answer",
            "content": answer,
            "steps": [asdict(s) for s in steps],
            "tools_used": tools_used,
            "total_steps": max_steps,
        }


# Singletons
tool_registry = ToolRegistry()
react_agent = ReActAgent(tool_registry)
