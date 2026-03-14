"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  ArrowLeft,
  Bot,
  Send,
  Loader2,
  Brain,
  Wrench,
  ChevronDown,
  ChevronRight,
  Trash2,
  Plus,
  History,
  Lightbulb,
  Cog,
  Eye,
  AlertCircle,
  CheckCircle2,
  X,
  BookOpen,
  Clock,
} from "lucide-react";
import {
  listAgentTools,
  streamAgentRun,
  getAgentMemories,
  addAgentMemory,
  deleteAgentMemory,
  clearAgentMemories,
  getAgentRuns,
  listModels,
} from "@/lib/api";
import type { AgentTool, AgentMemory, AgentRun, ModelInfo } from "@/types";

interface AgentViewProps {
  onBack: () => void;
  selectedModel: string;
}

type Tab = "agent" | "tools" | "memory" | "history";

interface AgentStepDisplay {
  step: number;
  type: "thought" | "action" | "observation" | "answer" | "error";
  content: string;
  tool?: string;
  input?: Record<string, unknown>;
  elapsedMs?: number;
}

export default function AgentView({
  onBack,
  selectedModel,
}: AgentViewProps) {
  const [activeTab, setActiveTab] = useState<Tab>("agent");
  const [query, setQuery] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [steps, setSteps] = useState<AgentStepDisplay[]>([]);
  const [finalAnswer, setFinalAnswer] = useState("");
  const [toolsUsed, setToolsUsed] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isThinking, setIsThinking] = useState(false);
  const [thinkingStep, setThinkingStep] = useState(0);

  // Model selector
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedAgentModel, setSelectedAgentModel] = useState(selectedModel);
  const [reasoningModel, setReasoningModel] = useState<string>("");

  // Elapsed time tracking
  const [runStartTime, setRunStartTime] = useState<number | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const stepStartRef = useRef<number>(Date.now());

  // Tools tab
  const [tools, setTools] = useState<AgentTool[]>([]);
  const [toolsLoading, setToolsLoading] = useState(false);
  const [expandedTool, setExpandedTool] = useState<string | null>(null);

  // Memory tab
  const [memories, setMemories] = useState<AgentMemory[]>([]);
  const [memoriesLoading, setMemoriesLoading] = useState(false);
  const [newMemory, setNewMemory] = useState("");
  const [newMemoryCategory, setNewMemoryCategory] = useState("general");

  // History tab
  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const stepsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll steps
  useEffect(() => {
    stepsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [steps]);

  // Elapsed time timer
  useEffect(() => {
    if (!isRunning || !runStartTime) return;
    const interval = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - runStartTime) / 1000));
    }, 200);
    return () => clearInterval(interval);
  }, [isRunning, runStartTime]);

  // Load models and auto-select first available if current selection is missing
  useEffect(() => {
    listModels()
      .then((m) => {
        setModels(m);
        if (m.length > 0 && !m.some((mod) => mod.name === selectedAgentModel)) {
          setSelectedAgentModel(m[0].name);
        }
      })
      .catch(() => { });
  }, []);

  // Sync selectedModel prop
  useEffect(() => {
    setSelectedAgentModel(selectedModel);
  }, [selectedModel]);

  // Load tools
  const loadTools = useCallback(async () => {
    setToolsLoading(true);
    try {
      const data = await listAgentTools();
      setTools(data);
    } catch {
      // Backend may not be ready
    }
    setToolsLoading(false);
  }, []);

  // Load memories
  const loadMemories = useCallback(async () => {
    setMemoriesLoading(true);
    try {
      const data = await getAgentMemories();
      setMemories(data.memories);
    } catch {
      // ignore
    }
    setMemoriesLoading(false);
  }, []);

  // Load history
  const loadRuns = useCallback(async () => {
    setRunsLoading(true);
    try {
      const data = await getAgentRuns();
      setRuns(data.runs);
    } catch {
      // ignore
    }
    setRunsLoading(false);
  }, []);

  useEffect(() => {
    if (activeTab === "tools") loadTools();
    if (activeTab === "memory") loadMemories();
    if (activeTab === "history") loadRuns();
  }, [activeTab, loadTools, loadMemories, loadRuns]);

  // Run agent
  const handleRun = useCallback(() => {
    if (!query.trim() || isRunning) return;
    setIsRunning(true);
    setSteps([]);
    setFinalAnswer("");
    setToolsUsed([]);
    setError(null);
    setIsThinking(false);
    setThinkingStep(0);
    setRunStartTime(Date.now());
    setElapsedTime(0);
    stepStartRef.current = Date.now();

    const controller = streamAgentRun(
      {
        query: query.trim(),
        model: selectedAgentModel,
        max_steps: 10,
        reasoning_model: reasoningModel || undefined,
      },
      {
        onThinking: (step, _content) => {
          setIsThinking(true);
          setThinkingStep(step);
          stepStartRef.current = Date.now();
        },
        onThought: (step, content) => {
          const elapsed = Date.now() - stepStartRef.current;
          setIsThinking(false);
          setSteps((prev) => [
            ...prev,
            { step, type: "thought", content, elapsedMs: elapsed },
          ]);
        },
        onAction: (step, tool, input) => {
          const elapsed = Date.now() - stepStartRef.current;
          setIsThinking(false);
          setSteps((prev) => [
            ...prev,
            { step, type: "action", content: `Using ${tool}`, tool, input, elapsedMs: elapsed },
          ]);
        },
        onObservation: (step, content) => {
          const elapsed = Date.now() - stepStartRef.current;
          setSteps((prev) => [
            ...prev,
            { step, type: "observation", content, elapsedMs: elapsed },
          ]);
        },
        onAnswer: (content, _totalSteps, used) => {
          setIsRunning(false);
          setIsThinking(false);
          setFinalAnswer(content);
          setToolsUsed(used);
          setSteps((prev) => [
            ...prev,
            { step: 0, type: "answer", content },
          ]);
        },
        onError: (err) => {
          setIsRunning(false);
          setIsThinking(false);
          setError(err);
          setSteps((prev) => [
            ...prev,
            { step: 0, type: "error", content: err },
          ]);
        },
      }
    );
    abortRef.current = controller;
  }, [query, isRunning, selectedAgentModel, reasoningModel]);

  const handleStop = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      setIsRunning(false);
    }
  }, []);

  const handleAddMemory = async () => {
    if (!newMemory.trim()) return;
    try {
      await addAgentMemory(newMemory.trim(), newMemoryCategory);
      setNewMemory("");
      loadMemories();
    } catch {
      // ignore
    }
  };

  const handleDeleteMemory = async (id: string) => {
    try {
      await deleteAgentMemory(id);
      setMemories((prev) => prev.filter((m) => m.id !== id));
    } catch {
      // ignore
    }
  };

  const handleClearMemories = async () => {
    try {
      await clearAgentMemories();
      setMemories([]);
    } catch {
      // ignore
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleRun();
    }
  };

  const tabs: { key: Tab; label: string; icon: React.ReactNode }[] = [
    { key: "agent", label: "Agent", icon: <Bot size={15} /> },
    { key: "tools", label: "Tools", icon: <Wrench size={15} /> },
    { key: "memory", label: "Memory", icon: <Brain size={15} /> },
    { key: "history", label: "History", icon: <History size={15} /> },
  ];

  const getCategoryColor = (cat: string) => {
    const colors: Record<string, string> = {
      general: "var(--text-secondary)",
      fact: "var(--accent)",
      user_preference: "#a78bfa",
      task: "#34d399",
    };
    return colors[cat] || "var(--text-secondary)";
  };

  return (
    <div
      className="flex-1 flex flex-col h-screen overflow-hidden"
      style={{ background: "var(--bg-primary)" }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-3 px-4 py-3"
        style={{
          borderBottom: "1px solid var(--border-color)",
          background: "rgba(18, 18, 26, 0.5)",
          backdropFilter: "blur(12px)",
        }}
      >
        <button
          onClick={onBack}
          className="p-2 rounded-xl transition-smooth"
          style={{
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
            color: "var(--text-secondary)",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = "var(--accent)";
            e.currentTarget.style.color = "var(--text-primary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = "var(--border-color)";
            e.currentTarget.style.color = "var(--text-secondary)";
          }}
        >
          <ArrowLeft size={16} />
        </button>
        <div className="flex items-center gap-2">
          <Bot
            size={18}
            style={{ color: "var(--accent)" }}
          />
          <span
            className="text-lg font-semibold"
            style={{ color: "var(--text-primary)" }}
          >
            Agent Mode
          </span>
        </div>
        <span
          className="text-xs px-2 py-0.5 rounded-full"
          style={{
            background: "rgba(139, 92, 246, 0.1)",
            color: "var(--accent)",
            border: "1px solid rgba(139, 92, 246, 0.2)",
          }}
        >
          Phase 5
        </span>

        {/* Model Selectors */}
        <div className="flex items-center gap-2 ml-2">
          <select
            value={selectedAgentModel}
            onChange={(e) => setSelectedAgentModel(e.target.value)}
            className="text-xs px-2 py-1 rounded-lg bg-transparent outline-none"
            style={{
              color: "var(--text-secondary)",
              border: "1px solid var(--border-color)",
              background: "var(--bg-secondary)",
              maxWidth: 150,
            }}
            title="Agent model"
          >
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>
          <select
            value={reasoningModel}
            onChange={(e) => setReasoningModel(e.target.value)}
            className="text-xs px-2 py-1 rounded-lg bg-transparent outline-none"
            style={{
              color: "var(--text-muted)",
              border: "1px solid var(--border-color)",
              background: "var(--bg-secondary)",
              maxWidth: 150,
            }}
            title="Reasoning model (optional)"
          >
            <option value="">No reasoning model</option>
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 ml-auto">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-smooth"
              style={{
                background:
                  activeTab === tab.key
                    ? "var(--accent)"
                    : "var(--bg-secondary)",
                color:
                  activeTab === tab.key
                    ? "white"
                    : "var(--text-secondary)",
                border: `1px solid ${activeTab === tab.key
                    ? "var(--accent)"
                    : "var(--border-color)"
                  }`,
              }}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === "agent" && (
          <AgentTab
            query={query}
            setQuery={setQuery}
            isRunning={isRunning}
            steps={steps}
            finalAnswer={finalAnswer}
            toolsUsed={toolsUsed}
            error={error}
            onRun={handleRun}
            onStop={handleStop}
            onKeyDown={handleKeyDown}
            stepsEndRef={stepsEndRef}
            isThinking={isThinking}
            thinkingStep={thinkingStep}
            elapsedTime={elapsedTime}
          />
        )}
        {activeTab === "tools" && (
          <ToolsTab
            tools={tools}
            loading={toolsLoading}
            expandedTool={expandedTool}
            setExpandedTool={setExpandedTool}
          />
        )}
        {activeTab === "memory" && (
          <MemoryTab
            memories={memories}
            loading={memoriesLoading}
            newMemory={newMemory}
            setNewMemory={setNewMemory}
            newMemoryCategory={newMemoryCategory}
            setNewMemoryCategory={setNewMemoryCategory}
            onAdd={handleAddMemory}
            onDelete={handleDeleteMemory}
            onClear={handleClearMemories}
            getCategoryColor={getCategoryColor}
          />
        )}
        {activeTab === "history" && (
          <HistoryTab runs={runs} loading={runsLoading} />
        )}
      </div>
    </div>
  );
}

// ── Agent Tab ──────────────────────────────────────────────

function AgentTab({
  query,
  setQuery,
  isRunning,
  steps,
  finalAnswer,
  toolsUsed,
  error,
  onRun,
  onStop,
  onKeyDown,
  stepsEndRef,
  isThinking,
  thinkingStep,
  elapsedTime,
}: {
  query: string;
  setQuery: (q: string) => void;
  isRunning: boolean;
  steps: AgentStepDisplay[];
  finalAnswer: string;
  toolsUsed: string[];
  error: string | null;
  onRun: () => void;
  onStop: () => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  stepsEndRef: React.RefObject<HTMLDivElement | null>;
  isThinking: boolean;
  thinkingStep: number;
  elapsedTime: number;
}) {
  return (
    <div className="flex flex-col h-full">
      {/* Steps display */}
      <div className="flex-1 overflow-y-auto p-4">
        {steps.length === 0 && !isRunning && (
          <div className="flex flex-col items-center justify-center h-full gap-4 animate-fadeIn">
            <div
              className="w-16 h-16 rounded-2xl flex items-center justify-center"
              style={{
                background: "rgba(139, 92, 246, 0.1)",
                border: "1px solid rgba(139, 92, 246, 0.2)",
              }}
            >
              <Bot size={28} style={{ color: "var(--accent)" }} />
            </div>
            <div className="text-center max-w-md">
              <h3
                className="text-lg font-semibold mb-2"
                style={{ color: "var(--text-primary)" }}
              >
                ReAct Agent
              </h3>
              <p
                className="text-sm leading-relaxed"
                style={{ color: "var(--text-secondary)" }}
              >
                The agent reasons step-by-step, using tools like web search,
                document retrieval, code execution, and more to answer complex
                questions.
              </p>
            </div>
            <div className="flex flex-wrap gap-2 mt-2">
              {["Search the web for recent news about AI", "What documents do I have about machine learning?", "Calculate the compound interest on $10,000 at 5% for 10 years"].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => setQuery(suggestion)}
                  className="text-xs px-3 py-2 rounded-xl transition-smooth"
                  style={{
                    background: "var(--bg-secondary)",
                    color: "var(--text-secondary)",
                    border: "1px solid var(--border-color)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "var(--accent)";
                    e.currentTarget.style.color = "var(--text-primary)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "var(--border-color)";
                    e.currentTarget.style.color = "var(--text-secondary)";
                  }}
                >
                  <Lightbulb
                    size={11}
                    style={{ display: "inline", marginRight: 4, verticalAlign: "middle" }}
                  />
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {steps.length > 0 && (
          <div className="max-w-3xl mx-auto space-y-3 animate-fadeIn">
            {steps.map((step, idx) => (
              <StepCard key={idx} step={step} />
            ))}
            {isRunning && isThinking && (
              <div className="flex items-center gap-2 px-4 py-3" style={{ color: "var(--text-muted)" }}>
                <Brain size={14} className="animate-pulse" style={{ color: "var(--accent)" }} />
                <span className="text-xs">Thinking...</span>
                <span className="text-[10px]">{elapsedTime}s</span>
              </div>
            )}
            {isRunning && !isThinking && (
              <div className="flex items-center gap-2 p-3 rounded-xl" style={{ background: "var(--bg-secondary)" }}>
                <Loader2 size={14} className="animate-spin" style={{ color: "var(--accent)" }} />
                <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                  Agent is working...
                </span>
                <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>{elapsedTime}s</span>
              </div>
            )}
            <div ref={stepsEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div
        className="p-4"
        style={{ borderTop: "1px solid var(--border-color)" }}
      >
        <div className="max-w-3xl mx-auto">
          {toolsUsed.length > 0 && finalAnswer && (
            <div
              className="flex items-center gap-2 mb-2 text-xs"
              style={{ color: "var(--text-secondary)" }}
            >
              <Wrench size={12} />
              Tools used: {toolsUsed.join(", ")}
            </div>
          )}
          <div className="flex gap-2">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask the agent a complex question..."
              rows={1}
              className="flex-1 px-4 py-3 rounded-xl text-sm resize-none transition-smooth"
              style={{
                background: "var(--bg-secondary)",
                color: "var(--text-primary)",
                border: "1px solid var(--border-color)",
                outline: "none",
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = "var(--accent)";
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = "var(--border-color)";
              }}
              disabled={isRunning}
            />
            {isRunning ? (
              <button
                onClick={onStop}
                className="px-4 py-3 rounded-xl text-sm font-medium transition-smooth flex items-center gap-2"
                style={{
                  background: "rgba(248, 113, 113, 0.15)",
                  color: "var(--error)",
                  border: "1px solid rgba(248, 113, 113, 0.3)",
                }}
              >
                <X size={16} />
                Stop
              </button>
            ) : (
              <button
                onClick={onRun}
                disabled={!query.trim()}
                className="px-4 py-3 rounded-xl text-sm font-medium transition-smooth flex items-center gap-2"
                style={{
                  background: query.trim()
                    ? "var(--accent)"
                    : "var(--bg-secondary)",
                  color: query.trim() ? "white" : "var(--text-muted)",
                  border: `1px solid ${query.trim() ? "var(--accent)" : "var(--border-color)"
                    }`,
                  opacity: query.trim() ? 1 : 0.5,
                }}
              >
                <Send size={16} />
                Run
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Step Card ──────────────────────────────────────────────

function StepCard({ step }: { step: AgentStepDisplay }) {
  const [expanded, setExpanded] = useState(step.type !== "observation");

  const config: Record<
    string,
    { icon: React.ReactNode; color: string; bg: string; label: string }
  > = {
    thought: {
      icon: <Lightbulb size={13} />,
      color: "#facc15",
      bg: "rgba(250, 204, 21, 0.08)",
      label: "Thinking",
    },
    action: {
      icon: <Wrench size={13} />,
      color: "var(--accent)",
      bg: "rgba(139, 92, 246, 0.08)",
      label: "Tool Call",
    },
    observation: {
      icon: <Eye size={13} />,
      color: "#38bdf8",
      bg: "rgba(56, 189, 248, 0.08)",
      label: "Observation",
    },
    answer: {
      icon: <CheckCircle2 size={13} />,
      color: "#34d399",
      bg: "rgba(52, 211, 153, 0.08)",
      label: "Final Answer",
    },
    error: {
      icon: <AlertCircle size={13} />,
      color: "var(--error)",
      bg: "rgba(248, 113, 113, 0.08)",
      label: "Error",
    },
  };

  const { icon, color, bg, label } = config[step.type] || config.thought;

  return (
    <div
      className="rounded-xl overflow-hidden transition-smooth"
      style={{
        background: bg,
        border: `1px solid ${color}22`,
      }}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2.5 text-left"
        style={{ color }}
      >
        {icon}
        <span className="text-xs font-semibold uppercase tracking-wide">
          {step.step > 0 ? `Step ${step.step} — ` : ""}
          {label}
        </span>
        {step.tool && (
          <span
            className="text-xs px-2 py-0.5 rounded-full ml-1"
            style={{
              background: "rgba(139, 92, 246, 0.15)",
              color: "var(--accent)",
            }}
          >
            {step.tool}
          </span>
        )}
        {step.elapsedMs !== undefined && step.elapsedMs > 0 && (
          <span
            className="text-[10px] ml-1 flex items-center gap-0.5"
            style={{ color: "var(--text-muted)" }}
          >
            <Clock size={10} />
            {(step.elapsedMs / 1000).toFixed(1)}s
          </span>
        )}
        <span className="ml-auto">
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
      </button>
      {expanded && (
        <div
          className="px-3 pb-3 text-sm whitespace-pre-wrap"
          style={{ color: "var(--text-primary)" }}
        >
          {step.content}
          {step.input && Object.keys(step.input).length > 0 && (
            <div
              className="mt-2 p-2 rounded-lg text-xs font-mono"
              style={{
                background: "rgba(0,0,0,0.2)",
                color: "var(--text-secondary)",
              }}
            >
              {JSON.stringify(step.input, null, 2)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Tools Tab ──────────────────────────────────────────────

function ToolsTab({
  tools,
  loading,
  expandedTool,
  setExpandedTool,
}: {
  tools: AgentTool[];
  loading: boolean;
  expandedTool: string | null;
  setExpandedTool: (t: string | null) => void;
}) {
  const categories = [...new Set(tools.map((t) => t.category))];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2
          size={20}
          className="animate-spin"
          style={{ color: "var(--accent)" }}
        />
      </div>
    );
  }

  return (
    <div className="p-4 max-w-3xl mx-auto">
      <h3
        className="text-sm font-semibold mb-4"
        style={{ color: "var(--text-primary)" }}
      >
        Available Tools ({tools.length})
      </h3>
      {categories.map((cat) => (
        <div key={cat} className="mb-4">
          <div
            className="text-xs font-medium uppercase tracking-wide mb-2 px-1"
            style={{ color: "var(--text-muted)" }}
          >
            {cat}
          </div>
          <div className="space-y-2">
            {tools
              .filter((t) => t.category === cat)
              .map((tool) => (
                <div
                  key={tool.name}
                  className="rounded-xl transition-smooth"
                  style={{
                    background: "var(--bg-secondary)",
                    border: "1px solid var(--border-color)",
                  }}
                >
                  <button
                    onClick={() =>
                      setExpandedTool(
                        expandedTool === tool.name ? null : tool.name
                      )
                    }
                    className="w-full flex items-center gap-2 px-3 py-2.5 text-left"
                  >
                    <Cog size={13} style={{ color: "var(--accent)" }} />
                    <span
                      className="text-sm font-medium"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {tool.name}
                    </span>
                    {tool.is_dangerous && (
                      <span
                        className="text-[10px] px-1.5 py-0.5 rounded-full"
                        style={{
                          background: "rgba(248, 113, 113, 0.1)",
                          color: "var(--error)",
                        }}
                      >
                        dangerous
                      </span>
                    )}
                    <span
                      className="text-xs ml-auto"
                      style={{ color: "var(--text-muted)" }}
                    >
                      {expandedTool === tool.name ? (
                        <ChevronDown size={14} />
                      ) : (
                        <ChevronRight size={14} />
                      )}
                    </span>
                  </button>
                  {expandedTool === tool.name && (
                    <div className="px-3 pb-3 space-y-2">
                      <p
                        className="text-xs"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {tool.description}
                      </p>
                      {tool.parameters.length > 0 && (
                        <div>
                          <div
                            className="text-[10px] font-semibold uppercase mb-1"
                            style={{ color: "var(--text-muted)" }}
                          >
                            Parameters
                          </div>
                          {tool.parameters.map((p) => (
                            <div
                              key={p.name}
                              className="flex items-center gap-2 text-xs py-0.5"
                            >
                              <code
                                className="px-1.5 py-0.5 rounded"
                                style={{
                                  background: "rgba(139, 92, 246, 0.1)",
                                  color: "var(--accent)",
                                  fontSize: "11px",
                                }}
                              >
                                {p.name}
                              </code>
                              <span
                                className="text-[10px]"
                                style={{ color: "var(--text-muted)" }}
                              >
                                {p.type}
                                {p.required ? " (required)" : ""}
                              </span>
                              <span
                                style={{ color: "var(--text-secondary)" }}
                              >
                                — {p.description}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Memory Tab ─────────────────────────────────────────────

function MemoryTab({
  memories,
  loading,
  newMemory,
  setNewMemory,
  newMemoryCategory,
  setNewMemoryCategory,
  onAdd,
  onDelete,
  onClear,
  getCategoryColor,
}: {
  memories: AgentMemory[];
  loading: boolean;
  newMemory: string;
  setNewMemory: (v: string) => void;
  newMemoryCategory: string;
  setNewMemoryCategory: (v: string) => void;
  onAdd: () => void;
  onDelete: (id: string) => void;
  onClear: () => void;
  getCategoryColor: (cat: string) => string;
}) {
  return (
    <div className="p-4 max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <h3
          className="text-sm font-semibold"
          style={{ color: "var(--text-primary)" }}
        >
          <Brain
            size={14}
            style={{
              display: "inline",
              marginRight: 6,
              verticalAlign: "middle",
              color: "var(--accent)",
            }}
          />
          Agent Memory ({memories.length})
        </h3>
        {memories.length > 0 && (
          <button
            onClick={onClear}
            className="text-xs px-2 py-1 rounded-lg transition-smooth flex items-center gap-1"
            style={{
              color: "var(--error)",
              background: "rgba(248, 113, 113, 0.08)",
              border: "1px solid rgba(248, 113, 113, 0.15)",
            }}
          >
            <Trash2 size={11} />
            Clear All
          </button>
        )}
      </div>

      <p
        className="text-xs mb-4"
        style={{ color: "var(--text-secondary)" }}
      >
        Agent memories persist across sessions. The agent can store and recall
        facts, preferences, and context automatically during runs.
      </p>

      {/* Add memory */}
      <div
        className="flex gap-2 mb-4 p-3 rounded-xl"
        style={{
          background: "var(--bg-secondary)",
          border: "1px solid var(--border-color)",
        }}
      >
        <input
          value={newMemory}
          onChange={(e) => setNewMemory(e.target.value)}
          placeholder="Add a memory for the agent..."
          className="flex-1 bg-transparent text-sm outline-none"
          style={{ color: "var(--text-primary)" }}
          onKeyDown={(e) => {
            if (e.key === "Enter") onAdd();
          }}
        />
        <select
          value={newMemoryCategory}
          onChange={(e) => setNewMemoryCategory(e.target.value)}
          className="text-xs px-2 py-1 rounded-lg bg-transparent outline-none"
          style={{
            color: "var(--text-secondary)",
            border: "1px solid var(--border-color)",
          }}
        >
          <option value="general">general</option>
          <option value="fact">fact</option>
          <option value="user_preference">preference</option>
          <option value="task">task</option>
        </select>
        <button
          onClick={onAdd}
          disabled={!newMemory.trim()}
          className="px-3 py-1 rounded-lg text-xs font-medium transition-smooth"
          style={{
            background: newMemory.trim()
              ? "var(--accent)"
              : "var(--bg-hover)",
            color: newMemory.trim() ? "white" : "var(--text-muted)",
          }}
        >
          <Plus size={14} />
        </button>
      </div>

      {/* Memory list */}
      {loading ? (
        <div className="flex items-center justify-center h-32">
          <Loader2
            size={20}
            className="animate-spin"
            style={{ color: "var(--accent)" }}
          />
        </div>
      ) : memories.length === 0 ? (
        <div
          className="text-center py-12 text-sm"
          style={{ color: "var(--text-muted)" }}
        >
          No memories yet. The agent will store facts here automatically.
        </div>
      ) : (
        <div className="space-y-2">
          {memories.map((m) => (
            <div
              key={m.id}
              className="flex items-start gap-3 p-3 rounded-xl group transition-smooth"
              style={{
                background: "var(--bg-secondary)",
                border: "1px solid var(--border-color)",
              }}
            >
              <BookOpen
                size={14}
                className="mt-0.5 shrink-0"
                style={{ color: getCategoryColor(m.category) }}
              />
              <div className="flex-1 min-w-0">
                <p
                  className="text-sm"
                  style={{ color: "var(--text-primary)" }}
                >
                  {m.content}
                </p>
                <div className="flex items-center gap-2 mt-1">
                  <span
                    className="text-[10px] px-1.5 py-0.5 rounded-full"
                    style={{
                      background: `${getCategoryColor(m.category)}15`,
                      color: getCategoryColor(m.category),
                    }}
                  >
                    {m.category}
                  </span>
                  <span
                    className="text-[10px]"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {m.created_at?.slice(0, 10)}
                  </span>
                </div>
              </div>
              <button
                onClick={() => onDelete(m.id)}
                className="opacity-0 group-hover:opacity-100 p-1 rounded-lg transition-smooth"
                style={{ color: "var(--text-muted)" }}
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── History Tab ────────────────────────────────────────────

function HistoryTab({
  runs,
  loading,
}: {
  runs: AgentRun[];
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2
          size={20}
          className="animate-spin"
          style={{ color: "var(--accent)" }}
        />
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div
        className="text-center py-12 text-sm"
        style={{ color: "var(--text-muted)" }}
      >
        No agent runs yet. Try running a query in the Agent tab.
      </div>
    );
  }

  return (
    <div className="p-4 max-w-3xl mx-auto">
      <h3
        className="text-sm font-semibold mb-4"
        style={{ color: "var(--text-primary)" }}
      >
        <History
          size={14}
          style={{
            display: "inline",
            marginRight: 6,
            verticalAlign: "middle",
            color: "var(--accent)",
          }}
        />
        Recent Agent Runs ({runs.length})
      </h3>
      <div className="space-y-2">
        {runs.map((run) => {
          let parsedTools: string[] = [];
          try {
            parsedTools = JSON.parse(run.tools_used);
          } catch {
            // ignore
          }

          return (
            <div
              key={run.id}
              className="p-3 rounded-xl transition-smooth"
              style={{
                background: "var(--bg-secondary)",
                border: "1px solid var(--border-color)",
              }}
            >
              <div className="flex items-center gap-2 mb-1">
                {run.success ? (
                  <CheckCircle2
                    size={13}
                    style={{ color: "#34d399" }}
                  />
                ) : (
                  <AlertCircle
                    size={13}
                    style={{ color: "var(--error)" }}
                  />
                )}
                <span
                  className="text-sm font-medium truncate flex-1"
                  style={{ color: "var(--text-primary)" }}
                >
                  {run.query}
                </span>
                <span
                  className="text-[10px] shrink-0"
                  style={{ color: "var(--text-muted)" }}
                >
                  {run.total_steps} steps
                </span>
              </div>
              <p
                className="text-xs line-clamp-2"
                style={{ color: "var(--text-secondary)" }}
              >
                {run.answer}
              </p>
              <div className="flex items-center gap-2 mt-1.5">
                {parsedTools.map((tool) => (
                  <span
                    key={tool}
                    className="text-[10px] px-1.5 py-0.5 rounded-full"
                    style={{
                      background: "rgba(139, 92, 246, 0.1)",
                      color: "var(--accent)",
                    }}
                  >
                    {tool}
                  </span>
                ))}
                <span
                  className="text-[10px] ml-auto"
                  style={{ color: "var(--text-muted)" }}
                >
                  {run.created_at?.slice(0, 16).replace("T", " ")}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
