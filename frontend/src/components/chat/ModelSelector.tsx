"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { ChevronDown, Cpu, Check, Sparkles, Plus, Minus } from "lucide-react";
import { ModelInfo, FinetunedModel } from "@/types";
import { listModels, getModelInfo, listFinetunedModels, registerFinetunedModel, unregisterFinetunedModel } from "@/lib/api";

// Model family detection from name
const MODEL_FAMILIES: Record<string, { pattern: RegExp; color: string }> = {
  SoloLLM: { pattern: /^solollm-/i, color: "#8b5cf6" },
  Llama: { pattern: /llama/i, color: "#3b82f6" },
  Qwen: { pattern: /qwen/i, color: "#a855f7" },
  Mistral: { pattern: /mistral/i, color: "#f97316" },
  Gemma: { pattern: /gemma/i, color: "#22c55e" },
  Phi: { pattern: /phi/i, color: "#06b6d4" },
  DeepSeek: { pattern: /deepseek/i, color: "#eab308" },
  TinyLlama: { pattern: /tinyllama/i, color: "#84cc16" },
  CodeLlama: { pattern: /codellama/i, color: "#ef4444" },
};

function getModelFamily(name: string): { family: string; color: string } {
  for (const [family, { pattern, color }] of Object.entries(MODEL_FAMILIES)) {
    if (pattern.test(name)) {
      return { family, color };
    }
  }
  return { family: "Other", color: "#6b7280" };
}

interface ModelSelectorProps {
  selectedModel: string;
  onSelectModel: (model: string) => void;
  onContextLengthChange?: (contextLength: number | null) => void;
}

export default function ModelSelector({
  selectedModel,
  onSelectModel,
  onContextLengthChange,
}: ModelSelectorProps) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [finetunedModels, setFinetunedModels] = useState<FinetunedModel[]>([]);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [contextLength, setContextLength] = useState<number | null>(null);
  const [registering, setRegistering] = useState<string | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const loadModels = useCallback(async () => {
    try {
      const [m, ft] = await Promise.all([
        listModels(),
        listFinetunedModels().catch(() => []),
      ]);
      setModels(m);
      setFinetunedModels(ft);
      setError(null);
    } catch {
      setError("Cannot connect to Ollama");
    }
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  // Fetch context length when selected model changes
  useEffect(() => {
    if (!selectedModel) return;
    getModelInfo(selectedModel)
      .then((info) => {
        setContextLength(info.context_length);
        onContextLengthChange?.(info.context_length);
      })
      .catch(() => {
        setContextLength(null);
        onContextLengthChange?.(null);
      });
  }, [selectedModel, onContextLengthChange]);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const formatSize = (bytes: number | null) => {
    if (!bytes) return "";
    const gb = bytes / (1024 * 1024 * 1024);
    if (gb >= 1) return `${gb.toFixed(1)} GB`;
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(0)} MB`;
  };

  const getModelTag = (model: ModelInfo) => {
    const size = model.parameter_size || formatSize(model.size);
    if (!size) return null;
    return size;
  };

  const formatContextLength = (ctx: number) => {
    if (ctx >= 1024) return `${Math.round(ctx / 1024)}K`;
    return `${ctx}`;
  };

  // Group models by family
  const groupedModels = useCallback(() => {
    const groups: Record<string, { models: ModelInfo[]; color: string }> = {};

    // Add SoloLLM fine-tuned models first (only registered ones)
    const registeredFinetuned = finetunedModels.filter((m) => m.is_registered);
    if (registeredFinetuned.length > 0 || finetunedModels.length > 0) {
      groups["SoloLLM"] = {
        models: [],
        color: MODEL_FAMILIES.SoloLLM.color,
      };
    }

    // Group regular Ollama models by family
    for (const model of models) {
      // Skip fine-tuned models from regular list (they're shown in SoloLLM)
      const isFinetuned = finetunedModels.some((ft) => ft.name === model.name);
      if (isFinetuned) {
        // Add to SoloLLM group
        groups["SoloLLM"].models.push(model);
        continue;
      }

      const { family, color } = getModelFamily(model.name);
      if (!groups[family]) {
        groups[family] = { models: [], color };
      }
      groups[family].models.push(model);
    }

    // Order: SoloLLM first, then alphabetical
    const ordered: [string, { models: ModelInfo[]; color: string }][] = [];
    if (groups["SoloLLM"]) {
      ordered.push(["SoloLLM", groups["SoloLLM"]]);
      delete groups["SoloLLM"];
    }
    ordered.push(...Object.entries(groups).sort((a, b) => a[0].localeCompare(b[0])));

    return ordered;
  }, [models, finetunedModels]);

  // Handle register/unregister fine-tuned model
  const handleToggleRegistration = async (model: FinetunedModel, e: React.MouseEvent) => {
    e.stopPropagation();
    setRegistering(model.name);
    try {
      if (model.is_registered) {
        await unregisterFinetunedModel(model.name);
      } else {
        await registerFinetunedModel(model.name);
      }
      await loadModels();
    } catch (err) {
      console.error("Failed to toggle registration:", err);
    } finally {
      setRegistering(null);
    }
  };

  // Get fine-tuned model info by name
  const getFinetunedInfo = (name: string): FinetunedModel | undefined => {
    return finetunedModels.find((m) => m.name === name);
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-2 rounded-xl text-sm transition-smooth"
        style={{
          background: "var(--bg-secondary)",
          border: "1px solid var(--border-color)",
          color: error ? "var(--error)" : "var(--text-primary)",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.borderColor = "var(--accent)";
          e.currentTarget.style.boxShadow = "0 0 12px rgba(99, 102, 241, 0.1)";
        }}
        onMouseLeave={(e) => {
          if (!open) {
            e.currentTarget.style.borderColor = "var(--border-color)";
            e.currentTarget.style.boxShadow = "none";
          }
        }}
      >
        <div
          className="w-5 h-5 rounded-md flex items-center justify-center"
          style={{
            background: error
              ? "rgba(248, 113, 113, 0.1)"
              : "var(--accent-muted)",
          }}
        >
          <Cpu
            size={12}
            style={{ color: error ? "var(--error)" : "var(--accent)" }}
          />
        </div>
        <span className="max-w-44 truncate font-medium">
          {error || selectedModel || "Select model"}
        </span>
        {contextLength && (
          <span
            className="text-[10px] px-1.5 py-0.5 rounded-md font-medium"
            style={{
              background: "var(--accent-muted)",
              color: "var(--accent)",
            }}
          >
            {formatContextLength(contextLength)} ctx
          </span>
        )}
        <ChevronDown
          size={14}
          className={`transition-transform duration-200 ${open ? "rotate-180" : ""}`}
          style={{ color: "var(--text-muted)" }}
        />
      </button>

      {open && (
        <div
          className="absolute top-full left-0 mt-2 min-w-72 max-h-96 overflow-y-auto rounded-xl py-1.5 z-50 animate-slideDown"
          style={{
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
            boxShadow:
              "0 12px 40px rgba(0, 0, 0, 0.4), 0 0 1px rgba(255, 255, 255, 0.05)",
          }}
        >
          {models.length === 0 && finetunedModels.length === 0 ? (
            <p
              className="px-4 py-3 text-sm"
              style={{ color: "var(--text-muted)" }}
            >
              {error || "No models found. Pull a model via Ollama first."}
            </p>
          ) : (
            <>
              {/* Show unregistered fine-tuned models first if any */}
              {finetunedModels.filter((m) => !m.is_registered).length > 0 && (
                <div className="px-3 py-2 border-b" style={{ borderColor: "var(--border-color)" }}>
                  <div className="flex items-center gap-2 mb-2">
                    <Sparkles size={12} style={{ color: "#8b5cf6" }} />
                    <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: "#8b5cf6" }}>
                      Available to Register
                    </span>
                  </div>
                  {finetunedModels.filter((m) => !m.is_registered).map((model) => (
                    <div
                      key={model.name}
                      className="flex items-center justify-between py-1.5 px-2 rounded-lg text-sm"
                      style={{ background: "var(--bg-hover)" }}
                    >
                      <div className="flex items-center gap-2">
                        <span style={{ color: "var(--text-secondary)" }}>{model.display_name}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: "rgba(139, 92, 246, 0.15)", color: "#8b5cf6" }}>
                          {model.base_model}
                        </span>
                      </div>
                      <button
                        onClick={(e) => handleToggleRegistration(model, e)}
                        disabled={registering === model.name}
                        className="p-1 rounded hover:bg-opacity-80 transition-colors"
                        style={{ background: "#8b5cf6", color: "white" }}
                        title="Register with Ollama"
                      >
                        {registering === model.name ? (
                          <span className="w-3 h-3 border border-white border-t-transparent rounded-full animate-spin inline-block" />
                        ) : (
                          <Plus size={12} />
                        )}
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Grouped models */}
              {groupedModels().map(([family, { models: familyModels, color }]) => (
                <div key={family}>
                  {/* Family header */}
                  <div className="px-3 pt-2 pb-1 flex items-center gap-2">
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{ background: color }}
                    />
                    <span
                      className="text-[10px] font-semibold uppercase tracking-wider"
                      style={{ color }}
                    >
                      {family}
                    </span>
                    {family === "SoloLLM" && (
                      <Sparkles size={10} style={{ color }} />
                    )}
                  </div>

                  {/* Models in this family */}
                  {familyModels.map((model) => {
                    const isSelected = selectedModel === model.name;
                    const tag = getModelTag(model);
                    const ftInfo = getFinetunedInfo(model.name);

                    return (
                      <div key={model.name} className="flex items-center">
                        <button
                          onClick={() => {
                            onSelectModel(model.name);
                            setOpen(false);
                          }}
                          className="flex-1 text-left px-4 py-2 text-sm flex items-center justify-between transition-smooth"
                          style={{
                            background: isSelected
                              ? "var(--accent-muted)"
                              : "transparent",
                            color: isSelected
                              ? "var(--text-primary)"
                              : "var(--text-secondary)",
                          }}
                          onMouseEnter={(e) => {
                            if (!isSelected)
                              e.currentTarget.style.background = "var(--bg-hover)";
                          }}
                          onMouseLeave={(e) => {
                            if (!isSelected)
                              e.currentTarget.style.background = "transparent";
                          }}
                        >
                          <div className="flex items-center gap-2.5">
                            {isSelected && (
                              <Check
                                size={14}
                                style={{ color: "var(--accent)" }}
                                className="shrink-0"
                              />
                            )}
                            <span
                              className={`truncate ${isSelected ? "font-medium" : ""}`}
                            >
                              {ftInfo?.display_name || model.name}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            {ftInfo && (
                              <span
                                className="text-[10px] px-1.5 py-0.5 rounded"
                                style={{ background: "rgba(139, 92, 246, 0.15)", color: "#8b5cf6" }}
                              >
                                {ftInfo.base_model}
                              </span>
                            )}
                            {tag && (
                              <span
                                className="text-[11px] shrink-0 px-2 py-0.5 rounded-md font-medium"
                                style={{
                                  color: "var(--accent)",
                                  background: "var(--accent-muted)",
                                }}
                              >
                                {tag}
                              </span>
                            )}
                          </div>
                        </button>
                        {ftInfo && (
                          <button
                            onClick={(e) => handleToggleRegistration(ftInfo, e)}
                            disabled={registering === ftInfo.name}
                            className="p-1.5 mr-2 rounded hover:bg-opacity-80 transition-colors"
                            style={{ background: "rgba(248, 113, 113, 0.1)", color: "#f87171" }}
                            title="Unregister from Ollama"
                          >
                            {registering === ftInfo.name ? (
                              <span className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin inline-block" />
                            ) : (
                              <Minus size={12} />
                            )}
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}
