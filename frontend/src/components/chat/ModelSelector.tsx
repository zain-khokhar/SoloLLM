"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { ChevronDown, Cpu, Check } from "lucide-react";
import { ModelInfo } from "@/types";
import { listModels, getModelInfo } from "@/lib/api";

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
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [contextLength, setContextLength] = useState<number | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const m = await listModels();
        setModels(m);
        setError(null);
      } catch {
        setError("Cannot connect to Ollama");
      }
    };
    load();
  }, []);

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
          className="absolute top-full left-0 mt-2 min-w-64 rounded-xl py-1.5 z-50 animate-slideDown"
          style={{
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
            boxShadow:
              "0 12px 40px rgba(0, 0, 0, 0.4), 0 0 1px rgba(255, 255, 255, 0.05)",
          }}
        >
          {models.length === 0 ? (
            <p
              className="px-4 py-3 text-sm"
              style={{ color: "var(--text-muted)" }}
            >
              {error || "No models found. Pull a model via Ollama first."}
            </p>
          ) : (
            models.map((model) => {
              const isSelected = selectedModel === model.name;
              const tag = getModelTag(model);

              return (
                <button
                  key={model.name}
                  onClick={() => {
                    onSelectModel(model.name);
                    setOpen(false);
                  }}
                  className="w-full text-left px-4 py-2.5 text-sm flex items-center justify-between transition-smooth"
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
                      {model.name}
                    </span>
                  </div>
                  {tag && (
                    <span
                      className="text-[11px] ml-3 shrink-0 px-2 py-0.5 rounded-md font-medium"
                      style={{
                        color: "var(--accent)",
                        background: "var(--accent-muted)",
                      }}
                    >
                      {tag}
                    </span>
                  )}
                </button>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}
