"use client";

import { useState, useEffect } from "react";
import {
  Settings,
  Database,
  Cpu,
  Layers,
  Sliders,
  Save,
  CheckCircle2,
  X,
  Zap,
  HardDrive,
  Brain,
  Crosshair,
} from "lucide-react";
import { ThreadSettings } from "@/types";
import { getThreadSettings, updateThreadSettings } from "@/lib/api";

interface ThreadSettingsPanelProps {
  threadId: string | null;
  onClose: () => void;
}

export default function ThreadSettingsPanel({
  threadId,
  onClose,
}: ThreadSettingsPanelProps) {
  const [settings, setSettings] = useState<ThreadSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (!threadId) return;
    setLoading(true);
    getThreadSettings(threadId)
      .then((s) => setSettings(s))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [threadId]);

  const handleSave = async () => {
    if (!threadId || !settings) return;
    setSaving(true);
    try {
      const updated = await updateThreadSettings(threadId, settings);
      setSettings(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch {
      // ignore
    }
    setSaving(false);
  };

  const updateField = <K extends keyof ThreadSettings>(
    key: K,
    value: ThreadSettings[K]
  ) => {
    if (!settings) return;
    setSettings({ ...settings, [key]: value });
  };

  if (!threadId) return null;

  return (
    <div
      className="absolute right-0 top-0 h-full w-80 z-20 overflow-y-auto animate-slideInRight"
      style={{
        background: "var(--bg-secondary)",
        borderLeft: "1px solid var(--border-color)",
        boxShadow: "-4px 0 24px rgba(0,0,0,0.3)",
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 sticky top-0"
        style={{
          background: "var(--bg-secondary)",
          borderBottom: "1px solid var(--border-color)",
        }}
      >
        <div className="flex items-center gap-2">
          <Settings size={14} style={{ color: "var(--accent)" }} />
          <span
            className="text-sm font-semibold"
            style={{ color: "var(--text-primary)" }}
          >
            Thread Settings
          </span>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg transition-smooth"
          style={{ color: "var(--text-muted)" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--bg-hover)";
            e.currentTarget.style.color = "var(--text-primary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "var(--text-muted)";
          }}
        >
          <X size={14} />
        </button>
      </div>

      {loading ? (
        <div className="p-6 text-center">
          <div
            className="w-8 h-8 rounded-lg mx-auto animate-pulse"
            style={{ background: "var(--accent-muted)" }}
          />
        </div>
      ) : settings ? (
        <div className="p-4 space-y-5">
          {/* RAG Settings */}
          <section>
            <div className="flex items-center gap-2 mb-3">
              <Database size={13} style={{ color: "var(--accent)" }} />
              <span
                className="text-xs font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                RAG (Retrieval Augmented Generation)
              </span>
            </div>
            <div
              className="rounded-xl p-3 space-y-3"
              style={{
                background: "var(--bg-primary)",
                border: "1px solid var(--border-color)",
              }}
            >
              <ToggleSetting
                label="Enable RAG"
                description="Inject relevant document chunks into context"
                value={settings.rag_enabled}
                onChange={(v) => updateField("rag_enabled", v)}
              />
              {settings.rag_enabled && (
                <NumberSetting
                  label="Top-K Results"
                  description="Number of chunks to retrieve"
                  value={settings.rag_top_k}
                  onChange={(v) => updateField("rag_top_k", v)}
                  min={1}
                  max={20}
                />
              )}
            </div>
          </section>

          {/* RAG Precision */}
          {settings.rag_enabled && (
            <section>
              <div className="flex items-center gap-2 mb-3">
                <Crosshair size={13} style={{ color: "var(--accent)" }} />
                <span
                  className="text-xs font-semibold"
                  style={{ color: "var(--text-primary)" }}
                >
                  RAG Precision
                </span>
              </div>
              <div
                className="rounded-xl p-3 space-y-3"
                style={{
                  background: "var(--bg-primary)",
                  border: "1px solid var(--border-color)",
                }}
              >
                <div>
                  <label
                    className="text-[11px] font-medium block mb-1"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Precision Mode
                  </label>
                  <select
                    value={settings.rag_precision_mode ?? "legacy_rrf"}
                    onChange={(e) =>
                      updateField("rag_precision_mode", e.target.value)
                    }
                    className="w-full rounded-lg px-2.5 py-1.5 text-xs outline-none"
                    style={{
                      background: "var(--bg-secondary)",
                      border: "1px solid var(--border-color)",
                      color: "var(--text-primary)",
                    }}
                  >
                    <option value="legacy_rrf">Legacy RRF (Reciprocal Rank Fusion)</option>
                    <option value="precision_fusion">Precision Fusion (calibrated)</option>
                  </select>
                  <p
                    className="text-[10px] mt-1"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {settings.rag_precision_mode === "precision_fusion"
                      ? "Calibrated score fusion with threshold gates, MMR diversity, and per-doc caps"
                      : "Classic hybrid retrieval using rank-based fusion"}
                  </p>
                </div>

                {settings.rag_precision_mode === "precision_fusion" && (
                  <>
                    <NumberSetting
                      label="Min Similarity Threshold"
                      description="Minimum normalized vector similarity (below → rejected)"
                      value={settings.rag_vector_min_score ?? 0.28}
                      onChange={(v) => updateField("rag_vector_min_score", v)}
                      min={0}
                      max={1}
                      step={0.02}
                    />
                    <NumberSetting
                      label="Required Term Coverage"
                      description="Minimum fraction of query terms in keyword results"
                      value={settings.rag_lexical_required_coverage ?? 0.5}
                      onChange={(v) => updateField("rag_lexical_required_coverage", v)}
                      min={0}
                      max={1}
                      step={0.05}
                    />
                    <NumberSetting
                      label="Per-Document Cap"
                      description="Max chunks from a single document (0 = unlimited)"
                      value={settings.rag_per_document_cap ?? 2}
                      onChange={(v) => updateField("rag_per_document_cap", v)}
                      min={0}
                      max={10}
                    />
                    <NumberSetting
                      label="Candidate Pool Size"
                      description="How many candidates to fetch from each source"
                      value={settings.rag_candidate_pool_size ?? 80}
                      onChange={(v) => updateField("rag_candidate_pool_size", v)}
                      min={20}
                      max={200}
                      step={10}
                    />
                    <ToggleSetting
                      label="MMR Diversity"
                      description="Use Maximal Marginal Relevance for result diversity"
                      value={settings.rag_use_mmr ?? true}
                      onChange={(v) => updateField("rag_use_mmr", v)}
                    />
                    {settings.rag_use_mmr && (
                      <NumberSetting
                        label="MMR Lambda"
                        description="Relevance vs diversity tradeoff (higher = more relevance)"
                        value={settings.rag_mmr_lambda ?? 0.65}
                        onChange={(v) => updateField("rag_mmr_lambda", v)}
                        min={0}
                        max={1}
                        step={0.05}
                      />
                    )}
                  </>
                )}
              </div>
            </section>
          )}

          {/* KV Cache Compression */}
          <section>
            <div className="flex items-center gap-2 mb-3">
              <Zap size={13} style={{ color: "var(--accent)" }} />
              <span
                className="text-xs font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                KV Cache Compression
              </span>
            </div>
            <div
              className="rounded-xl p-3 space-y-3"
              style={{
                background: "var(--bg-primary)",
                border: "1px solid var(--border-color)",
              }}
            >
              <ToggleSetting
                label="Enable Compression"
                description="Compress old context to save tokens (4x-20x reduction)"
                value={settings.kv_compression_enabled}
                onChange={(v) => updateField("kv_compression_enabled", v)}
              />
              {settings.kv_compression_enabled && (
                <NumberSetting
                  label="Compression Ratio"
                  description="Target ratio (0.3 = 70% compression)"
                  value={settings.compression_ratio}
                  onChange={(v) => updateField("compression_ratio", v)}
                  min={0.1}
                  max={1.0}
                  step={0.1}
                />
              )}
            </div>
          </section>

          {/* Memory Layers */}
          <section>
            <div className="flex items-center gap-2 mb-3">
              <Brain size={13} style={{ color: "var(--accent)" }} />
              <span
                className="text-xs font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                Memory Layers (MemGPT-style)
              </span>
            </div>
            <div
              className="rounded-xl p-3 space-y-3"
              style={{
                background: "var(--bg-primary)",
                border: "1px solid var(--border-color)",
              }}
            >
              <div>
                <label
                  className="text-[11px] font-medium block mb-1"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Memory Mode
                </label>
                <select
                  value={settings.memory_layer_mode}
                  onChange={(e) =>
                    updateField(
                      "memory_layer_mode",
                      e.target.value as ThreadSettings["memory_layer_mode"]
                    )
                  }
                  className="w-full rounded-lg px-2.5 py-1.5 text-xs outline-none"
                  style={{
                    background: "var(--bg-secondary)",
                    border: "1px solid var(--border-color)",
                    color: "var(--text-primary)",
                  }}
                >
                  <option value="none">None (standard history)</option>
                  <option value="sliding_window">Sliding Window</option>
                  <option value="virtual_paging">
                    Virtual Paging (MemGPT-style)
                  </option>
                </select>
                <p
                  className="text-[10px] mt-1"
                  style={{ color: "var(--text-muted)" }}
                >
                  {settings.memory_layer_mode === "none"
                    ? "All messages kept in context up to the history limit"
                    : settings.memory_layer_mode === "sliding_window"
                    ? "Old messages slide out of context, keeping only recent ones"
                    : "Old context is paged to disk, relevant pages are loaded on demand"}
                </p>
              </div>
              {settings.memory_layer_mode === "virtual_paging" && (
                <NumberSetting
                  label="Page Size (tokens)"
                  description="Size of each memory page"
                  value={settings.memory_page_size}
                  onChange={(v) => updateField("memory_page_size", v)}
                  min={512}
                  max={8192}
                  step={256}
                />
              )}
            </div>
          </section>

          {/* Inference Settings */}
          <section>
            <div className="flex items-center gap-2 mb-3">
              <Sliders size={13} style={{ color: "var(--accent)" }} />
              <span
                className="text-xs font-semibold"
                style={{ color: "var(--text-primary)" }}
              >
                Inference Settings
              </span>
            </div>
            <div
              className="rounded-xl p-3 space-y-3"
              style={{
                background: "var(--bg-primary)",
                border: "1px solid var(--border-color)",
              }}
            >
              <NumberSetting
                label="Context Window"
                description="Maximum tokens for context"
                value={settings.context_window_size}
                onChange={(v) => updateField("context_window_size", v)}
                min={1024}
                max={131072}
                step={1024}
              />
              <NumberSetting
                label="Max Output Tokens"
                description="Maximum response length"
                value={settings.max_tokens ?? 4096}
                onChange={(v) => updateField("max_tokens", v)}
                min={256}
                max={16384}
                step={256}
              />
              <NumberSetting
                label="Temperature"
                description="Creativity (0=focused, 1=creative)"
                value={settings.temperature ?? 0.7}
                onChange={(v) => updateField("temperature", v)}
                min={0}
                max={2}
                step={0.1}
              />
              <NumberSetting
                label="Max History Messages"
                description="Maximum message pairs to keep in context"
                value={settings.max_history_messages}
                onChange={(v) => updateField("max_history_messages", v)}
                min={2}
                max={100}
              />
            </div>
          </section>

          {/* Save Button */}
          <button
            onClick={handleSave}
            disabled={saving}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs font-medium transition-smooth disabled:opacity-50"
            style={{
              background: saved
                ? "rgba(52, 211, 153, 0.15)"
                : "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
              color: saved ? "var(--success)" : "white",
              border: saved
                ? "1px solid rgba(52, 211, 153, 0.3)"
                : "none",
            }}
          >
            {saved ? (
              <>
                <CheckCircle2 size={13} />
                Saved!
              </>
            ) : (
              <>
                <Save size={13} />
                {saving ? "Saving..." : "Save Settings"}
              </>
            )}
          </button>
        </div>
      ) : null}
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────

function ToggleSetting({
  label,
  description,
  value,
  onChange,
}: {
  label: string;
  description: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex-1 mr-3">
        <span
          className="text-[11px] font-medium"
          style={{ color: "var(--text-primary)" }}
        >
          {label}
        </span>
        <p
          className="text-[10px] mt-0.5"
          style={{ color: "var(--text-muted)" }}
        >
          {description}
        </p>
      </div>
      <button
        onClick={() => onChange(!value)}
        className="w-9 h-5 rounded-full transition-smooth relative shrink-0"
        style={{
          background: value
            ? "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))"
            : "var(--bg-tertiary)",
          border: `1px solid ${value ? "transparent" : "var(--border-color)"}`,
        }}
      >
        <div
          className="w-3.5 h-3.5 rounded-full absolute top-0.5 transition-all duration-200"
          style={{
            background: "white",
            left: value ? "calc(100% - 17px)" : "2px",
            boxShadow: "0 1px 3px rgba(0,0,0,0.3)",
          }}
        />
      </button>
    </div>
  );
}

function NumberSetting({
  label,
  description,
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
}: {
  label: string;
  description: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span
          className="text-[11px] font-medium"
          style={{ color: "var(--text-primary)" }}
        >
          {label}
        </span>
        <span
          className="text-[10px] font-mono"
          style={{ color: "var(--accent)" }}
        >
          {value}
        </span>
      </div>
      <p
        className="text-[10px] mb-1.5"
        style={{ color: "var(--text-muted)" }}
      >
        {description}
      </p>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1 rounded-full appearance-none cursor-pointer"
        style={{
          background: `linear-gradient(to right, var(--accent) ${
            ((value - min) / (max - min)) * 100
          }%, var(--bg-tertiary) ${((value - min) / (max - min)) * 100}%)`,
        }}
      />
    </div>
  );
}
