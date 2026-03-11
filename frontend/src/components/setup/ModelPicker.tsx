"use client";

import { useState, useEffect, useRef } from "react";
import {
  Download,
  Check,
  AlertCircle,
  Loader2,
  HardDrive,
  Cpu,
  Search,
  ArrowRight,
  Trash2,
  X,
} from "lucide-react";
import { getModelCatalog, listModels, streamModelPull, deleteModel } from "@/lib/api";
import { CatalogModel, ModelInfo } from "@/types";

interface ModelPickerProps {
  onModelInstalled: () => void;
  onSkip: () => void;
}

// Family colors for visual grouping
const FAMILY_COLORS: Record<string, string> = {
  Llama: "#3b82f6",
  Gemma: "#22c55e",
  Qwen: "#a855f7",
  Mistral: "#f97316",
  Phi: "#06b6d4",
  "Code Llama": "#ef4444",
  DeepSeek: "#eab308",
  TinyLlama: "#84cc16",
  LLaVA: "#ec4899",
  StarCoder: "#14b8a6",
};

export default function ModelPicker({
  onModelInstalled,
  onSkip,
}: ModelPickerProps) {
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [installedModels, setInstalledModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFamily, setSelectedFamily] = useState<string | null>(null);
  const [pullingModel, setPullingModel] = useState<string | null>(null);
  const [pullProgress, setPullProgress] = useState(0);
  const [pullStatus, setPullStatus] = useState("");
  const [deletingModel, setDeletingModel] = useState<string | null>(null);
  const [hardware, setHardware] = useState<{
    vram_mb: number | null;
    ram_mb: number | null;
  }>({ vram_mb: null, ram_mb: null });
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [catalogData, installed] = await Promise.all([
        getModelCatalog(),
        listModels().catch(() => []),
      ]);
      setCatalog(catalogData.catalog);
      setHardware(catalogData.hardware);
      setInstalledModels(installed);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Failed to load model catalog"
      );
    }
    setLoading(false);
  };

  const handlePull = (modelName: string) => {
    setPullingModel(modelName);
    setPullProgress(0);
    setPullStatus("Starting download...");

    const controller = streamModelPull(modelName, {
      onProgress: (data) => {
        setPullProgress(data.progress || 0);
        setPullStatus(data.status || "Downloading...");
      },
      onDone: () => {
        setPullingModel(null);
        setPullProgress(0);
        setPullStatus("");
        // Refresh installed models
        listModels()
          .then(setInstalledModels)
          .catch(() => {});
      },
      onError: (err) => {
        setPullingModel(null);
        setPullProgress(0);
        setPullStatus("");
        setError(`Failed to download: ${err}`);
        setTimeout(() => setError(null), 5000);
      },
    });

    abortRef.current = controller;
  };

  const handleCancelPull = () => {
    if (abortRef.current) {
      abortRef.current.abort();
      setPullingModel(null);
      setPullProgress(0);
      setPullStatus("");
    }
  };

  const handleDelete = async (modelName: string) => {
    setDeletingModel(modelName);
    try {
      await deleteModel(modelName);
      const updated = await listModels().catch(() => []);
      setInstalledModels(updated);
    } catch {
      setError(`Failed to delete ${modelName}`);
      setTimeout(() => setError(null), 5000);
    }
    setDeletingModel(null);
  };

  const isInstalled = (modelName: string) =>
    installedModels.some((m) => m.name === modelName);

  const hasAnyModel = installedModels.length > 0;

  // Get unique families from catalog
  const families = [...new Set(catalog.map((m) => m.family))];

  // Filter models
  const filtered = catalog.filter((m) => {
    const matchesSearch =
      !searchQuery ||
      m.display_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      m.family.toLowerCase().includes(searchQuery.toLowerCase()) ||
      m.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFamily = !selectedFamily || m.family === selectedFamily;
    return matchesSearch && matchesFamily;
  });

  // Sort: compatible first, then by size
  const sorted = [...filtered].sort((a, b) => {
    if (a.compatible !== b.compatible) return a.compatible ? -1 : 1;
    return a.size_gb - b.size_gb;
  });

  if (loading) {
    return (
      <div
        className="flex-1 flex items-center justify-center"
        style={{ background: "var(--bg-primary)" }}
      >
        <div className="text-center">
          <Loader2
            size={32}
            className="animate-spin mx-auto mb-3"
            style={{ color: "var(--accent)" }}
          />
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            Loading model catalog...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      className="flex-1 flex flex-col h-screen"
      style={{ background: "var(--bg-primary)" }}
    >
      {/* Header */}
      <div
        className="px-6 py-4"
        style={{
          borderBottom: "1px solid var(--border-color)",
          background: "rgba(18, 18, 26, 0.5)",
          backdropFilter: "blur(12px)",
        }}
      >
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h1 className="text-xl font-bold gradient-text">
                Choose Your AI Model
              </h1>
              <p
                className="text-xs mt-0.5"
                style={{ color: "var(--text-muted)" }}
              >
                {hardware.vram_mb
                  ? `GPU: ${Math.round(hardware.vram_mb / 1024)} GB VRAM`
                  : "No GPU detected"}{" "}
                • RAM: {hardware.ram_mb ? `${Math.round(hardware.ram_mb / 1024)} GB` : "Unknown"}
              </p>
            </div>
            <div className="flex gap-2">
              {hasAnyModel && (
                <button
                  onClick={onModelInstalled}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-smooth"
                  style={{
                    background:
                      "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
                    color: "white",
                    boxShadow: "0 2px 12px rgba(99, 102, 241, 0.25)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.boxShadow =
                      "0 4px 20px rgba(99, 102, 241, 0.4)";
                    e.currentTarget.style.transform = "translateY(-1px)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.boxShadow =
                      "0 2px 12px rgba(99, 102, 241, 0.25)";
                    e.currentTarget.style.transform = "translateY(0)";
                  }}
                >
                  Start Chatting
                  <ArrowRight size={14} />
                </button>
              )}
              <button
                onClick={onSkip}
                className="px-3 py-2 rounded-xl text-sm transition-smooth"
                style={{
                  color: "var(--text-muted)",
                  border: "1px solid var(--border-color)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "var(--accent)";
                  e.currentTarget.style.color = "var(--text-secondary)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "var(--border-color)";
                  e.currentTarget.style.color = "var(--text-muted)";
                }}
              >
                Skip
              </button>
            </div>
          </div>

          {/* Search + Family filter */}
          <div className="flex gap-2 items-center">
            <div
              className="flex-1 flex items-center gap-2 px-3 py-2 rounded-xl"
              style={{
                background: "var(--bg-secondary)",
                border: "1px solid var(--border-color)",
              }}
            >
              <Search size={14} style={{ color: "var(--text-muted)" }} />
              <input
                type="text"
                placeholder="Search models..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="flex-1 bg-transparent text-sm outline-none"
                style={{
                  color: "var(--text-primary)",
                  caretColor: "var(--accent)",
                }}
              />
              {searchQuery && (
                <button onClick={() => setSearchQuery("")}>
                  <X size={14} style={{ color: "var(--text-muted)" }} />
                </button>
              )}
            </div>

            {/* Family pills */}
            <div className="flex gap-1 flex-wrap">
              <button
                onClick={() => setSelectedFamily(null)}
                className="px-2.5 py-1 rounded-lg text-xs font-medium transition-smooth"
                style={{
                  background: !selectedFamily
                    ? "var(--accent-muted)"
                    : "var(--bg-secondary)",
                  color: !selectedFamily
                    ? "var(--accent)"
                    : "var(--text-muted)",
                  border: `1px solid ${
                    !selectedFamily ? "var(--accent)" : "var(--border-color)"
                  }`,
                }}
              >
                All
              </button>
              {families.map((family) => (
                <button
                  key={family}
                  onClick={() =>
                    setSelectedFamily(
                      selectedFamily === family ? null : family
                    )
                  }
                  className="px-2.5 py-1 rounded-lg text-xs font-medium transition-smooth"
                  style={{
                    background:
                      selectedFamily === family
                        ? `${FAMILY_COLORS[family] || "var(--accent)"}20`
                        : "var(--bg-secondary)",
                    color:
                      selectedFamily === family
                        ? FAMILY_COLORS[family] || "var(--accent)"
                        : "var(--text-muted)",
                    border: `1px solid ${
                      selectedFamily === family
                        ? FAMILY_COLORS[family] || "var(--accent)"
                        : "var(--border-color)"
                    }`,
                  }}
                >
                  {family}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div
          className="mx-6 mt-3 px-4 py-2.5 rounded-xl flex items-center gap-2 text-sm animate-slideDown"
          style={{
            background: "rgba(248, 113, 113, 0.08)",
            border: "1px solid rgba(248, 113, 113, 0.15)",
            color: "var(--error)",
          }}
        >
          <AlertCircle size={14} />
          {error}
        </div>
      )}

      {/* Model Grid */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-3">
          {sorted.map((model) => {
            const installed = isInstalled(model.name);
            const isPulling = pullingModel === model.name;
            const isDeleting = deletingModel === model.name;
            const familyColor =
              FAMILY_COLORS[model.family] || "var(--accent)";

            return (
              <div
                key={model.name}
                className="rounded-xl p-4 transition-smooth"
                style={{
                  background: "var(--bg-secondary)",
                  border: `1px solid ${
                    installed ? `${familyColor}40` : "var(--border-color)"
                  }`,
                  opacity: model.compatible ? 1 : 0.6,
                }}
                onMouseEnter={(e) => {
                  if (model.compatible) {
                    e.currentTarget.style.borderColor = familyColor;
                    e.currentTarget.style.boxShadow = `0 4px 20px ${familyColor}15`;
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = installed
                    ? `${familyColor}40`
                    : "var(--border-color)";
                  e.currentTarget.style.boxShadow = "none";
                }}
              >
                {/* Top row: name + badges */}
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ background: familyColor }}
                    />
                    <h3
                      className="text-sm font-semibold"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {model.display_name}
                    </h3>
                    {installed && (
                      <span
                        className="flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded-md"
                        style={{
                          background: "rgba(52, 211, 153, 0.12)",
                          color: "var(--success)",
                        }}
                      >
                        <Check size={10} />
                        Installed
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span
                      className="text-[11px] font-medium px-2 py-0.5 rounded-md"
                      style={{
                        background: `${familyColor}15`,
                        color: familyColor,
                      }}
                    >
                      {model.parameters}
                    </span>
                    <span
                      className="text-[11px] font-medium px-2 py-0.5 rounded-md flex items-center gap-1"
                      style={{
                        background: "var(--bg-tertiary)",
                        color: "var(--text-muted)",
                      }}
                    >
                      <HardDrive size={10} />
                      {model.size_gb} GB
                    </span>
                  </div>
                </div>

                {/* Description */}
                <p
                  className="text-xs mb-2"
                  style={{ color: "var(--text-muted)" }}
                >
                  {model.description}
                </p>

                {/* Performance note */}
                <div className="flex items-center gap-1 mb-3">
                  <Cpu size={10} style={{ color: "var(--text-muted)" }} />
                  <span
                    className="text-[10px]"
                    style={{
                      color: model.compatible
                        ? "var(--text-muted)"
                        : "var(--error)",
                    }}
                  >
                    {model.performance_note}
                  </span>
                </div>

                {/* Action button */}
                {isPulling ? (
                  <div className="space-y-2">
                    <div
                      className="h-1.5 rounded-full overflow-hidden"
                      style={{ background: "var(--bg-tertiary)" }}
                    >
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          background: `linear-gradient(90deg, ${familyColor}, ${familyColor}cc)`,
                          width: `${Math.max(pullProgress, 2)}%`,
                        }}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <span
                        className="text-[10px]"
                        style={{ color: "var(--text-muted)" }}
                      >
                        {pullStatus}{" "}
                        {pullProgress > 0 ? `(${pullProgress.toFixed(1)}%)` : ""}
                      </span>
                      <button
                        onClick={handleCancelPull}
                        className="text-[10px] px-2 py-0.5 rounded-md transition-smooth"
                        style={{
                          color: "var(--error)",
                          background: "rgba(248, 113, 113, 0.1)",
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : installed ? (
                  <div className="flex gap-2">
                    <button
                      onClick={onModelInstalled}
                      className="flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-xs font-medium transition-smooth"
                      style={{
                        background: `${familyColor}15`,
                        color: familyColor,
                        border: `1px solid ${familyColor}30`,
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = `${familyColor}25`;
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = `${familyColor}15`;
                      }}
                    >
                      Use this model
                      <ArrowRight size={12} />
                    </button>
                    <button
                      onClick={() => handleDelete(model.name)}
                      disabled={isDeleting}
                      className="flex items-center justify-center px-2 py-1.5 rounded-lg text-xs transition-smooth"
                      style={{
                        background: "rgba(248, 113, 113, 0.08)",
                        color: "var(--error)",
                        border: "1px solid rgba(248, 113, 113, 0.15)",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = "rgba(248, 113, 113, 0.15)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = "rgba(248, 113, 113, 0.08)";
                      }}
                    >
                      {isDeleting ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
                    </button>
                  </div>
                ) : model.compatible ? (
                  <button
                    onClick={() => handlePull(model.name)}
                    disabled={pullingModel !== null}
                    className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-xs font-medium transition-smooth disabled:opacity-50"
                    style={{
                      background: "var(--bg-tertiary)",
                      color: "var(--text-secondary)",
                      border: "1px solid var(--border-color)",
                    }}
                    onMouseEnter={(e) => {
                      if (!pullingModel) {
                        e.currentTarget.style.borderColor = familyColor;
                        e.currentTarget.style.color = familyColor;
                      }
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = "var(--border-color)";
                      e.currentTarget.style.color = "var(--text-secondary)";
                    }}
                  >
                    <Download size={12} />
                    Download ({model.size_gb} GB)
                  </button>
                ) : (
                  <div
                    className="w-full text-center py-1.5 rounded-lg text-xs"
                    style={{
                      background: "var(--bg-tertiary)",
                      color: "var(--text-muted)",
                      border: "1px solid var(--border-color)",
                    }}
                  >
                    Insufficient hardware
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {sorted.length === 0 && (
          <div className="text-center py-16">
            <p className="text-sm" style={{ color: "var(--text-muted)" }}>
              No models match your search.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
