"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  ArrowLeft, Layers, Download, Play, Loader2, CheckCircle2,
  XCircle, Trash2, AlertTriangle, HardDrive, Upload, RefreshCw,
  FileBox, Globe, Info, Square, FolderOpen,
} from "lucide-react";
import {
  getQuantizeToolsStatus, getQuantTypes, startQuantize,
  listQuantizeJobs, cancelQuantizeJob, deleteQuantizeJob,
  importQuantizeToOllama, streamQuantizeSetup, streamQuantizeJob,
  importGGUFDirect, uploadGGUF,
  QuantizeToolsStatus, QuantType, QuantizeJob,
} from "@/lib/api";

interface ModelQuantizerProps {
  onBack: () => void;
}

export default function ModelQuantizer({ onBack }: ModelQuantizerProps) {
  // Tool status
  const [toolsStatus, setToolsStatus] = useState<QuantizeToolsStatus | null>(null);
  const [settingUp, setSettingUp] = useState(false);
  const [setupProgress, setSetupProgress] = useState({ stage: "", message: "", percent: 0 });

  // Quant types
  const [quantTypes, setQuantTypes] = useState<Record<string, QuantType>>({});

  // Form state
  const [sourceType, setSourceType] = useState<"huggingface" | "local_gguf">("huggingface");
  const [source, setSource] = useState("");
  const [quantLevel, setQuantLevel] = useState("Q4_K_M");
  const [outputName, setOutputName] = useState("");
  const [importToOllama, setImportToOllama] = useState(true);

  // Quick Import state
  const [importPath, setImportPath] = useState("");
  const [importFile, setImportFile] = useState<File | null>(null);
  const [importName, setImportName] = useState("");
  const [importing, setImporting] = useState(false);
  const [importSuccess, setImportSuccess] = useState<string | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Jobs
  const [jobs, setJobs] = useState<QuantizeJob[]>([]);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<QuantizeJob | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const streamRef = useRef<AbortController | null>(null);
  const setupStreamRef = useRef<AbortController | null>(null);

  // Load tools status + quant types + jobs on mount
  const loadData = useCallback(async () => {
    try {
      const [status, types, jobList] = await Promise.all([
        getQuantizeToolsStatus(),
        getQuantTypes(),
        listQuantizeJobs(),
      ]);
      setToolsStatus(status);
      setQuantTypes(types);
      setJobs(jobList);
    } catch {
      // Silently handle initial load failures
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Auto-refresh jobs every 3s while there's an active job
  useEffect(() => {
    if (!activeJobId) return;
    const iv = setInterval(async () => {
      try {
        const jobList = await listQuantizeJobs();
        setJobs(jobList);
      } catch { /* ignore */ }
    }, 3000);
    return () => clearInterval(iv);
  }, [activeJobId]);

  // Cleanup streams on unmount
  useEffect(() => {
    return () => {
      streamRef.current?.abort();
      setupStreamRef.current?.abort();
    };
  }, []);

  // Auto-generate output name from source
  useEffect(() => {
    if (!source) { setOutputName(""); return; }
    if (sourceType === "huggingface") {
      const parts = source.split("/");
      const name = parts[parts.length - 1] || source;
      setOutputName(name.toLowerCase().replace(/[^a-z0-9_-]/g, "-"));
    } else {
      const base = source.replace(/\\/g, "/").split("/").pop() || "";
      const noExt = base.replace(/\.gguf$/i, "");
      setOutputName(noExt.toLowerCase().replace(/[^a-z0-9_-]/g, "-"));
    }
  }, [source, sourceType]);

  // Auto-generate import name from GGUF path or file
  useEffect(() => {
    const filename = importFile?.name || importPath;
    if (!filename) { setImportName(""); return; }
    const base = filename.replace(/\\/g, "/").split("/").pop() || "";
    const noExt = base.replace(/\.gguf$/i, "");
    setImportName(noExt.toLowerCase().replace(/[^a-z0-9_-]/g, "-"));
  }, [importPath, importFile]);

  // Quick import handler
  const handleQuickImport = async () => {
    setImportError(null);
    setImportSuccess(null);
    setImporting(true);
    try {
      let result;
      if (importFile) {
        // Upload the file
        result = await uploadGGUF(importFile, importName.trim());
      } else {
        // Use path-based import
        result = await importGGUFDirect({
          gguf_path: importPath.trim(),
          model_name: importName.trim(),
        });
      }
      const sizeGB = (result.size / (1024 * 1024 * 1024)).toFixed(1);
      setImportSuccess(`Imported "${result.model_name}" to Ollama (${sizeGB} GB)`);
      setImportPath("");
      setImportFile(null);
      setImportName("");
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (e: unknown) {
      setImportError(e instanceof Error ? e.message : "Import failed");
    } finally {
      setImporting(false);
    }
  };

  // Start tool setup
  const handleSetupTools = () => {
    setSettingUp(true);
    setSetupProgress({ stage: "checking", message: "Starting setup...", percent: 0 });

    setupStreamRef.current?.abort();
    setupStreamRef.current = streamQuantizeSetup({
      onProgress: (data) => {
        setSetupProgress({ stage: data.stage, message: data.message, percent: data.percent });
      },
      onDone: async () => {
        setSettingUp(false);
        setSetupProgress({ stage: "ready", message: "Tools ready!", percent: 100 });
        const status = await getQuantizeToolsStatus();
        setToolsStatus(status);
      },
      onError: (err) => {
        setSettingUp(false);
        setSetupProgress({ stage: "error", message: err, percent: 0 });
      },
    });
  };

  // Start quantization
  const handleStart = async () => {
    setError(null);
    setStarting(true);
    try {
      const result = await startQuantize({
        source_type: sourceType,
        source: source.trim(),
        quant_level: quantLevel,
        output_name: outputName.trim(),
        import_to_ollama: importToOllama,
      });
      setActiveJobId(result.job_id);
      if (result.job) setActiveJob(result.job);

      // Start streaming
      streamRef.current?.abort();
      streamRef.current = streamQuantizeJob(result.job_id, {
        onStatus: (job) => {
          setActiveJob(job);
          if (job.status === "complete" || job.status === "error" || job.status === "cancelled") {
            setActiveJobId(null);
            loadData();
          }
        },
        onDone: () => {
          loadData();
        },
        onError: () => {
          setActiveJobId(null);
          loadData();
        },
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to start quantization");
    } finally {
      setStarting(false);
    }
  };

  // Cancel job
  const handleCancel = async (jobId: string) => {
    try {
      await cancelQuantizeJob(jobId);
      streamRef.current?.abort();
      setActiveJobId(null);
      setActiveJob(null);
      loadData();
    } catch { /* ignore */ }
  };

  // Delete job
  const handleDelete = async (jobId: string) => {
    try {
      await deleteQuantizeJob(jobId);
      loadData();
    } catch { /* ignore */ }
  };

  // Import to Ollama
  const handleImport = async (jobId: string) => {
    try {
      await importQuantizeToOllama(jobId);
      loadData();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Import failed");
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
    return `${(bytes / 1024).toFixed(0)} KB`;
  };

  const stageLabel = (stage: string) => {
    const map: Record<string, string> = {
      pending: "Pending", downloading: "Downloading", converting: "Converting",
      quantizing: "Quantizing", importing: "Importing", complete: "Complete",
      cancelled: "Cancelled",
    };
    return map[stage] || stage;
  };

  const statusColor = (status: string) => {
    if (status === "complete") return "var(--success)";
    if (status === "error") return "var(--error)";
    if (status === "cancelled") return "var(--warning)";
    return "var(--accent)";
  };

  const canStart = source.trim() && outputName.trim() && !starting && !activeJobId;

  return (
    <div className="flex-1 overflow-y-auto" style={{ background: "var(--bg-primary)" }}>
      <div className="max-w-3xl mx-auto p-6 animate-fadeIn">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <button
            onClick={onBack}
            className="p-2 rounded-xl transition-smooth hover-lift"
            style={{ background: "var(--bg-secondary)", border: "1px solid var(--border-color)" }}
          >
            <ArrowLeft size={18} style={{ color: "var(--text-secondary)" }} />
          </button>
          <Layers size={24} style={{ color: "var(--accent)" }} />
          <h1 className="text-xl font-bold gradient-text">Model Quantizer</h1>
          <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: "var(--accent-muted)", color: "var(--accent)" }}>
            GGUF
          </span>
        </div>

        {/* Quick Import GGUF */}
        <div className="glass-card p-5 rounded-xl mb-5">
          <h2 className="text-sm font-semibold mb-1 flex items-center gap-2" style={{ color: "var(--text-primary)" }}>
            <Upload size={16} style={{ color: "var(--accent)" }} />
            Quick Import GGUF
          </h2>
          <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>
            Import a local .gguf model directly to Ollama — ready to chat instantly, no quantization needed.
          </p>

          {importSuccess && (
            <div className="flex items-center gap-2 p-3 rounded-xl mb-3" style={{ background: "rgba(34,197,94,0.1)", border: "1px solid var(--success)" }}>
              <CheckCircle2 size={14} style={{ color: "var(--success)" }} />
              <span className="text-sm" style={{ color: "var(--success)" }}>{importSuccess}</span>
            </div>
          )}
          {importError && (
            <div className="flex items-center gap-2 p-3 rounded-xl mb-3" style={{ background: "rgba(248,113,113,0.1)", border: "1px solid var(--error)" }}>
              <XCircle size={14} style={{ color: "var(--error)" }} />
              <span className="text-sm" style={{ color: "var(--error)" }}>{importError}</span>
            </div>
          )}

          <div className="space-y-3">
            {/* File Chooser */}
            <div>
              <label className="text-xs font-medium mb-1.5 block" style={{ color: "var(--text-muted)" }}>Choose GGUF File</label>
              <input
                ref={fileInputRef}
                type="file"
                accept=".gguf"
                onChange={(e) => {
                  const f = e.target.files?.[0] || null;
                  setImportFile(f);
                  if (f) setImportPath("");
                }}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full px-3 py-2.5 rounded-xl text-sm transition-smooth flex items-center gap-2"
                style={{
                  background: "var(--bg-tertiary)",
                  color: importFile ? "var(--text-primary)" : "var(--text-muted)",
                  border: importFile ? "1px solid var(--accent)" : "1px dashed var(--border-color)",
                  cursor: "pointer",
                }}
              >
                <FolderOpen size={14} style={{ color: importFile ? "var(--accent)" : "var(--text-muted)" }} />
                {importFile ? importFile.name : "Browse for .gguf file..."}
              </button>
            </div>

            {/* OR divider */}
            <div className="flex items-center gap-3">
              <div className="flex-1 h-px" style={{ background: "var(--border-color)" }} />
              <span className="text-xs" style={{ color: "var(--text-muted)" }}>OR</span>
              <div className="flex-1 h-px" style={{ background: "var(--border-color)" }} />
            </div>

            {/* Path Input */}
            <div>
              <label className="text-xs font-medium mb-1.5 block" style={{ color: "var(--text-muted)" }}>Paste File Path</label>
              <input
                type="text"
                value={importPath}
                onChange={(e) => {
                  setImportPath(e.target.value);
                  if (e.target.value) {
                    setImportFile(null);
                    if (fileInputRef.current) fileInputRef.current.value = "";
                  }
                }}
                placeholder="e.g. C:\models\llama-3-8b-q4_k_m.gguf"
                className="w-full px-3 py-2.5 rounded-xl text-sm glass-input transition-smooth"
                style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", border: "1px solid var(--border-color)" }}
              />
            </div>

            <div>
              <label className="text-xs font-medium mb-1.5 block" style={{ color: "var(--text-muted)" }}>Model Name in Ollama</label>
              <input
                type="text"
                value={importName}
                onChange={(e) => setImportName(e.target.value)}
                placeholder="e.g. llama-3-8b-q4km"
                className="w-full px-3 py-2.5 rounded-xl text-sm glass-input transition-smooth"
                style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", border: "1px solid var(--border-color)" }}
              />
            </div>
            <button
              onClick={handleQuickImport}
              disabled={(!importPath.trim() && !importFile) || !importName.trim() || importing}
              className="w-full py-2.5 rounded-xl text-sm font-medium transition-smooth hover-lift flex items-center justify-center gap-2"
              style={{
                background: (importPath.trim() || importFile) && importName.trim() && !importing
                  ? "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))" : "var(--bg-tertiary)",
                color: (importPath.trim() || importFile) && importName.trim() && !importing ? "#fff" : "var(--text-muted)",
                border: "none",
                cursor: (!importPath.trim() && !importFile) || !importName.trim() || importing ? "not-allowed" : "pointer",
              }}
            >
              {importing ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />}
              {importing ? "Importing..." : "Import to Ollama"}
            </button>
          </div>
        </div>

        {/* Tool Setup Banner */}
        {toolsStatus && !toolsStatus.ready && (
          <div
            className="glass-card p-4 rounded-xl mb-5 animate-slideDown"
            style={{ borderColor: "var(--warning)", borderWidth: 1 }}
          >
            <div className="flex items-center gap-3 mb-2">
              <AlertTriangle size={18} style={{ color: "var(--warning)" }} />
              <span className="font-medium" style={{ color: "var(--warning)" }}>
                Quantization tools not installed
              </span>
            </div>
            <p className="text-sm mb-3" style={{ color: "var(--text-secondary)" }}>
              llama.cpp tools are required for quantization. They will be downloaded from GitHub (~50 MB).
            </p>
            {settingUp ? (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Loader2 size={14} className="animate-spin" style={{ color: "var(--accent)" }} />
                  <span className="text-sm" style={{ color: "var(--text-secondary)" }}>{setupProgress.message}</span>
                </div>
                <div className="w-full h-2 rounded-full overflow-hidden" style={{ background: "var(--bg-tertiary)" }}>
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{ width: `${setupProgress.percent}%`, background: "linear-gradient(90deg, var(--gradient-start), var(--gradient-end))" }}
                  />
                </div>
              </div>
            ) : (
              <button
                onClick={handleSetupTools}
                className="px-4 py-2 rounded-xl text-sm font-medium transition-smooth hover-lift"
                style={{ background: "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))", color: "#fff" }}
              >
                <Download size={14} className="inline mr-2" />
                Download Tools
              </button>
            )}
            {setupProgress.stage === "error" && (
              <p className="text-sm mt-2" style={{ color: "var(--error)" }}>{setupProgress.message}</p>
            )}
          </div>
        )}

        {/* Active Job Progress */}
        {activeJob && activeJob.status === "running" && (
          <div className="glass-card p-5 rounded-xl mb-5 animate-slideDown">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Loader2 size={16} className="animate-spin" style={{ color: "var(--accent)" }} />
                <span className="font-medium" style={{ color: "var(--text-primary)" }}>
                  {activeJob.output_name}
                </span>
                <span
                  className="text-xs px-2 py-0.5 rounded-full font-medium"
                  style={{ background: "rgba(99,102,241,0.15)", color: "var(--accent)" }}
                >
                  {stageLabel(activeJob.stage)}
                </span>
              </div>
              <button
                onClick={() => handleCancel(activeJob.id)}
                className="p-1.5 rounded-lg transition-smooth"
                style={{ color: "var(--error)" }}
                title="Cancel"
              >
                <Square size={14} />
              </button>
            </div>
            <p className="text-sm mb-3" style={{ color: "var(--text-secondary)" }}>{activeJob.message}</p>
            <div className="w-full h-2.5 rounded-full overflow-hidden" style={{ background: "var(--bg-tertiary)" }}>
              {activeJob.progress === 0 && activeJob.stage === "downloading" ? (
                <div
                  className="h-full rounded-full"
                  style={{
                    width: "30%",
                    background: "linear-gradient(90deg, var(--gradient-start), var(--gradient-end))",
                    animation: "pulse 1.5s ease-in-out infinite",
                  }}
                />
              ) : (
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${Math.round(activeJob.progress * 100)}%`,
                    background: "linear-gradient(90deg, var(--gradient-start), var(--gradient-end))",
                  }}
                />
              )}
            </div>
            <p className="text-xs mt-1.5 text-right" style={{ color: "var(--text-muted)" }}>
              {activeJob.progress === 0 && activeJob.stage === "downloading"
                ? "Starting download..."
                : `${Math.round(activeJob.progress * 100)}%`}
            </p>
          </div>
        )}

        {/* Completion Banner */}
        {activeJob && activeJob.status === "complete" && (
          <div
            className="glass-card p-5 rounded-xl mb-5 animate-slideDown"
            style={{ borderColor: "var(--success)", borderWidth: 1 }}
          >
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle2 size={18} style={{ color: "var(--success)" }} />
              <span className="font-medium" style={{ color: "var(--success)" }}>Quantization Complete!</span>
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm" style={{ color: "var(--text-secondary)" }}>
              <div>Output: <span style={{ color: "var(--text-primary)" }}>{activeJob.output_name}</span></div>
              <div>Level: <span style={{ color: "var(--text-primary)" }}>{activeJob.quant_level}</span></div>
              {activeJob.output_size && (
                <div>Size: <span style={{ color: "var(--text-primary)" }}>{formatSize(activeJob.output_size)}</span></div>
              )}
              {activeJob.import_to_ollama && (
                <div className="flex items-center gap-1">
                  <CheckCircle2 size={12} style={{ color: "var(--success)" }} />
                  <span>Imported to Ollama</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Error Banner */}
        {error && (
          <div
            className="glass-card p-4 rounded-xl mb-5 flex items-center gap-3"
            style={{ borderColor: "var(--error)", borderWidth: 1 }}
          >
            <XCircle size={18} style={{ color: "var(--error)" }} />
            <span className="text-sm" style={{ color: "var(--error)" }}>{error}</span>
            <button onClick={() => setError(null)} className="ml-auto text-xs" style={{ color: "var(--text-muted)" }}>
              Dismiss
            </button>
          </div>
        )}

        {/* Source Selection */}
        <div className="glass-card p-5 rounded-xl mb-5">
          <h2 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>
            Model Source
          </h2>

          <div className="flex gap-3 mb-4">
            <button
              onClick={() => setSourceType("huggingface")}
              className="flex-1 flex items-center gap-2 p-3 rounded-xl transition-smooth text-sm"
              style={{
                background: sourceType === "huggingface" ? "var(--accent-muted)" : "var(--bg-tertiary)",
                border: `1px solid ${sourceType === "huggingface" ? "var(--accent)" : "var(--border-color)"}`,
                color: sourceType === "huggingface" ? "var(--accent)" : "var(--text-secondary)",
              }}
            >
              <Globe size={16} />
              HuggingFace
            </button>
            <button
              onClick={() => setSourceType("local_gguf")}
              className="flex-1 flex items-center gap-2 p-3 rounded-xl transition-smooth text-sm"
              style={{
                background: sourceType === "local_gguf" ? "var(--accent-muted)" : "var(--bg-tertiary)",
                border: `1px solid ${sourceType === "local_gguf" ? "var(--accent)" : "var(--border-color)"}`,
                color: sourceType === "local_gguf" ? "var(--accent)" : "var(--text-secondary)",
              }}
            >
              <HardDrive size={16} />
              Local GGUF
            </button>
          </div>

          <div>
            <label className="text-xs font-medium mb-1.5 block" style={{ color: "var(--text-muted)" }}>
              {sourceType === "huggingface" ? "HuggingFace Model ID" : "Local GGUF File Path"}
            </label>
            <input
              type="text"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder={sourceType === "huggingface" ? "e.g. mistralai/Mistral-7B-v0.3" : "e.g. C:\\models\\mistral-7b.gguf"}
              className="w-full px-3 py-2.5 rounded-xl text-sm glass-input transition-smooth"
              style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", border: "1px solid var(--border-color)" }}
            />
            {sourceType === "huggingface" && (
              <p className="text-xs mt-1.5" style={{ color: "var(--text-muted)" }}>
                The model will be downloaded, converted to GGUF, then quantized.
              </p>
            )}
          </div>
        </div>

        {/* Quantization Settings */}
        <div className="glass-card p-5 rounded-xl mb-5">
          <h2 className="text-sm font-semibold mb-4" style={{ color: "var(--text-primary)" }}>
            Quantization Settings
          </h2>

          <div className="space-y-4">
            {/* Quant Level */}
            <div>
              <label className="text-xs font-medium mb-1.5 block" style={{ color: "var(--text-muted)" }}>
                Quantization Level
              </label>
              <select
                value={quantLevel}
                onChange={(e) => setQuantLevel(e.target.value)}
                className="w-full px-3 py-2.5 rounded-xl text-sm transition-smooth"
                style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", border: "1px solid var(--border-color)" }}
              >
                {Object.entries(quantTypes).map(([key, qt]) => (
                  <option key={key} value={key}>
                    {key} — {qt.description} ({qt.bits_per_weight} bpw, ~{Math.round(qt.size_ratio * 100)}% size)
                  </option>
                ))}
              </select>
              {quantTypes[quantLevel] && (
                <div className="flex items-start gap-2 mt-2 p-2.5 rounded-lg" style={{ background: "var(--bg-tertiary)" }}>
                  <Info size={14} style={{ color: "var(--accent)", marginTop: 2, flexShrink: 0 }} />
                  <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                    {quantTypes[quantLevel].quality_note}
                  </span>
                </div>
              )}
            </div>

            {/* Output Name */}
            <div>
              <label className="text-xs font-medium mb-1.5 block" style={{ color: "var(--text-muted)" }}>
                Output Model Name
              </label>
              <input
                type="text"
                value={outputName}
                onChange={(e) => setOutputName(e.target.value)}
                placeholder="e.g. mistral-7b-q4km"
                className="w-full px-3 py-2.5 rounded-xl text-sm glass-input transition-smooth"
                style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", border: "1px solid var(--border-color)" }}
              />
            </div>

            {/* Import Toggle */}
            <div className="flex items-center justify-between">
              <div>
                <span className="text-sm" style={{ color: "var(--text-primary)" }}>Import to Ollama</span>
                <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                  Automatically register the quantized model in Ollama
                </p>
              </div>
              <button
                onClick={() => setImportToOllama(!importToOllama)}
                className="w-11 h-6 rounded-full transition-smooth relative"
                style={{
                  background: importToOllama ? "var(--accent)" : "var(--bg-tertiary)",
                  border: `1px solid ${importToOllama ? "var(--accent)" : "var(--border-color)"}`,
                }}
              >
                <div
                  className="w-4 h-4 rounded-full transition-smooth absolute top-0.5"
                  style={{
                    background: "#fff",
                    left: importToOllama ? "calc(100% - 20px)" : "4px",
                  }}
                />
              </button>
            </div>
          </div>
        </div>

        {/* Start Button */}
        <button
          onClick={handleStart}
          disabled={!canStart}
          className="w-full py-3 rounded-xl text-sm font-semibold transition-smooth hover-lift mb-6 flex items-center justify-center gap-2"
          style={{
            background: canStart
              ? "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))"
              : "var(--bg-tertiary)",
            color: canStart ? "#fff" : "var(--text-muted)",
            border: "none",
            opacity: canStart ? 1 : 0.6,
            cursor: canStart ? "pointer" : "not-allowed",
          }}
        >
          {starting ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <Play size={16} />
          )}
          {starting ? "Starting..." : "Start Quantization"}
        </button>

        {/* Job History */}
        {jobs.length > 0 && (
          <div className="glass-card p-5 rounded-xl">
            <h2 className="text-sm font-semibold mb-4 flex items-center gap-2" style={{ color: "var(--text-primary)" }}>
              <FileBox size={16} style={{ color: "var(--accent)" }} />
              Job History
              <button onClick={loadData} className="ml-auto p-1 rounded-lg transition-smooth" style={{ color: "var(--text-muted)" }}>
                <RefreshCw size={14} />
              </button>
            </h2>

            <div className="space-y-3">
              {jobs.map((job) => (
                <div
                  key={job.id}
                  className="p-3 rounded-xl transition-smooth"
                  style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border-color)" }}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <div className="flex items-center gap-2">
                      {job.status === "running" ? (
                        <Loader2 size={14} className="animate-spin" style={{ color: "var(--accent)" }} />
                      ) : job.status === "complete" ? (
                        <CheckCircle2 size={14} style={{ color: "var(--success)" }} />
                      ) : job.status === "error" ? (
                        <XCircle size={14} style={{ color: "var(--error)" }} />
                      ) : (
                        <AlertTriangle size={14} style={{ color: "var(--warning)" }} />
                      )}
                      <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                        {job.output_name}
                      </span>
                      <span
                        className="text-xs px-1.5 py-0.5 rounded"
                        style={{ background: "var(--bg-hover)", color: "var(--text-muted)" }}
                      >
                        {job.quant_level}
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      {job.status === "complete" && !job.import_to_ollama && (
                        <button
                          onClick={() => handleImport(job.id)}
                          className="p-1.5 rounded-lg transition-smooth"
                          style={{ color: "var(--accent)" }}
                          title="Import to Ollama"
                        >
                          <Upload size={13} />
                        </button>
                      )}
                      {job.status === "running" && (
                        <button
                          onClick={() => handleCancel(job.id)}
                          className="p-1.5 rounded-lg transition-smooth"
                          style={{ color: "var(--error)" }}
                          title="Cancel"
                        >
                          <Square size={13} />
                        </button>
                      )}
                      {job.status !== "running" && (
                        <button
                          onClick={() => handleDelete(job.id)}
                          className="p-1.5 rounded-lg transition-smooth"
                          style={{ color: "var(--text-muted)" }}
                          title="Delete"
                        >
                          <Trash2 size={13} />
                        </button>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-3 text-xs" style={{ color: "var(--text-muted)" }}>
                    <span style={{ color: statusColor(job.status) }}>
                      {job.status === "running" ? stageLabel(job.stage) : job.status}
                    </span>
                    {job.source_type === "huggingface" && (
                      <span className="flex items-center gap-1"><Globe size={10} />{job.source}</span>
                    )}
                    {job.output_size && <span>{formatSize(job.output_size)}</span>}
                    {job.error && (
                      <span className="truncate" style={{ color: "var(--error)", maxWidth: 200 }} title={job.error}>
                        {job.error}
                      </span>
                    )}
                  </div>

                  {job.status === "running" && (
                    <div className="mt-2">
                      <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: "var(--bg-hover)" }}>
                        <div
                          className="h-full rounded-full transition-all duration-300"
                          style={{
                            width: `${Math.round(job.progress * 100)}%`,
                            background: "linear-gradient(90deg, var(--gradient-start), var(--gradient-end))",
                          }}
                        />
                      </div>
                      <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>{job.message}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
