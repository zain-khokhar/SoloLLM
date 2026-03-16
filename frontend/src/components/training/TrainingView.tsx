"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { ArrowLeft, Zap, Brain, Loader2, CheckCircle2, XCircle, BarChart3, FileText, Trash2, Upload, Cpu, Monitor } from "lucide-react";
import { DocumentInfo, ModelInfo, TrainingStatus, TrainingDataPreview, TrainingCapabilities } from "@/types";
import { startTraining, getTrainingStatus, cancelTraining, previewTrainingData, getTrainingCapabilities, listDocuments, uploadDocument, deleteDocument, listModels } from "@/lib/api";

interface TrainingViewProps {
  onBack: () => void;
  selectedModel: string;
  onSelectModel?: (model: string) => void;
}

export default function TrainingView({ onBack, selectedModel, onSelectModel }: TrainingViewProps) {
  const [dataPreview, setDataPreview] = useState<TrainingDataPreview | null>(null);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [isModelsLoading, setIsModelsLoading] = useState(false);
  const [trainingModel, setTrainingModel] = useState(selectedModel);
  const [sourceMode, setSourceMode] = useState<"conversation" | "documents" | "mixed">("mixed");
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [isDocumentsLoading, setIsDocumentsLoading] = useState(false);
  const [isUploadingDocument, setIsUploadingDocument] = useState(false);
  const [qualityLossThreshold, setQualityLossThreshold] = useState(1.8);
  const [capabilities, setCapabilities] = useState<TrainingCapabilities | null>(null);
  const [capabilitiesLoading, setCapabilitiesLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Config form
  const [epochs, setEpochs] = useState(3);
  const [learningRate, setLearningRate] = useState(2e-4);
  const [loraRank, setLoraRank] = useState(16);
  const [outputName, setOutputName] = useState("solollm-custom");

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadModels = useCallback(async () => {
    setIsModelsLoading(true);
    try {
      const available = await listModels();
      setModels(available);
      if (available.length > 0) {
        const modelToUse = available.some((m) => m.name === selectedModel)
            ? selectedModel
            : available[0].name;
        setTrainingModel(modelToUse);
        onSelectModel?.(modelToUse);
      }
    } catch {
      setModels([]);
    }
    setIsModelsLoading(false);
  }, [onSelectModel, selectedModel]);

  const loadDocuments = useCallback(async () => {
    setIsDocumentsLoading(true);
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      setSelectedDocumentIds((prev) => {
        if (prev.length > 0) {
          return prev.filter((id) => docs.some((doc) => doc.document_id === id));
        }
        return docs.map((doc) => doc.document_id);
      });
    } catch {
      setDocuments([]);
    }
    setIsDocumentsLoading(false);
  }, []);

  const loadPreview = useCallback(async () => {
    try {
      const docIds = sourceMode === "conversation" ? undefined : selectedDocumentIds;
      const data = await previewTrainingData(undefined, docIds, sourceMode);
      setDataPreview(data);
    } catch {
      // ignore
    }
  }, [sourceMode, selectedDocumentIds]);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  useEffect(() => {
    setTrainingModel(selectedModel);
  }, [selectedModel]);

  const fetchCapabilities = useCallback(async (model: string) => {
    if (!model) {
      setCapabilities(null);
      return;
    }
    setCapabilitiesLoading(true);
    try {
      const caps = await getTrainingCapabilities(model);
      setCapabilities(caps);
    } catch {
      setCapabilities(null);
    }
    setCapabilitiesLoading(false);
  }, []);

  useEffect(() => {
    if (trainingModel) void fetchCapabilities(trainingModel);
    else setCapabilities(null);
  }, [trainingModel, fetchCapabilities]);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  useEffect(() => {
    loadPreview();
    // Also check if training is running
    getTrainingStatus()
      .then((s) => {
        if (s.status !== "idle" && s.status !== "complete" && s.status !== "error") {
          setStatus(s);
          startPolling();
        }
      })
      .catch(() => {});
    return () => stopPolling();
  }, [loadPreview]);

  const hasSelectedDocuments = selectedDocumentIds.length > 0;
  const canStartByMode = sourceMode === "conversation" || hasSelectedDocuments;

  const handleToggleDocument = (documentId: string) => {
    setSelectedDocumentIds((prev) =>
      prev.includes(documentId)
        ? prev.filter((id) => id !== documentId)
        : [...prev, documentId]
    );
  };

  const handleUploadDocument = async (file: File) => {
    setIsUploadingDocument(true);
    setError(null);
    try {
      const result = await uploadDocument(file);
      if (result.success) {
        await loadDocuments();
        setSelectedDocumentIds((prev) =>
          prev.includes(result.document_id) ? prev : [...prev, result.document_id]
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload document");
    }
    setIsUploadingDocument(false);
  };

  const handleDeleteDocument = async (documentId: string) => {
    setError(null);
    try {
      await deleteDocument(documentId);
      setSelectedDocumentIds((prev) => prev.filter((id) => id !== documentId));
      await loadDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete document");
    }
  };

  const startPolling = () => {
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      try {
        const s = await getTrainingStatus();
        setStatus(s);
        if (s.status === "complete" || s.status === "error" || s.status === "idle") {
          stopPolling();
        }
      } catch {
        // ignore
      }
    }, 2000);
  };

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const handleStart = async () => {
    setIsLoading(true);
    setError(null);
    try {
      await startTraining({
        model: trainingModel,
        output_name: outputName,
        source_mode: sourceMode,
        document_ids: sourceMode === "conversation" ? undefined : selectedDocumentIds,
        lora_rank: loraRank,
        num_epochs: epochs,
        learning_rate: learningRate,
        quality_loss_threshold: qualityLossThreshold,
      });
      startPolling();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start training");
    }
    setIsLoading(false);
  };

  const handleCancel = async () => {
    try {
      await cancelTraining();
      stopPolling();
      setStatus(null);
    } catch {
      // ignore
    }
  };

  const isTraining = status && !["idle", "complete", "error"].includes(status.status);
  const hasAvailableModels = models.length > 0;
  const progressPercent = status?.total_steps
    ? Math.round((status.current_step / status.total_steps) * 100)
    : 0;

  return (
    <div className="flex-1 flex flex-col h-screen" style={{ background: "var(--bg-primary)" }}>
      {/* Top bar */}
      <div
        className="flex items-center gap-3 px-4 py-2.5"
        style={{
          borderBottom: "1px solid var(--border-color)",
          background: "rgba(18, 18, 26, 0.5)",
          backdropFilter: "blur(12px)",
        }}
      >
        <button
          onClick={onBack}
          className="p-2 rounded-lg transition-smooth"
          style={{ color: "var(--text-secondary)" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--bg-hover)";
            e.currentTarget.style.color = "var(--text-primary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "var(--text-secondary)";
          }}
        >
          <ArrowLeft size={18} />
        </button>
        <Brain size={18} style={{ color: "var(--accent)" }} />
        <span className="font-semibold gradient-text">Self-Training</span>
        <span className="text-xs px-2 py-0.5 rounded-md" style={{ background: "var(--accent-muted)", color: "var(--accent)" }}>
          {trainingModel}
        </span>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-3xl mx-auto space-y-6">

          {/* Training device & resources */}
          {capabilitiesLoading && (
            <div className="glass-card p-4 rounded-xl flex items-center gap-2" style={{ color: "var(--text-muted)" }}>
              <Loader2 size={16} className="animate-spin" />
              <span className="text-sm">Checking GPU/CPU and model requirements...</span>
            </div>
          )}
          {!capabilitiesLoading && capabilities && (
            <div
              className="glass-card p-4 rounded-xl border"
              style={{
                borderColor: capabilities.recommended_device === "gpu" ? "var(--accent)" : "var(--border-color)",
                background: capabilities.recommended_device === "gpu" ? "var(--accent-muted)" : "var(--bg-secondary)",
              }}
            >
              <div className="flex items-start gap-3">
                {capabilities.recommended_device === "gpu" ? (
                  <Monitor size={20} style={{ color: "var(--accent)", flexShrink: 0, marginTop: 2 }} />
                ) : (
                  <Cpu size={20} style={{ color: "var(--text-secondary)", flexShrink: 0, marginTop: 2 }} />
                )}
                <div className="min-w-0">
                  <div className="font-medium text-sm" style={{ color: "var(--text-primary)" }}>
                    Training will run on: {capabilities.recommended_device === "gpu" ? "GPU (4-bit QLoRA)" : "CPU"}
                  </div>
                  <p className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>
                    {capabilities.message}
                  </p>
                  <div className="flex flex-wrap gap-4 mt-3 text-xs" style={{ color: "var(--text-muted)" }}>
                    {capabilities.gpu_available && (
                      <span>
                        GPU: {capabilities.gpu_memory_gb} GB available
                        {capabilities.can_train_on_gpu ? (
                          <span className="ml-1" style={{ color: "var(--success)" }}>✓ enough for this model</span>
                        ) : (
                          <span className="ml-1">(need {capabilities.required_gpu_memory_gb} GB for 4-bit)</span>
                        )}
                      </span>
                    )}
                    {!capabilities.gpu_available && <span>No GPU detected</span>}
                    <span>
                      CPU RAM: {capabilities.cpu_ram_gb} GB available
                      {capabilities.can_train_on_cpu ? (
                        <span className="ml-1" style={{ color: "var(--success)" }}>✓</span>
                      ) : (
                        <span className="ml-1">(recommended {capabilities.required_cpu_ram_gb} GB for CPU training)</span>
                      )}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Training Data Preview */}
          <div className="glass-card p-5 rounded-xl">
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 size={16} style={{ color: "var(--accent)" }} />
              <h3 className="font-semibold" style={{ color: "var(--text-primary)" }}>Training Data</h3>
            </div>
            {dataPreview ? (
              <div>
                <div className="flex gap-6 mb-4">
                  <div>
                    <span className="text-2xl font-bold gradient-text">{dataPreview.total_examples}</span>
                    <span className="text-xs block" style={{ color: "var(--text-muted)" }}>examples</span>
                  </div>
                  <div>
                    <span className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>{dataPreview.conversations_used}</span>
                    <span className="text-xs block" style={{ color: "var(--text-muted)" }}>conversations</span>
                  </div>
                  <div>
                    <span className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>{dataPreview.documents_used?.length || 0}</span>
                    <span className="text-xs block" style={{ color: "var(--text-muted)" }}>documents</span>
                  </div>
                </div>
                {dataPreview.preview.length > 0 && (
                  <div className="space-y-2">
                    <span className="text-xs font-medium" style={{ color: "var(--text-muted)" }}>Preview:</span>
                    {dataPreview.preview.slice(0, 3).map((ex, i) => (
                      <div
                        key={i}
                        className="p-3 rounded-lg text-xs"
                        style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border-color)" }}
                      >
                        <div style={{ color: "var(--accent)" }} className="font-medium mb-1">
                          Q: {ex.instruction.slice(0, 100)}{ex.instruction.length > 100 ? "..." : ""}
                        </div>
                        <div style={{ color: "var(--text-secondary)" }}>
                          A: {ex.output.slice(0, 150)}{ex.output.length > 150 ? "..." : ""}
                        </div>
                        <div className="mt-1" style={{ color: "var(--text-muted)" }}>
                          Source: {ex.source_type || "conversation"} {ex.source_name ? `- ${ex.source_name}` : ""}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                {dataPreview.sequence && dataPreview.sequence.length > 0 && (
                  <div className="mt-4">
                    <span className="text-xs font-medium" style={{ color: "var(--text-muted)" }}>Training Sequence:</span>
                    <div className="mt-2 max-h-40 overflow-y-auto space-y-1.5">
                      {dataPreview.sequence.slice(0, 12).map((step) => (
                        <div
                          key={`${step.index}-${step.source_type}-${step.source_name}`}
                          className="px-2.5 py-1.5 rounded-md text-xs"
                          style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border-color)", color: "var(--text-secondary)" }}
                        >
                          {step.index}. [{step.source_type}] {step.source_name || "conversation"}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center gap-2" style={{ color: "var(--text-muted)" }}>
                <Loader2 size={14} className="animate-spin" />
                <span className="text-sm">Loading training data...</span>
              </div>
            )}
          </div>

          {/* Documents/PDFs Section */}
          <div className="glass-card p-5 rounded-xl">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <FileText size={16} style={{ color: "var(--accent)" }} />
                <h3 className="font-semibold" style={{ color: "var(--text-primary)" }}>Documents / PDFs</h3>
              </div>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-smooth"
                style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border-color)", color: "var(--text-secondary)" }}
                disabled={isUploadingDocument}
              >
                {isUploadingDocument ? <Loader2 size={12} className="animate-spin" /> : <Upload size={12} />}
                {isUploadingDocument ? "Uploading..." : "Upload"}
              </button>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".pdf,.doc,.docx,.txt,.md,.html,.csv,.py,.js,.ts,.java,.c,.cpp,.json,.xml"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void handleUploadDocument(file);
                e.currentTarget.value = "";
              }}
            />

            {isDocumentsLoading ? (
              <div className="text-sm" style={{ color: "var(--text-muted)" }}>Loading documents...</div>
            ) : documents.length === 0 ? (
              <div className="text-sm" style={{ color: "var(--text-muted)" }}>No uploaded documents yet. Upload PDFs/files here for document self-training.</div>
            ) : (
              <div className="space-y-2 max-h-56 overflow-y-auto">
                {documents.map((doc, index) => (
                  <div
                    key={doc.document_id || `${doc.filename}-${index}`}
                    className="flex items-center justify-between px-3 py-2 rounded-lg"
                    style={{ background: "var(--bg-tertiary)", border: "1px solid var(--border-color)" }}
                  >
                    <label className="flex items-center gap-2 text-xs cursor-pointer" style={{ color: "var(--text-secondary)" }}>
                      <input
                        type="checkbox"
                        checked={selectedDocumentIds.includes(doc.document_id)}
                        onChange={() => handleToggleDocument(doc.document_id)}
                      />
                      <span>{doc.filename}</span>
                    </label>
                    <button
                      onClick={() => void handleDeleteDocument(doc.document_id)}
                      className="p-1 rounded"
                      style={{ color: "var(--error)" }}
                      title="Delete document"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Configuration */}
          {!isTraining && (
            <div className="glass-card p-5 rounded-xl">
              <h3 className="font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Configuration</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-medium block mb-1" style={{ color: "var(--text-muted)" }}>
                    Base Model
                  </label>
                  <select
                    value={trainingModel}
                    onChange={(e) => {
                      const nextModel = e.target.value;
                      setTrainingModel(nextModel);
                      onSelectModel?.(nextModel);
                    }}
                    className="w-full rounded-lg px-3 py-2 text-sm"
                    style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", border: "1px solid var(--border-color)" }}
                    disabled={isModelsLoading || !hasAvailableModels}
                  >
                    {isModelsLoading && <option value="">Loading downloaded models...</option>}
                    {!isModelsLoading && !hasAvailableModels && <option value="">No downloaded models found</option>}
                    {!isModelsLoading && hasAvailableModels && models.map((model) => (
                      <option key={model.name} value={model.name}>{model.name}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-xs font-medium block mb-1" style={{ color: "var(--text-muted)" }}>
                    Training Source
                  </label>
                  <select
                    value={sourceMode}
                    onChange={(e) => setSourceMode(e.target.value as "conversation" | "documents" | "mixed")}
                    className="w-full rounded-lg px-3 py-2 text-sm"
                    style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", border: "1px solid var(--border-color)" }}
                  >
                    <option value="conversation">Conversations</option>
                    <option value="documents">Documents / PDFs</option>
                    <option value="mixed">Mixed (Conversations + Documents)</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs font-medium block mb-1" style={{ color: "var(--text-muted)" }}>
                    Epochs ({epochs})
                  </label>
                  <input
                    type="range" min={1} max={10} step={1} value={epochs}
                    onChange={(e) => setEpochs(Number(e.target.value))}
                    className="w-full accent-indigo-500"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium block mb-1" style={{ color: "var(--text-muted)" }}>
                    LoRA Rank ({loraRank})
                  </label>
                  <select
                    value={loraRank}
                    onChange={(e) => setLoraRank(Number(e.target.value))}
                    className="w-full rounded-lg px-3 py-2 text-sm"
                    style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", border: "1px solid var(--border-color)" }}
                  >
                    <option value={4}>4 (Fastest)</option>
                    <option value={8}>8</option>
                    <option value={16}>16 (Balanced)</option>
                    <option value={32}>32 (Highest Quality)</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs font-medium block mb-1" style={{ color: "var(--text-muted)" }}>
                    Learning Rate ({learningRate.toExponential(1)})
                  </label>
                  <input
                    type="range" min={-5} max={-3} step={0.1}
                    value={Math.log10(learningRate)}
                    onChange={(e) => setLearningRate(Math.pow(10, Number(e.target.value)))}
                    className="w-full accent-indigo-500"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium block mb-1" style={{ color: "var(--text-muted)" }}>
                    Quality Loss Threshold ({qualityLossThreshold.toFixed(2)})
                  </label>
                  <input
                    type="range" min={1.2} max={3.0} step={0.1}
                    value={qualityLossThreshold}
                    onChange={(e) => setQualityLossThreshold(Number(e.target.value))}
                    className="w-full accent-indigo-500"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium block mb-1" style={{ color: "var(--text-muted)" }}>
                    Output Model Name
                  </label>
                  <input
                    type="text" value={outputName}
                    onChange={(e) => setOutputName(e.target.value)}
                    className="w-full rounded-lg px-3 py-2 text-sm"
                    style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)", border: "1px solid var(--border-color)" }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Progress */}
          {isTraining && status && (
            <div className="glass-card p-5 rounded-xl">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <h3 className="font-semibold" style={{ color: "var(--text-primary)" }}>Training Progress</h3>
                  {status.device && (
                    <span
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium"
                      style={{
                        background: status.device === "gpu" ? "var(--accent-muted)" : "var(--bg-tertiary)",
                        color: status.device === "gpu" ? "var(--accent)" : "var(--text-secondary)",
                        border: `1px solid ${status.device === "gpu" ? "var(--accent)" : "var(--border-color)"}`,
                      }}
                    >
                      {status.device === "gpu" ? <Monitor size={12} /> : <Cpu size={12} />}
                      {status.device === "gpu" ? "GPU (4-bit QLoRA)" : "CPU (fp32)"}
                    </span>
                  )}
                </div>
                <button
                  onClick={handleCancel}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-smooth"
                  style={{ background: "rgba(248, 113, 113, 0.1)", color: "var(--error)", border: "1px solid rgba(248, 113, 113, 0.2)" }}
                >
                  <XCircle size={12} />
                  Cancel
                </button>
              </div>

              {/* Progress bar */}
              <div className="mb-3">
                <div className="flex justify-between text-xs mb-1" style={{ color: "var(--text-muted)" }}>
                  <span>{status.message}</span>
                  <span>{progressPercent}%</span>
                </div>
                <div className="w-full h-2 rounded-full overflow-hidden" style={{ background: "var(--bg-tertiary)" }}>
                  <div
                    className="h-2 rounded-full transition-all duration-300"
                    style={{
                      width: `${progressPercent}%`,
                      background: "linear-gradient(90deg, var(--gradient-start), var(--gradient-end))",
                    }}
                  />
                </div>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <span className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>
                    {status.current_step}/{status.total_steps}
                  </span>
                  <span className="text-[10px] block" style={{ color: "var(--text-muted)" }}>Steps</span>
                </div>
                <div>
                  <span className="text-lg font-bold" style={{ color: status.loss < 1 ? "var(--success)" : "var(--warning)" }}>
                    {status.loss.toFixed(4)}
                  </span>
                  <span className="text-[10px] block" style={{ color: "var(--text-muted)" }}>Loss</span>
                </div>
                <div>
                  <span className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>
                    {(status.val_loss || 0).toFixed(4)}
                  </span>
                  <span className="text-[10px] block" style={{ color: "var(--text-muted)" }}>Val Loss</span>
                </div>
                <div>
                  <span className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>
                    {status.epoch.toFixed(1)}
                  </span>
                  <span className="text-[10px] block" style={{ color: "var(--text-muted)" }}>Epoch</span>
                </div>
              </div>
            </div>
          )}

          {/* Complete */}
          {status?.status === "complete" && (
            <div
              className="glass-card p-5 rounded-xl flex items-center gap-3"
              style={{ border: "1px solid rgba(52, 211, 153, 0.3)" }}
            >
              <CheckCircle2 size={24} style={{ color: "var(--success)" }} />
              <div>
                <h3 className="font-semibold" style={{ color: "var(--success)" }}>Training Complete!</h3>
                <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                  Your model <strong>{outputName}</strong> is now available in the model selector.
                </p>
                {status?.quality_passed && (
                  <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                    Validation quality gate passed. Best validation loss: {(status.best_val_loss || 0).toFixed(4)}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Error */}
          {(error || status?.status === "error") && (
            <div
              className="glass-card p-5 rounded-xl flex items-center gap-3"
              style={{ border: "1px solid rgba(248, 113, 113, 0.3)" }}
            >
              <XCircle size={24} style={{ color: "var(--error)" }} />
              <div>
                <h3 className="font-semibold" style={{ color: "var(--error)" }}>Training Error</h3>
                <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                  {error || status?.error}
                </p>
              </div>
            </div>
          )}

          {/* Start button */}
          {!isTraining && (
            <button
              onClick={handleStart}
              disabled={isLoading || !dataPreview || dataPreview.total_examples < 10 || !canStartByMode || !hasAvailableModels}
              className="w-full py-3.5 rounded-xl font-semibold text-sm transition-smooth disabled:opacity-30 flex items-center justify-center gap-2"
              style={{
                background: "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
                color: "white",
                boxShadow: "0 4px 20px rgba(99, 102, 241, 0.3)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.boxShadow = "0 6px 30px rgba(99, 102, 241, 0.4)";
                e.currentTarget.style.transform = "translateY(-1px)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.boxShadow = "0 4px 20px rgba(99, 102, 241, 0.3)";
                e.currentTarget.style.transform = "translateY(0)";
              }}
            >
              {isLoading ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Zap size={16} />
              )}
              {isLoading ? "Starting..." : "Start Training"}
            </button>
          )}

          {dataPreview && dataPreview.total_examples < 10 && !isTraining && (
            <p className="text-center text-xs" style={{ color: "var(--text-muted)" }}>
              Need at least 10 examples to train. Currently have {dataPreview.total_examples}.
            </p>
          )}
          {!canStartByMode && !isTraining && (
            <p className="text-center text-xs" style={{ color: "var(--text-muted)" }}>
              Select at least one uploaded document for document or mixed training modes.
            </p>
          )}
          {!hasAvailableModels && !isTraining && !isModelsLoading && (
            <p className="text-center text-xs" style={{ color: "var(--text-muted)" }}>
              No downloaded models found. Download a model in Manage Models first.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
