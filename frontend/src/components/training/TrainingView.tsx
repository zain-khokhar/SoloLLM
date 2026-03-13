"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { ArrowLeft, Zap, Brain, Loader2, CheckCircle2, XCircle, BarChart3 } from "lucide-react";
import { TrainingStatus, TrainingDataPreview } from "@/types";
import { startTraining, getTrainingStatus, cancelTraining, previewTrainingData } from "@/lib/api";

interface TrainingViewProps {
  onBack: () => void;
  selectedModel: string;
}

export default function TrainingView({ onBack, selectedModel }: TrainingViewProps) {
  const [dataPreview, setDataPreview] = useState<TrainingDataPreview | null>(null);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Config form
  const [epochs, setEpochs] = useState(3);
  const [learningRate, setLearningRate] = useState(2e-4);
  const [loraRank, setLoraRank] = useState(16);
  const [outputName, setOutputName] = useState("solollm-custom");

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadPreview = useCallback(async () => {
    try {
      const data = await previewTrainingData();
      setDataPreview(data);
    } catch {
      // ignore
    }
  }, []);

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
        model: selectedModel,
        output_name: outputName,
        lora_rank: loraRank,
        num_epochs: epochs,
        learning_rate: learningRate,
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
          {selectedModel}
        </span>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-3xl mx-auto space-y-6">

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
                      </div>
                    ))}
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

          {/* Configuration */}
          {!isTraining && (
            <div className="glass-card p-5 rounded-xl">
              <h3 className="font-semibold mb-4" style={{ color: "var(--text-primary)" }}>Configuration</h3>
              <div className="grid grid-cols-2 gap-4">
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
                <h3 className="font-semibold" style={{ color: "var(--text-primary)" }}>Training Progress</h3>
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
              disabled={isLoading || !dataPreview || dataPreview.total_examples < 10}
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
              Need at least 10 conversation examples to train. Currently have {dataPreview.total_examples}.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
