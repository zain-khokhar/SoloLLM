"use client";

import { useState, useEffect, useRef } from "react";
import {
  Loader2,
  CheckCircle2,
  AlertCircle,
  Download,
  Cpu,
  Zap,
  RefreshCw,
  Package,
} from "lucide-react";
import { getRuntimeStatus, setupRuntime, streamSetupProgress, SetupProgress } from "@/lib/api";
import ModelPicker from "./ModelPicker";

type SetupStage = "checking" | "downloading" | "ready" | "pick-model" | "error";

interface SetupWizardProps {
  onComplete: () => void;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

export default function SetupWizard({ onComplete }: SetupWizardProps) {
  const [stage, setStage] = useState<SetupStage>("checking");
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState("Checking AI engine...");
  const [downloadProgress, setDownloadProgress] = useState<SetupProgress | null>(null);
  const progressAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    checkRuntime();
    return () => {
      progressAbortRef.current?.abort();
    };
  }, []);

  const startProgressStream = () => {
    progressAbortRef.current?.abort();
    progressAbortRef.current = streamSetupProgress({
      onProgress: (data) => {
        setDownloadProgress(data);
        if (data.stage === "downloading") {
          setStatusMessage("Downloading AI engine...");
        } else if (data.stage === "extracting") {
          setStatusMessage("Extracting files...");
        } else if (data.stage === "starting") {
          setStatusMessage("Starting AI engine...");
        } else if (data.stage === "ready") {
          setStage("pick-model");
        } else if (data.stage === "error") {
          setStage("error");
          setError(data.error || "Setup failed");
        }
      },
      onError: () => {
        // SSE connection lost — not critical, setup POST will still resolve
      },
    });
  };

  const checkRuntime = async () => {
    setStage("checking");
    setError(null);
    setDownloadProgress(null);
    setStatusMessage("Checking AI engine...");

    try {
      const status = await getRuntimeStatus();

      if (status.ollama.available) {
        setStage("pick-model");
      } else if (status.ollama.binary_exists) {
        setStatusMessage("Starting AI engine...");
        setStage("downloading");
        startProgressStream();
        await triggerSetup();
      } else {
        setStatusMessage("Setting up AI engine for first time...");
        setStage("downloading");
        startProgressStream();
        await triggerSetup();
      }
    } catch {
      setStage("error");
      setError("Cannot connect to backend. Make sure the backend server is running.");
    }
  };

  const triggerSetup = async () => {
    try {
      setStage("downloading");
      const result = await setupRuntime();

      progressAbortRef.current?.abort();

      if (result.success) {
        setStage("pick-model");
      } else {
        setStage("error");
        setError("Failed to set up AI engine. You can install Ollama manually from ollama.com");
      }
    } catch (e) {
      progressAbortRef.current?.abort();
      setStage("error");
      setError(
        e instanceof Error
          ? e.message
          : "Setup failed. You can install Ollama manually from ollama.com"
      );
    }
  };

  const handleModelInstalled = () => {
    onComplete();
  };

  const handleSkip = () => {
    onComplete();
  };

  if (stage === "pick-model") {
    return <ModelPicker onModelInstalled={handleModelInstalled} onSkip={handleSkip} />;
  }

  const pct = downloadProgress?.percent ?? 0;
  const dlBytes = downloadProgress?.downloaded_bytes ?? 0;
  const totalBytes = downloadProgress?.total_bytes ?? 0;
  const dlStage = downloadProgress?.stage ?? "idle";

  return (
    <div
      className="flex-1 flex items-center justify-center"
      style={{ background: "var(--bg-primary)" }}
    >
      <div className="max-w-md w-full mx-4 text-center">
        {/* Logo / Branding */}
        <div
          className="w-16 h-16 rounded-2xl mx-auto mb-6 flex items-center justify-center"
          style={{
            background:
              "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
            boxShadow: "0 8px 32px rgba(99, 102, 241, 0.3)",
          }}
        >
          <Zap size={28} color="white" />
        </div>

        <h1 className="text-2xl font-bold mb-2 gradient-text">
          Welcome to SoloLLM
        </h1>
        <p
          className="text-sm mb-8"
          style={{ color: "var(--text-muted)" }}
        >
          Your private AI assistant — runs 100% on your device
        </p>

        {/* Status Area */}
        <div
          className="rounded-2xl p-6 mb-6"
          style={{
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
          }}
        >
          {stage === "checking" && (
            <div className="space-y-4">
              <Loader2
                size={32}
                className="animate-spin mx-auto"
                style={{ color: "var(--accent)" }}
              />
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                {statusMessage}
              </p>
            </div>
          )}

          {stage === "downloading" && (
            <div className="space-y-4">
              <div className="relative w-12 h-12 mx-auto">
                {dlStage === "extracting" ? (
                  <Package
                    size={24}
                    className="absolute inset-0 m-auto animate-pulse"
                    style={{ color: "var(--accent)" }}
                  />
                ) : dlStage === "starting" ? (
                  <Loader2
                    size={24}
                    className="absolute inset-0 m-auto animate-spin"
                    style={{ color: "var(--accent)" }}
                  />
                ) : (
                  <Download
                    size={24}
                    className="absolute inset-0 m-auto animate-bounce"
                    style={{ color: "var(--accent)" }}
                  />
                )}
              </div>
              <div>
                <p
                  className="text-sm font-medium mb-1"
                  style={{ color: "var(--text-primary)" }}
                >
                  {statusMessage}
                </p>
                {dlStage === "downloading" && totalBytes > 0 ? (
                  <p
                    className="text-xs"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {formatBytes(dlBytes)} / {formatBytes(totalBytes)} ({pct}%)
                  </p>
                ) : (
                  <p
                    className="text-xs"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {dlStage === "extracting"
                      ? "Unpacking files..."
                      : dlStage === "starting"
                      ? "Almost there..."
                      : "This is a one-time setup. Please wait..."}
                  </p>
                )}
              </div>
              {/* Progress bar */}
              <div
                className="h-2 rounded-full overflow-hidden"
                style={{ background: "var(--bg-tertiary)" }}
              >
                {dlStage === "downloading" && totalBytes > 0 ? (
                  <div
                    className="h-full rounded-full transition-all duration-300 ease-out"
                    style={{
                      background:
                        "linear-gradient(90deg, var(--gradient-start), var(--gradient-end))",
                      width: `${pct}%`,
                    }}
                  />
                ) : dlStage === "extracting" || dlStage === "starting" ? (
                  <div
                    className="h-full rounded-full"
                    style={{
                      background:
                        "linear-gradient(90deg, var(--gradient-start), var(--gradient-end))",
                      width: "100%",
                      animation: "pulse-bar 1.5s ease-in-out infinite",
                    }}
                  />
                ) : (
                  <div
                    className="h-full rounded-full"
                    style={{
                      background:
                        "linear-gradient(90deg, var(--gradient-start), var(--gradient-end))",
                      width: "60%",
                      animation: "indeterminate 1.5s ease-in-out infinite",
                    }}
                  />
                )}
              </div>
            </div>
          )}

          {stage === "ready" && (
            <div className="space-y-4">
              <CheckCircle2
                size={32}
                className="mx-auto"
                style={{ color: "var(--success)" }}
              />
              <p
                className="text-sm font-medium"
                style={{ color: "var(--success)" }}
              >
                AI engine is ready!
              </p>
            </div>
          )}

          {stage === "error" && (
            <div className="space-y-4">
              <AlertCircle
                size={32}
                className="mx-auto"
                style={{ color: "var(--error)" }}
              />
              <div>
                <p
                  className="text-sm font-medium mb-1"
                  style={{ color: "var(--error)" }}
                >
                  Setup encountered an issue
                </p>
                <p
                  className="text-xs"
                  style={{ color: "var(--text-muted)" }}
                >
                  {error}
                </p>
              </div>
              <div className="flex gap-2 justify-center pt-2">
                <button
                  onClick={checkRuntime}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-smooth"
                  style={{
                    background: "var(--accent-muted)",
                    color: "var(--accent)",
                    border: "1px solid transparent",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "var(--accent)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "transparent";
                  }}
                >
                  <RefreshCw size={14} />
                  Retry
                </button>
                <button
                  onClick={onComplete}
                  className="px-4 py-2 rounded-xl text-sm font-medium transition-smooth"
                  style={{
                    background: "var(--bg-tertiary)",
                    color: "var(--text-secondary)",
                    border: "1px solid var(--border-color)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "var(--accent)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "var(--border-color)";
                  }}
                >
                  Skip for now
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Bottom info */}
        <div className="flex items-center justify-center gap-4 text-xs" style={{ color: "var(--text-muted)" }}>
          <div className="flex items-center gap-1">
            <Cpu size={12} />
            <span>Offline & Private</span>
          </div>
          <span>•</span>
          <span>No cloud required</span>
        </div>
      </div>

      {/* CSS for animations */}
      <style jsx>{`
        @keyframes indeterminate {
          0% { transform: translateX(-100%); width: 60%; }
          50% { transform: translateX(60%); width: 60%; }
          100% { transform: translateX(200%); width: 60%; }
        }
        @keyframes pulse-bar {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}
