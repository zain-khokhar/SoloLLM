"use client";

import { useState, useRef } from "react";
import {
  ArrowLeft,
  Download,
  Upload,
  MessageSquare,
  Settings,
  CheckCircle2,
  AlertCircle,
  FileJson,
  Loader2,
} from "lucide-react";
import {
  exportConversations,
  importConversations,
  exportSettings,
  importSettings,
  OPENAI_COMPAT_BASE_URL,
} from "@/lib/api";

interface ExportImportViewProps {
  onBack: () => void;
  onImportComplete?: () => void;
}

export default function ExportImportView({
  onBack,
  onImportComplete,
}: ExportImportViewProps) {
  const [status, setStatus] = useState<{
    type: "success" | "error";
    message: string;
  } | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const convFileRef = useRef<HTMLInputElement>(null);
  const settingsFileRef = useRef<HTMLInputElement>(null);

  const clearStatus = () => setTimeout(() => setStatus(null), 4000);

  const handleExportConversations = async () => {
    setLoading("export-conv");
    setStatus(null);
    try {
      await exportConversations();
      setStatus({ type: "success", message: "Conversations exported successfully" });
      clearStatus();
    } catch (e) {
      setStatus({
        type: "error",
        message: e instanceof Error ? e.message : "Export failed",
      });
    }
    setLoading(null);
  };

  const handleImportConversations = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading("import-conv");
    setStatus(null);
    try {
      const result = await importConversations(file);
      setStatus({
        type: "success",
        message: `Imported ${result.imported_count} conversation${result.imported_count !== 1 ? "s" : ""}`,
      });
      clearStatus();
      onImportComplete?.();
    } catch (err) {
      setStatus({
        type: "error",
        message: err instanceof Error ? err.message : "Import failed",
      });
    }
    setLoading(null);
    if (convFileRef.current) convFileRef.current.value = "";
  };

  const handleExportSettings = async () => {
    setLoading("export-settings");
    setStatus(null);
    try {
      await exportSettings();
      setStatus({ type: "success", message: "Settings exported successfully" });
      clearStatus();
    } catch (e) {
      setStatus({
        type: "error",
        message: e instanceof Error ? e.message : "Export failed",
      });
    }
    setLoading(null);
  };

  const handleImportSettings = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading("import-settings");
    setStatus(null);
    try {
      const result = await importSettings(file);
      setStatus({
        type: "success",
        message: `Imported ${result.imported_keys.length} setting${result.imported_keys.length !== 1 ? "s" : ""}`,
      });
      clearStatus();
    } catch (err) {
      setStatus({
        type: "error",
        message: err instanceof Error ? err.message : "Import failed",
      });
    }
    setLoading(null);
    if (settingsFileRef.current) settingsFileRef.current.value = "";
  };

  return (
    <div
      className="flex-1 overflow-y-auto"
      style={{ background: "var(--bg-primary)" }}
    >
      <div className="max-w-2xl mx-auto p-6 animate-fadeIn">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
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
            <ArrowLeft size={18} />
          </button>
          <div>
            <h1 className="text-xl font-bold gradient-text">
              Export / Import
            </h1>
            <p className="text-xs" style={{ color: "var(--text-muted)" }}>
              Back up or restore your conversations and settings
            </p>
          </div>
        </div>

        {/* Status Banner */}
        {status && (
          <div
            className="mb-6 px-4 py-3 rounded-xl flex items-center gap-2 text-sm animate-slideDown"
            style={{
              background:
                status.type === "success"
                  ? "rgba(52, 211, 153, 0.08)"
                  : "rgba(248, 113, 113, 0.08)",
              border: `1px solid ${
                status.type === "success"
                  ? "rgba(52, 211, 153, 0.2)"
                  : "rgba(248, 113, 113, 0.15)"
              }`,
              color:
                status.type === "success" ? "var(--success)" : "var(--error)",
            }}
          >
            {status.type === "success" ? (
              <CheckCircle2 size={16} />
            ) : (
              <AlertCircle size={16} />
            )}
            {status.message}
          </div>
        )}

        {/* Conversations Section */}
        <section className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <MessageSquare size={16} style={{ color: "var(--accent)" }} />
            <h2
              className="text-sm font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              Conversations
            </h2>
          </div>
          <div
            className="rounded-xl p-5"
            style={{
              background: "var(--bg-secondary)",
              border: "1px solid var(--border-color)",
            }}
          >
            <p
              className="text-xs mb-4"
              style={{ color: "var(--text-muted)" }}
            >
              Export all conversations and messages as a JSON file, or import
              from a previous export.
            </p>
            <div className="flex gap-3">
              <ActionButton
                icon={Download}
                label="Export Conversations"
                onClick={handleExportConversations}
                loading={loading === "export-conv"}
              />
              <ActionButton
                icon={Upload}
                label="Import Conversations"
                onClick={() => convFileRef.current?.click()}
                loading={loading === "import-conv"}
                variant="secondary"
              />
              <input
                ref={convFileRef}
                type="file"
                accept=".json"
                className="hidden"
                onChange={handleImportConversations}
              />
            </div>
          </div>
        </section>

        {/* Settings Section */}
        <section className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <Settings size={16} style={{ color: "var(--accent)" }} />
            <h2
              className="text-sm font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              Settings
            </h2>
          </div>
          <div
            className="rounded-xl p-5"
            style={{
              background: "var(--bg-secondary)",
              border: "1px solid var(--border-color)",
            }}
          >
            <p
              className="text-xs mb-4"
              style={{ color: "var(--text-muted)" }}
            >
              Export your application settings or restore from a backup.
            </p>
            <div className="flex gap-3">
              <ActionButton
                icon={Download}
                label="Export Settings"
                onClick={handleExportSettings}
                loading={loading === "export-settings"}
              />
              <ActionButton
                icon={Upload}
                label="Import Settings"
                onClick={() => settingsFileRef.current?.click()}
                loading={loading === "import-settings"}
                variant="secondary"
              />
              <input
                ref={settingsFileRef}
                type="file"
                accept=".json"
                className="hidden"
                onChange={handleImportSettings}
              />
            </div>
          </div>
        </section>

        {/* OpenAI Compatibility Info */}
        <section>
          <div className="flex items-center gap-2 mb-3">
            <FileJson size={16} style={{ color: "var(--accent)" }} />
            <h2
              className="text-sm font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              OpenAI-Compatible API
            </h2>
          </div>
          <div
            className="rounded-xl p-5"
            style={{
              background: "var(--bg-secondary)",
              border: "1px solid var(--border-color)",
            }}
          >
            <p
              className="text-xs mb-3"
              style={{ color: "var(--text-muted)" }}
            >
              SoloLLM exposes an OpenAI-compatible endpoint so you can use it as
              a drop-in replacement with any tool that supports the OpenAI API.
            </p>
            <div className="space-y-2">
              <EndpointRow
                method="POST"
                path="/v1/chat/completions"
                description="Chat completions (streaming + non-streaming)"
              />
              <EndpointRow
                method="GET"
                path="/v1/models"
                description="List available models"
              />
            </div>
            <div
              className="mt-4 rounded-lg px-3.5 py-2.5"
              style={{
                background: "var(--bg-primary)",
                border: "1px solid var(--border-color)",
              }}
            >
              <p
                className="text-[11px] font-mono"
                style={{ color: "var(--text-secondary)" }}
              >
                <span style={{ color: "var(--text-muted)" }}>Base URL:</span>{" "}
                {OPENAI_COMPAT_BASE_URL}
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────

function ActionButton({
  icon: Icon,
  label,
  onClick,
  loading,
  variant = "primary",
}: {
  icon: React.ComponentType<{ size: number }>;
  label: string;
  onClick: () => void;
  loading: boolean;
  variant?: "primary" | "secondary";
}) {
  const isPrimary = variant === "primary";
  return (
    <button
      onClick={onClick}
      disabled={loading}
      className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-smooth disabled:opacity-50"
      style={{
        background: isPrimary
          ? "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))"
          : "var(--bg-tertiary)",
        color: isPrimary ? "white" : "var(--text-secondary)",
        border: isPrimary ? "none" : "1px solid var(--border-color)",
        boxShadow: isPrimary
          ? "0 2px 12px rgba(99, 102, 241, 0.25)"
          : "none",
      }}
      onMouseEnter={(e) => {
        if (isPrimary) {
          e.currentTarget.style.boxShadow =
            "0 4px 20px rgba(99, 102, 241, 0.4)";
          e.currentTarget.style.transform = "translateY(-1px)";
        } else {
          e.currentTarget.style.borderColor = "var(--accent)";
          e.currentTarget.style.color = "var(--text-primary)";
        }
      }}
      onMouseLeave={(e) => {
        if (isPrimary) {
          e.currentTarget.style.boxShadow =
            "0 2px 12px rgba(99, 102, 241, 0.25)";
          e.currentTarget.style.transform = "translateY(0)";
        } else {
          e.currentTarget.style.borderColor = "var(--border-color)";
          e.currentTarget.style.color = "var(--text-secondary)";
        }
      }}
    >
      {loading ? <Loader2 size={15} className="animate-spin" /> : <Icon size={15} />}
      {label}
    </button>
  );
}

function EndpointRow({
  method,
  path,
  description,
}: {
  method: string;
  path: string;
  description: string;
}) {
  return (
    <div className="flex items-center gap-3">
      <span
        className="text-[10px] font-bold px-2 py-0.5 rounded"
        style={{
          background:
            method === "POST"
              ? "rgba(52, 211, 153, 0.12)"
              : "rgba(99, 102, 241, 0.12)",
          color:
            method === "POST" ? "var(--success)" : "var(--accent)",
        }}
      >
        {method}
      </span>
      <span
        className="text-xs font-mono"
        style={{ color: "var(--text-primary)" }}
      >
        {path}
      </span>
      <span
        className="text-[11px]"
        style={{ color: "var(--text-muted)" }}
      >
        — {description}
      </span>
    </div>
  );
}
