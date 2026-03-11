"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import Sidebar from "@/components/sidebar/Sidebar";
import ChatArea from "@/components/chat/ChatArea";
import ChatInput from "@/components/chat/ChatInput";
import ModelSelector from "@/components/chat/ModelSelector";
import KnowledgeGraphView from "@/components/graph/KnowledgeGraphView";
import AgentView from "@/components/agents/AgentView";
import DashboardView from "@/components/dashboard/DashboardView";
import ExportImportView from "@/components/export/ExportImportView";
import SetupWizard from "@/components/setup/SetupWizard";
import ModelPicker from "@/components/setup/ModelPicker";
import { streamChat, streamContinuation, getConversation, getSettings, updateSettings, getSystemProfile, runProfiler, checkHealth, listModels } from "@/lib/api";
import { DistillationMeta } from "@/types";
import {
  ArrowLeft,
  RefreshCw,
  Save,
  Cpu,
  Monitor,
  HardDrive,
  Zap,
  Settings,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

interface TruncationInfo {
  message_id: string;
  conversation_id: string;
  reason: string;
  confidence: number;
}

export default function Home() {
  const [showSetup, setShowSetup] = useState<boolean | null>(null); // null = checking
  const [currentView, setCurrentView] = useState<"chat" | "settings" | "graph" | "agent" | "dashboard" | "export" | "models">("chat");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [selectedModel, setSelectedModel] = useState("llama3.2:1b");
  const [truncation, setTruncation] = useState<TruncationInfo | null>(null);
  const [isContinuing, setIsContinuing] = useState(false);
  const [refreshSidebar, setRefreshSidebar] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [distillationMeta, setDistillationMeta] = useState<DistillationMeta | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Check if we need to show the setup wizard
  useEffect(() => {
    const checkSetup = async () => {
      try {
        const health = await checkHealth();
        if (health.ollama_connected) {
          // Ollama is running, check if any models installed
          const models = await listModels().catch(() => []);
          setShowSetup(models.length === 0);
        } else {
          // Ollama not connected — show setup
          setShowSetup(true);
        }
      } catch {
        // Backend not reachable — show setup
        setShowSetup(true);
      }
    };
    checkSetup();
  }, []);

  const handleNewChat = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
    setConversationId(null);
    setMessages([]);
    setStreamingContent("");
    setIsStreaming(false);
    setTruncation(null);
    setError(null);
    setDistillationMeta(null);
    setCurrentView("chat");
  }, []);

  const handleSelectConversation = useCallback(async (id: string) => {
    try {
      setCurrentView("chat");
      const data = await getConversation(id);
      setConversationId(id);
      setMessages(
        data.messages
          .filter((m) => m.role !== "system")
          .map((m) => ({
            id: m.id,
            role: m.role as "user" | "assistant",
            content: m.content,
          }))
      );
      setTruncation(null);
      setError(null);
      setSelectedModel(data.conversation.model);
    } catch {
      setError("Failed to load conversation");
    }
  }, []);

  const handleSend = useCallback(
    (message: string) => {
      setError(null);
      setTruncation(null);
      setDistillationMeta(null);

      const userMsg: ChatMessage = {
        id: `temp-user-${Date.now()}`,
        role: "user",
        content: message,
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsStreaming(true);
      setStreamingContent("");

      let accumulated = "";

      const controller = streamChat(
        {
          message,
          conversation_id: conversationId || undefined,
          model: selectedModel,
        },
        {
          onToken: (content) => {
            accumulated += content;
            setStreamingContent(accumulated);
          },
          onDone: (data) => {
            setIsStreaming(false);
            setConversationId(data.conversation_id);
            setMessages((prev) => [
              ...prev,
              {
                id: data.message_id,
                role: "assistant",
                content: accumulated,
              },
            ]);
            setStreamingContent("");
            setRefreshSidebar((prev) => prev + 1);
          },
          onTruncated: (data) => {
            setIsStreaming(false);
            setConversationId(data.conversation_id);
            setMessages((prev) => [
              ...prev,
              {
                id: data.message_id,
                role: "assistant",
                content: accumulated,
              },
            ]);
            setStreamingContent("");
            setTruncation({
              message_id: data.message_id,
              conversation_id: data.conversation_id,
              reason: data.reason,
              confidence: data.confidence,
            });
            setRefreshSidebar((prev) => prev + 1);
          },
          onError: (err) => {
            setIsStreaming(false);
            setStreamingContent("");
            setError(err);
          },
          onDistillation: (data) => {
            setDistillationMeta(data);
          },
        }
      );

      abortRef.current = controller;
    },
    [conversationId, selectedModel]
  );

  const handleStop = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      setIsStreaming(false);
      if (streamingContent) {
        setMessages((prev) => [
          ...prev,
          {
            id: `stopped-${Date.now()}`,
            role: "assistant",
            content: streamingContent,
          },
        ]);
      }
      setStreamingContent("");
    }
  }, [streamingContent]);

  const handleContinue = useCallback(() => {
    if (!truncation) return;
    setIsContinuing(true);
    setIsStreaming(true);
    setStreamingContent("");

    let accumulated = "";

    const controller = streamContinuation(
      {
        conversation_id: truncation.conversation_id,
        message_id: truncation.message_id,
      },
      {
        onToken: (content) => {
          accumulated += content;
          setStreamingContent(accumulated);
        },
        onDone: () => {
          setIsStreaming(false);
          setIsContinuing(false);
          setTruncation(null);
          setMessages((prevMsgs) =>
            prevMsgs.map((m) =>
              m.id === truncation.message_id
                ? { ...m, content: m.content + accumulated }
                : m
            )
          );
          setStreamingContent("");
        },
        onTruncated: (data) => {
          setIsStreaming(false);
          setIsContinuing(false);
          setMessages((prevMsgs) =>
            prevMsgs.map((m) =>
              m.id === truncation.message_id
                ? { ...m, content: m.content + accumulated }
                : m
            )
          );
          setStreamingContent("");
          setTruncation({
            message_id: data.message_id,
            conversation_id: data.conversation_id,
            reason: data.reason,
            confidence: data.confidence,
          });
        },
        onError: (err) => {
          setIsStreaming(false);
          setIsContinuing(false);
          setError(err);
        },
      }
    );

    abortRef.current = controller;
  }, [truncation]);

  return (
    <div className="flex h-screen">
      {/* Setup wizard shown on first launch or when Ollama not available */}
      {showSetup === null ? (
        <div className="flex-1 flex items-center justify-center" style={{ background: "var(--bg-primary)" }}>
          <div className="w-8 h-8 rounded-lg animate-pulse" style={{ background: "var(--accent-muted)" }} />
        </div>
      ) : showSetup ? (
        <SetupWizard onComplete={() => setShowSetup(false)} />
      ) : (
        <>
      <Sidebar
        activeConversationId={conversationId}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
        onOpenSettings={() => setCurrentView("settings")}
        onOpenGraph={() => setCurrentView("graph")}
        onOpenAgent={() => setCurrentView("agent")}
        onOpenDashboard={() => setCurrentView("dashboard")}
        onOpenExport={() => setCurrentView("export")}
        onOpenModels={() => setCurrentView("models")}
        refreshTrigger={refreshSidebar}
      />

      {currentView === "settings" ? (
        <SettingsView onBack={() => setCurrentView("chat")} />
      ) : currentView === "graph" ? (
        <KnowledgeGraphView onBack={() => setCurrentView("chat")} />
      ) : currentView === "agent" ? (
        <AgentView onBack={() => setCurrentView("chat")} selectedModel={selectedModel} />
      ) : currentView === "dashboard" ? (
        <DashboardView onBack={() => setCurrentView("chat")} />
      ) : currentView === "export" ? (
        <ExportImportView
          onBack={() => setCurrentView("chat")}
          onImportComplete={() => setRefreshSidebar((prev) => prev + 1)}
        />
      ) : currentView === "models" ? (
        <div className="flex-1 flex flex-col h-screen" style={{ background: "var(--bg-primary)" }}>
          <div
            className="flex items-center gap-3 px-4 py-2.5"
            style={{
              borderBottom: "1px solid var(--border-color)",
              background: "rgba(18, 18, 26, 0.5)",
              backdropFilter: "blur(12px)",
            }}
          >
            <button
              onClick={() => setCurrentView("chat")}
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
            <span className="font-semibold gradient-text">Manage Models</span>
          </div>
          <ModelPicker onModelInstalled={() => {}} onSkip={() => setCurrentView("chat")} />
        </div>
      ) : (
        <div className="flex-1 flex flex-col h-screen">
          {/* Top bar */}
          <div
            className="flex items-center justify-between px-4 py-2.5"
            style={{
              borderBottom: "1px solid var(--border-color)",
              background: "rgba(18, 18, 26, 0.5)",
              backdropFilter: "blur(12px)",
            }}
          >
            <ModelSelector
              selectedModel={selectedModel}
              onSelectModel={setSelectedModel}
            />
            {error && (
              <div
                className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg animate-slideDown"
                style={{
                  color: "var(--error)",
                  background: "rgba(248, 113, 113, 0.08)",
                  border: "1px solid rgba(248, 113, 113, 0.15)",
                }}
              >
                <AlertCircle size={13} />
                {error}
              </div>
            )}
          </div>

          <ChatArea
            messages={messages}
            isStreaming={isStreaming}
            streamingContent={streamingContent}
            truncation={truncation}
            onContinue={handleContinue}
            isContinuing={isContinuing}
            distillationMeta={distillationMeta}
          />

          <ChatInput
            onSend={handleSend}
            onStop={handleStop}
            isGenerating={isStreaming}
          />
        </div>
      )}
        </>
      )}
    </div>
  );
}

// ── Settings View ──────────────────────────────────────────

interface SettingsForm {
  ollama_base_url: string;
  default_model: string;
  max_tokens: number;
  temperature: number;
  auto_continue: boolean;
  system_prompt: string;
}

function SettingsView({ onBack }: { onBack: () => void }) {
  const [profile, setProfile] = useState<Record<string, unknown> | null>(null);
  const [formData, setFormData] = useState<SettingsForm | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [profiling, setProfiling] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [profileData, settingsData] = await Promise.all([
          getSystemProfile().catch(() => null),
          getSettings().catch(() => null),
        ]);
        setProfile(profileData as unknown as Record<string, unknown>);
        if (settingsData) {
          setFormData({
            ollama_base_url: settingsData.ollama_base_url || "http://localhost:11434",
            default_model: settingsData.default_model || "llama3.2:latest",
            max_tokens: settingsData.max_tokens || 2048,
            temperature: settingsData.temperature || 0.7,
            auto_continue: settingsData.auto_continue ?? true,
            system_prompt: settingsData.system_prompt || "",
          });
        }
      } catch {
        setSettingsError("Could not connect to backend");
      }
      setLoading(false);
    };
    load();
  }, []);

  const handleSave = async () => {
    if (!formData) return;
    setSaving(true);
    setSettingsError(null);
    try {
      await updateSettings(formData);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch {
      setSettingsError("Failed to save settings");
    }
    setSaving(false);
  };

  const handleReprofile = async () => {
    setProfiling(true);
    try {
      const newProfile = await runProfiler();
      setProfile(newProfile as unknown as Record<string, unknown>);
    } catch {
      setSettingsError("Failed to run profiler");
    }
    setProfiling(false);
  };

  const updateField = (key: keyof SettingsForm, value: string | number | boolean) => {
    if (!formData) return;
    setFormData({ ...formData, [key]: value });
  };

  return (
    <div className="flex-1 overflow-y-auto" style={{ background: "var(--bg-primary)" }}>
      <div className="max-w-2xl mx-auto p-6 animate-fadeIn">
        {/* Header */}
        <div className="flex items-center gap-3 mb-8">
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
            <h1 className="text-xl font-bold gradient-text">Settings</h1>
            <p className="text-xs" style={{ color: "var(--text-muted)" }}>
              Configure your SoloLLM instance
            </p>
          </div>
        </div>

        {settingsError && (
          <div
            className="mb-6 px-4 py-3 rounded-xl flex items-center gap-2 text-sm animate-slideDown"
            style={{
              background: "rgba(248, 113, 113, 0.08)",
              border: "1px solid rgba(248, 113, 113, 0.15)",
              color: "var(--error)",
            }}
          >
            <AlertCircle size={16} />
            {settingsError}
          </div>
        )}

        {loading ? (
          <div className="text-center py-20">
            <div
              className="w-10 h-10 rounded-xl mx-auto mb-3 animate-pulse"
              style={{ background: "var(--accent-muted)" }}
            />
            <p style={{ color: "var(--text-muted)" }}>Loading settings...</p>
          </div>
        ) : (
          <div className="space-y-6">
            {/* System Profile */}
            <section>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Monitor size={16} style={{ color: "var(--accent)" }} />
                  <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                    System Profile
                  </h2>
                </div>
                <button
                  onClick={handleReprofile}
                  disabled={profiling}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-smooth disabled:opacity-50"
                  style={{
                    background: "var(--bg-tertiary)",
                    border: "1px solid var(--border-color)",
                    color: "var(--text-secondary)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "var(--accent)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "var(--border-color)";
                  }}
                >
                  <RefreshCw size={12} className={profiling ? "animate-spin" : ""} />
                  {profiling ? "Scanning..." : "Re-scan"}
                </button>
              </div>

              <div
                className="rounded-xl p-4 grid grid-cols-2 gap-3"
                style={{
                  background: "var(--bg-secondary)",
                  border: "1px solid var(--border-color)",
                }}
              >
                {profile ? (
                  <>
                    <ProfileItem icon={Cpu} label="GPU" value={String(profile.gpu_name || "None detected")} />
                    <ProfileItem icon={HardDrive} label="VRAM" value={profile.vram_mb ? `${profile.vram_mb} MB` : "N/A"} />
                    <ProfileItem icon={HardDrive} label="RAM" value={profile.ram_mb ? `${Math.round(Number(profile.ram_mb) / 1024)} GB` : "N/A"} />
                    <ProfileItem icon={Cpu} label="CPU" value={String(profile.cpu_name || "Unknown")} />
                    <ProfileItem icon={Zap} label="Cores" value={String(profile.cpu_cores || "Unknown")} />
                    <ProfileItem icon={Monitor} label="OS" value={String(profile.os_info || "Unknown")} />
                  </>
                ) : (
                  <p className="col-span-2 text-sm" style={{ color: "var(--text-muted)" }}>
                    Could not load system profile. Is the backend running?
                  </p>
                )}
              </div>
            </section>

            {/* Application Settings */}
            {formData && (
              <section>
                <div className="flex items-center gap-2 mb-3">
                  <Settings size={16} style={{ color: "var(--accent)" }} />
                  <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                    Application Settings
                  </h2>
                </div>

                <div
                  className="rounded-xl p-5 space-y-4"
                  style={{
                    background: "var(--bg-secondary)",
                    border: "1px solid var(--border-color)",
                  }}
                >
                  <SettingsField
                    label="Ollama URL"
                    desc="Base URL for the Ollama API server"
                    value={formData.ollama_base_url}
                    onChange={(v) => updateField("ollama_base_url", v)}
                  />
                  <SettingsField
                    label="Default Model"
                    desc="Model to use when starting a new chat"
                    value={formData.default_model}
                    onChange={(v) => updateField("default_model", v)}
                  />
                  <SettingsField
                    label="Max Tokens"
                    desc="Maximum output tokens per response (higher = longer answers)"
                    value={String(formData.max_tokens)}
                    onChange={(v) => updateField("max_tokens", parseInt(v) || 2048)}
                    type="number"
                  />
                  <SettingsField
                    label="Temperature"
                    desc="Creativity level (0.0 = focused, 1.0 = creative)"
                    value={String(formData.temperature)}
                    onChange={(v) => updateField("temperature", parseFloat(v) || 0.7)}
                    type="number"
                    step="0.1"
                  />

                  {/* Auto-Continue Toggle */}
                  <div className="flex items-center justify-between py-2">
                    <div>
                      <label className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                        Auto-Continue
                      </label>
                      <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
                        Detect truncated responses and offer continuation
                      </p>
                    </div>
                    <button
                      onClick={() => updateField("auto_continue", !formData.auto_continue)}
                      className="w-11 h-6 rounded-full transition-smooth relative"
                      style={{
                        background: formData.auto_continue
                          ? "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))"
                          : "var(--bg-tertiary)",
                        border: `1px solid ${formData.auto_continue ? "transparent" : "var(--border-color)"}`,
                      }}
                    >
                      <div
                        className="w-4 h-4 rounded-full absolute top-0.5 transition-all duration-200"
                        style={{
                          background: "white",
                          left: formData.auto_continue ? "calc(100% - 20px)" : "3px",
                          boxShadow: "0 1px 3px rgba(0,0,0,0.3)",
                        }}
                      />
                    </button>
                  </div>

                  {/* System Prompt */}
                  <div>
                    <label className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                      System Prompt
                    </label>
                    <p className="text-xs mt-0.5 mb-2" style={{ color: "var(--text-muted)" }}>
                      Default system instructions for all conversations
                    </p>
                    <textarea
                      value={formData.system_prompt}
                      onChange={(e) => updateField("system_prompt", e.target.value)}
                      rows={3}
                      className="w-full rounded-xl px-3.5 py-2.5 text-sm resize-none outline-none transition-smooth"
                      style={{
                        background: "var(--bg-primary)",
                        border: "1px solid var(--border-color)",
                        color: "var(--text-primary)",
                        caretColor: "var(--accent)",
                      }}
                      onFocus={(e) => {
                        e.currentTarget.style.borderColor = "var(--accent)";
                        e.currentTarget.style.boxShadow = "0 0 12px rgba(99, 102, 241, 0.1)";
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = "var(--border-color)";
                        e.currentTarget.style.boxShadow = "none";
                      }}
                      placeholder="You are a helpful assistant..."
                    />
                  </div>

                  {/* Save Button */}
                  <div className="pt-3 flex justify-end">
                    <button
                      onClick={handleSave}
                      disabled={saving}
                      className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-smooth disabled:opacity-50"
                      style={{
                        background: saved
                          ? "rgba(52, 211, 153, 0.15)"
                          : "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
                        color: saved ? "var(--success)" : "white",
                        border: saved ? "1px solid rgba(52, 211, 153, 0.3)" : "none",
                        boxShadow: saved ? "none" : "0 2px 12px rgba(99, 102, 241, 0.25)",
                      }}
                      onMouseEnter={(e) => {
                        if (!saved && !saving) {
                          e.currentTarget.style.boxShadow = "0 4px 20px rgba(99, 102, 241, 0.4)";
                          e.currentTarget.style.transform = "translateY(-1px)";
                        }
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.boxShadow = saved ? "none" : "0 2px 12px rgba(99, 102, 241, 0.25)";
                        e.currentTarget.style.transform = "translateY(0)";
                      }}
                    >
                      {saved ? (
                        <>
                          <CheckCircle2 size={15} />
                          Saved!
                        </>
                      ) : (
                        <>
                          <Save size={15} />
                          {saving ? "Saving..." : "Save Settings"}
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </section>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────

function ProfileItem({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ComponentType<{ size: number; style?: React.CSSProperties }>;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-start gap-2.5 py-1">
      <div
        className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5"
        style={{ background: "var(--accent-muted)" }}
      >
        <Icon size={13} style={{ color: "var(--accent)" }} />
      </div>
      <div className="min-w-0">
        <p className="text-[11px] font-medium" style={{ color: "var(--text-muted)" }}>
          {label}
        </p>
        <p className="text-sm font-medium truncate" style={{ color: "var(--text-primary)" }}>
          {value}
        </p>
      </div>
    </div>
  );
}

function SettingsField({
  label,
  desc,
  value,
  onChange,
  type = "text",
  step,
}: {
  label: string;
  desc: string;
  value: string;
  onChange: (val: string) => void;
  type?: string;
  step?: string;
}) {
  return (
    <div>
      <label className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
        {label}
      </label>
      <p className="text-xs mt-0.5 mb-1.5" style={{ color: "var(--text-muted)" }}>
        {desc}
      </p>
      <input
        type={type}
        step={step}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-xl px-3.5 py-2.5 text-sm outline-none transition-smooth"
        style={{
          background: "var(--bg-primary)",
          border: "1px solid var(--border-color)",
          color: "var(--text-primary)",
          caretColor: "var(--accent)",
        }}
        onFocus={(e) => {
          e.currentTarget.style.borderColor = "var(--accent)";
          e.currentTarget.style.boxShadow = "0 0 12px rgba(99, 102, 241, 0.1)";
        }}
        onBlur={(e) => {
          e.currentTarget.style.borderColor = "var(--border-color)";
          e.currentTarget.style.boxShadow = "none";
        }}
      />
    </div>
  );
}
