"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import Sidebar from "@/components/sidebar/Sidebar";
import OwnerLoginView from "@/components/auth/OwnerLoginView";
import ChatArea from "@/components/chat/ChatArea";
import ChatInput from "@/components/chat/ChatInput";
import ModelSelector from "@/components/chat/ModelSelector";
import ThreadPanel from "@/components/chat/ThreadPanel";
import ThreadSettingsPanel from "@/components/chat/ThreadSettingsPanel";
import KnowledgeGraphView from "@/components/graph/KnowledgeGraphView";
import AgentView from "@/components/agents/AgentView";
import DashboardView from "@/components/dashboard/DashboardView";
import ExportImportView from "@/components/export/ExportImportView";
import SetupWizard from "@/components/setup/SetupWizard";
import ModelPicker from "@/components/setup/ModelPicker";
import TrainingView from "@/components/training/TrainingView";
import AcademicStudio from "@/components/academic/AcademicStudio";
import ModelQuantizer from "@/components/quantize/ModelQuantizer";
import { addAuthRequiredListener, checkHealth, clearStoredAuthToken, getAuthConfig, getAuthSession, getBackendCapabilities, getConversation, getSettings, getSystemProfile, getThread, getThreadDocuments, listDocuments, listModels, loginOwner, logoutOwner, runProfiler, streamChat, streamWebSearch, updateSettings, uploadDocument, attachDocumentToThread, detachDocumentFromThread } from "@/lib/api";
import { AuthConfig, BackendCapabilities, DistillationMeta, Thread, DocumentInfo } from "@/types";
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
  Sliders,
  FileText,
  X,
} from "lucide-react";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

type AppView =
  | "chat"
  | "settings"
  | "graph"
  | "agent"
  | "dashboard"
  | "export"
  | "models"
  | "training"
  | "academic"
  | "quantize";

function isViewEnabled(view: AppView, capabilities: BackendCapabilities | null): boolean {
  if (!capabilities) {
    return true;
  }

  switch (view) {
    case "agent":
      return capabilities.features.agent;
    case "export":
      return capabilities.features.export_import;
    case "training":
      return capabilities.features.training;
    case "academic":
      return capabilities.features.academic;
    case "quantize":
      return capabilities.features.quantize;
    default:
      return true;
  }
}

export default function Home() {
  const [showSetup, setShowSetup] = useState<boolean | null>(null); // null = checking
  const [currentView, setCurrentView] = useState<AppView>("chat");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [showThreadSettings, setShowThreadSettings] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [selectedModel, setSelectedModel] = useState("llama3.2:1b");
  const [refreshSidebar, setRefreshSidebar] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [distillationMeta, setDistillationMeta] = useState<DistillationMeta | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [threadDocuments, setThreadDocuments] = useState<{ document_id: string; filename: string }[]>([]);
  const [allDocuments, setAllDocuments] = useState<DocumentInfo[]>([]);
  const [contextLimit, setContextLimit] = useState<number | null>(null);
  const [authConfig, setAuthConfig] = useState<AuthConfig | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [authenticating, setAuthenticating] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [capabilities, setCapabilities] = useState<BackendCapabilities | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const privateAccessEnabled = authConfig?.private_access_enabled ?? false;
  const featureVisibility = {
    agent: capabilities?.features.agent ?? false,
    training: capabilities?.features.training ?? false,
    academic: capabilities?.features.academic ?? false,
    quantize: capabilities?.features.quantize ?? false,
    exportImport: capabilities?.features.export_import ?? false,
  };

  const bootstrapAuth = useCallback(async () => {
    setAuthReady(false);
    setAuthError(null);
    try {
      const config = await getAuthConfig();
      setAuthConfig(config);

      if (!config.private_access_enabled) {
        setIsAuthenticated(true);
        return;
      }

      const session = await getAuthSession();
      if (!session.authenticated) {
        clearStoredAuthToken();
      }
      setIsAuthenticated(session.authenticated);
    } catch {
      setAuthError("Could not connect to the backend auth service.");
      setIsAuthenticated(false);
    } finally {
      setAuthReady(true);
    }
  }, []);

  const refreshDocumentCatalog = useCallback(async () => {
    try {
      const docs = await listDocuments();
      setAllDocuments(docs || []);
      return docs || [];
    } catch {
      setAllDocuments([]);
      return [] as DocumentInfo[];
    }
  }, []);

  const refreshThreadDocuments = useCallback(async (threadId: string | null, docsCatalog?: DocumentInfo[]) => {
    if (!threadId) {
      setThreadDocuments([]);
      return;
    }
    try {
      const attached = await getThreadDocuments(threadId);
      const docs = docsCatalog || allDocuments;
      const mapped = (attached.documents || []).map((doc) => {
        const fullDoc = docs.find((d) => d.document_id === doc.document_id);
        return {
          document_id: doc.document_id,
          filename: fullDoc?.filename || doc.document_id,
        };
      });
      setThreadDocuments(mapped);
    } catch {
      setThreadDocuments([]);
    }
  }, [allDocuments]);

  useEffect(() => {
    void bootstrapAuth();
  }, [bootstrapAuth]);

  useEffect(() => {
    return addAuthRequiredListener(() => {
      setIsAuthenticated(false);
      setCapabilities(null);
      setShowThreadSettings(false);
      setAuthError("Your owner session is no longer valid. Sign in again.");
    });
  }, []);

  useEffect(() => {
    if (!authReady || (privateAccessEnabled && !isAuthenticated)) {
      return;
    }

    const checkSetup = async () => {
      try {
        setShowSetup(null);
        const [deploymentCapabilities, health] = await Promise.all([
          getBackendCapabilities().catch(() => null),
          checkHealth(),
        ]);
        setCapabilities(deploymentCapabilities);

        if (!deploymentCapabilities?.deployment.runtime_management_enabled && !health.ollama_connected) {
          setShowSetup(false);
          return;
        }

        if (health.ollama_connected) {
          const models = await listModels().catch(() => []);
          setShowSetup(models.length === 0);
          if (models.length > 0) {
            setSelectedModel((prev) => {
              if (!models.some((m: { name: string }) => m.name === prev)) {
                return models[0].name;
              }
              return prev;
            });
          }
        } else if (deploymentCapabilities?.deployment.runtime_management_enabled ?? true) {
          setShowSetup(true);
        } else {
          setShowSetup(false);
        }
      } catch {
        setShowSetup(true);
      }
    };
    void checkSetup();
  }, [authReady, isAuthenticated, privateAccessEnabled]);

  useEffect(() => {
    if (currentView !== "chat" && !isViewEnabled(currentView, capabilities)) {
      setCurrentView("chat");
    }
  }, [capabilities, currentView]);

  const handleNewChat = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
    setConversationId(null);
    setActiveThreadId(null);
    setThreads([]);
    setShowThreadSettings(false);
    setMessages([]);
    setStreamingContent("");
    setIsStreaming(false);
    setIsSearching(false);
    setError(null);
    setDistillationMeta(null);
    setThreadDocuments([]);
    setCurrentView("chat");
  }, []);

  const handleLogin = useCallback(async (username: string, password: string) => {
    setAuthenticating(true);
    setAuthError(null);
    try {
      await loginOwner(username, password);
      setIsAuthenticated(true);
      setShowSetup(null);
    } catch (loginError) {
      setIsAuthenticated(false);
      setAuthError(loginError instanceof Error ? loginError.message : "Login failed");
    } finally {
      setAuthenticating(false);
    }
  }, []);

  const handleLogout = useCallback(async () => {
    await logoutOwner();
    handleNewChat();
    setIsAuthenticated(false);
    setCapabilities(null);
    setShowSetup(null);
    setAuthError(null);
  }, [handleNewChat]);

  const handleSelectConversation = useCallback(async (id: string) => {
    try {
      setCurrentView("chat");
      setShowThreadSettings(false);
      const data = await getConversation(id);
      setConversationId(id);
      setThreads(data.threads || []);
      // Set active thread to default thread
      const defaultThreadId = data.default_thread_id || (data.threads?.[0]?.id ?? null);
      setActiveThreadId(defaultThreadId);
      setSelectedModel(data.conversation.model);
      setError(null);
      setDistillationMeta(null);

      // Load ONLY this thread's messages (context isolation!)
      if (defaultThreadId) {
        try {
          const threadData = await getThread(defaultThreadId);
          setMessages(
            threadData.messages
              .filter((m: { role: string }) => m.role !== "system")
              .map((m: { id: string; role: string; content: string }) => ({
                id: m.id,
                role: m.role as "user" | "assistant",
                content: m.content,
              }))
          );
        } catch {
          // Fallback: show no messages for fresh start
          setMessages([]);
        }
      } else {
        setMessages([]);
      }
    } catch {
      setError("Failed to load conversation");
    }
  }, []);

  const handleSelectThread = useCallback(async (threadId: string) => {
    if (!conversationId) return;
    setActiveThreadId(threadId);
    setShowThreadSettings(false);
    try {
      const data = await getThread(threadId);
      setMessages(
        data.messages
          .filter((m: { role: string }) => m.role !== "system")
          .map((m: { id: string; role: string; content: string }) => ({
            id: m.id,
            role: m.role as "user" | "assistant",
            content: m.content,
          }))
      );
      setError(null);
      setDistillationMeta(null);
    } catch {
      setError("Failed to load thread");
    }
  }, [conversationId]);

  const handleSend = useCallback(
    (message: string, options?: { documentsOnly?: boolean }) => {
      setError(null);
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
          thread_id: activeThreadId || undefined,
          model: selectedModel,
          documents_only: options?.documentsOnly ?? false,
        },
        {
          onToken: (content) => {
            accumulated += content;
            setStreamingContent(accumulated);
          },
          onDone: (data) => {
            setIsStreaming(false);
            setConversationId(data.conversation_id);
            if (data.thread_id) setActiveThreadId(data.thread_id);
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
          onTruncated: () => {
            // Truncation disabled — treat as done
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
    [conversationId, activeThreadId, selectedModel]
  );

  const handleStop = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      setIsStreaming(false);
      setIsSearching(false);
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

  // Load document catalog on startup
  useEffect(() => {
    refreshDocumentCatalog();
  }, [refreshDocumentCatalog]);

  // Load thread documents when active thread or catalog changes
  useEffect(() => {
    void refreshThreadDocuments(activeThreadId);
  }, [activeThreadId, allDocuments, refreshThreadDocuments]);

  const handleUploadDocument = useCallback(async (file: File) => {
    if (!activeThreadId) {
      setError("Please start a conversation first, then upload documents to a thread.");
      return;
    }
    setIsUploading(true);
    setError(null);
    try {
      // Upload & embed (RAG pipeline: parse → chunk → embed → vector DB)
      const result = await uploadDocument(file);
      if (result.success && result.document_id) {
        // Attach to current thread for scoped RAG retrieval
        await attachDocumentToThread(activeThreadId, result.document_id);
        const docs = await refreshDocumentCatalog();
        await refreshThreadDocuments(activeThreadId, docs);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Document upload failed");
    }
    setIsUploading(false);
  }, [activeThreadId, refreshDocumentCatalog, refreshThreadDocuments]);

  const handleDetachDocument = useCallback(async (documentId: string) => {
    if (!activeThreadId) return;
    try {
      await detachDocumentFromThread(activeThreadId, documentId);
      await refreshThreadDocuments(activeThreadId);
    } catch {
      // ignore
    }
  }, [activeThreadId, refreshThreadDocuments]);

  const handleToggleThreadDocument = useCallback(async (documentId: string) => {
    if (!activeThreadId) return;
    const isAttached = threadDocuments.some((d) => d.document_id === documentId);
    try {
      if (isAttached) {
        await detachDocumentFromThread(activeThreadId, documentId);
      } else {
        await attachDocumentToThread(activeThreadId, documentId);
      }
      await refreshThreadDocuments(activeThreadId);
    } catch {
      // ignore
    }
  }, [activeThreadId, threadDocuments, refreshThreadDocuments]);

  const handleWebSearch = useCallback(
    (query: string) => {
      setError(null);
      setDistillationMeta(null);

      // Show user message immediately
      const userMsg: ChatMessage = {
        id: `temp-search-${Date.now()}`,
        role: "user",
        content: `[Web Search] ${query}`,
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsSearching(true);
      setStreamingContent("");

      let accumulated = "";

      const controller = streamWebSearch(
        {
          query,
          model: selectedModel,
          conversation_id: conversationId || undefined,
          thread_id: activeThreadId || undefined,
        },
        {
          onStatus: (_phase, _message) => {
            // Could show status in the streaming area
          },
          onSources: (_results) => {
            // Sources received — could display them later
          },
          onToken: (content) => {
            accumulated += content;
            setStreamingContent(accumulated);
          },
          onDone: (data) => {
            setIsSearching(false);
            setConversationId(data.conversation_id);
            if (data.thread_id) setActiveThreadId(data.thread_id);
            setMessages((prev) => [
              ...prev,
              {
                id: data.message_id || `search-${Date.now()}`,
                role: "assistant",
                content: accumulated,
              },
            ]);
            setStreamingContent("");
            setRefreshSidebar((prev) => prev + 1);
          },
          onError: (err) => {
            setIsSearching(false);
            setStreamingContent("");
            setError(err);
          },
        }
      );

      abortRef.current = controller;
    },
    [conversationId, activeThreadId, selectedModel]
  );

  return (
    <div className="flex h-screen">
      {!authReady ? (
        <div className="flex-1 flex items-center justify-center" style={{ background: "var(--bg-primary)" }}>
          <div className="w-8 h-8 rounded-lg animate-pulse" style={{ background: "var(--accent-muted)" }} />
        </div>
      ) : privateAccessEnabled && !isAuthenticated ? (
        <OwnerLoginView
          ownerUsername={authConfig?.owner_username || "admin"}
          loading={authenticating}
          error={authError}
          publicBaseUrl={authConfig?.public_base_url}
          onLogin={handleLogin}
        />
      ) : showSetup === null ? (
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
        onOpenTraining={() => setCurrentView("training")}
        onOpenAcademic={() => setCurrentView("academic")}
        onOpenQuantize={() => setCurrentView("quantize")}
        onLogout={handleLogout}
        featureVisibility={featureVisibility}
        refreshTrigger={refreshSidebar}
      />

      {currentView === "settings" ? (
        <SettingsView onBack={() => setCurrentView("chat")} />
      ) : currentView === "graph" ? (
        <KnowledgeGraphView onBack={() => setCurrentView("chat")} />
      ) : currentView === "agent" && featureVisibility.agent ? (
        <AgentView onBack={() => setCurrentView("chat")} selectedModel={selectedModel} />
      ) : currentView === "dashboard" ? (
        <DashboardView onBack={() => setCurrentView("chat")} />
      ) : currentView === "export" && featureVisibility.exportImport ? (
        <ExportImportView
          onBack={() => setCurrentView("chat")}
          onImportComplete={() => setRefreshSidebar((prev) => prev + 1)}
        />
      ) : currentView === "training" && featureVisibility.training ? (
        <TrainingView onBack={() => setCurrentView("chat")} selectedModel={selectedModel} onSelectModel={setSelectedModel} />
      ) : currentView === "academic" && featureVisibility.academic ? (
        <div className="flex-1 overflow-y-auto" style={{ background: "var(--bg-primary)" }}>
          <AcademicStudio />
        </div>
      ) : currentView === "quantize" && featureVisibility.quantize ? (
        <ModelQuantizer onBack={() => setCurrentView("chat")} />
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
        <div className="flex-1 flex flex-col h-screen relative">
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
              onContextLengthChange={setContextLimit}
            />
            <div className="flex items-center gap-2">
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
              {conversationId && activeThreadId && (
                <button
                  onClick={() => setShowThreadSettings(!showThreadSettings)}
                  className="p-2 rounded-lg transition-smooth"
                  style={{
                    color: showThreadSettings ? "var(--accent)" : "var(--text-muted)",
                    background: showThreadSettings ? "var(--accent-muted)" : "transparent",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = "var(--bg-hover)";
                    e.currentTarget.style.color = "var(--accent)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = showThreadSettings ? "var(--accent-muted)" : "transparent";
                    e.currentTarget.style.color = showThreadSettings ? "var(--accent)" : "var(--text-muted)";
                  }}
                  title="Thread Settings"
                >
                  <Sliders size={16} />
                </button>
              )}
            </div>
          </div>

          {/* Thread Panel — shows threads under the top bar */}
          <ThreadPanel
            conversationId={conversationId}
            activeThreadId={activeThreadId}
            onSelectThread={handleSelectThread}
            onThreadCreated={(thread) => setThreads((prev) => [...prev, thread])}
            refreshTrigger={refreshSidebar}
          />

          <ChatArea
            messages={messages}
            isStreaming={isStreaming || isSearching}
            streamingContent={streamingContent}
            distillationMeta={distillationMeta}
          />

          {/* Document attachments bar */}
          {threadDocuments.length > 0 && (
            <div
              className="px-4 py-2 flex items-center gap-2 flex-wrap"
              style={{
                borderTop: "1px solid var(--border-color)",
                background: "rgba(18, 18, 26, 0.5)",
              }}
            >
              <span className="text-[10px] font-medium" style={{ color: "var(--text-muted)" }}>
                Documents:
              </span>
              {threadDocuments.map((doc) => (
                <span
                  key={doc.document_id}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px]"
                  style={{
                    background: "var(--accent-muted)",
                    color: "var(--accent)",
                    border: "1px solid rgba(99, 102, 241, 0.15)",
                  }}
                >
                  <FileText size={10} />
                  {doc.filename}
                  <button
                    onClick={() => handleDetachDocument(doc.document_id)}
                    className="ml-0.5 p-0.5 rounded hover:bg-red-500/20 transition-colors"
                    style={{ color: "var(--text-muted)" }}
                    title="Remove from thread"
                  >
                    <X size={8} />
                  </button>
                </span>
              ))}
            </div>
          )}

          <ChatInput
            onSend={handleSend}
            onStop={handleStop}
            isGenerating={isStreaming}
            onUploadDocument={handleUploadDocument}
            isUploading={isUploading}
            contextLimit={contextLimit}
            onWebSearch={handleWebSearch}
            isSearching={isSearching}
            documentsAttached={threadDocuments.length > 0}
            availableDocuments={allDocuments.map((doc) => ({
              document_id: doc.document_id,
              filename: doc.filename,
            }))}
            selectedDocumentIds={threadDocuments.map((doc) => doc.document_id)}
            onToggleDocumentReference={handleToggleThreadDocument}
          />

          {/* Thread Settings Slide-over Panel */}
          {showThreadSettings && (
            <ThreadSettingsPanel
              threadId={activeThreadId}
              onClose={() => setShowThreadSettings(false)}
            />
          )}
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
  temperature: number;
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
            ollama_base_url: settingsData.ollama_base_url || "http://127.0.0.1:11434",
            default_model: settingsData.default_model || "llama3.2:latest",
            temperature: settingsData.temperature || 0.7,
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
                    label="Temperature"
                    desc="Creativity level (0.0 = focused, 1.0 = creative)"
                    value={String(formData.temperature)}
                    onChange={(v) => updateField("temperature", parseFloat(v) || 0.7)}
                    type="number"
                    step="0.1"
                  />

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
