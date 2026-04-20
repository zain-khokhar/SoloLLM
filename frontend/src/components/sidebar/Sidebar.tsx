"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Plus,
  MessageSquare,
  Trash2,
  Settings,
  Cpu,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  Sparkles,
  Brain,
  Bot,
  Activity,
  ArrowDownUp,
  HardDrive,
  GraduationCap,
  BookOpen,
  Layers,
  X,
  LogOut,
} from "lucide-react";
import { Conversation, ModelInfo } from "@/types";
import { listConversations, deleteConversation, listModels, deleteModel } from "@/lib/api";

interface SidebarProps {
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onOpenSettings: () => void;
  onOpenGraph: () => void;
  onOpenAgent: () => void;
  onOpenDashboard: () => void;
  onOpenExport: () => void;
  onOpenModels: () => void;
  onOpenTraining: () => void;
  onOpenAcademic: () => void;
  onOpenQuantize: () => void;
  onLogout: () => void;
  featureVisibility: {
    agent: boolean;
    training: boolean;
    academic: boolean;
    quantize: boolean;
    exportImport: boolean;
  };
  refreshTrigger: number;
}

function formatSize(bytes: number | null): string {
  if (!bytes) return "";
  const gb = bytes / (1024 * 1024 * 1024);
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(0)} MB`;
}

export default function Sidebar({
  activeConversationId,
  onSelectConversation,
  onNewChat,
  onOpenSettings,
  onOpenGraph,
  onOpenAgent,
  onOpenDashboard,
  onOpenExport,
  onOpenModels,
  onOpenTraining,
  onOpenAcademic,
  onOpenQuantize,
  onLogout,
  featureVisibility,
  refreshTrigger,
}: SidebarProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [collapsed, setCollapsed] = useState(false);
  const [navExpanded, setNavExpanded] = useState(false);
  const [modelsExpanded, setModelsExpanded] = useState(false);
  const [localModels, setLocalModels] = useState<ModelInfo[]>([]);
  const [deletingModel, setDeletingModel] = useState<string | null>(null);

  const loadConversations = useCallback(async () => {
    try {
      const convos = await listConversations();
      setConversations(convos);
    } catch {
      // Backend might not be running yet
    }
  }, []);

  const loadModels = useCallback(async () => {
    try {
      const models = await listModels();
      setLocalModels(models);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    loadConversations();
    loadModels();
  }, [loadConversations, loadModels, refreshTrigger]);

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeConversationId === id) {
        onNewChat();
      }
    } catch {
      // ignore
    }
  };

  const handleDeleteModel = async (e: React.MouseEvent, name: string) => {
    e.stopPropagation();
    setDeletingModel(name);
    try {
      await deleteModel(name);
      setLocalModels((prev) => prev.filter((m) => m.name !== name));
    } catch {
      // ignore
    }
    setDeletingModel(null);
  };

  const formatTime = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "Just now";
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
  };

  // ── Collapsed sidebar ──
  if (collapsed) {
    return (
      <div
        className="w-12 h-screen flex flex-col items-center py-3 gap-1.5 transition-smooth"
        style={{
          background: "var(--bg-secondary)",
          borderRight: "1px solid var(--border-color)",
        }}
      >
        <button
          onClick={() => setCollapsed(false)}
          className="p-2 rounded-lg transition-smooth hover-btn"
          style={{ color: "var(--text-muted)" }}
          title="Expand sidebar"
        >
          <ChevronRight size={16} />
        </button>

        <button
          onClick={onNewChat}
          className="p-2 rounded-lg transition-smooth"
          style={{ background: "var(--accent-muted)", color: "var(--accent)" }}
          title="New chat"
        >
          <Plus size={16} />
        </button>

        <div className="flex-1" />

        {[
          { icon: Brain, action: onOpenGraph, title: "Memory", visible: true },
          { icon: Bot, action: onOpenAgent, title: "Agent", visible: featureVisibility.agent },
          { icon: Activity, action: onOpenDashboard, title: "Dashboard", visible: true },
          { icon: HardDrive, action: onOpenModels, title: "Models", visible: true },
          { icon: GraduationCap, action: onOpenTraining, title: "Training", visible: featureVisibility.training },
          { icon: BookOpen, action: onOpenAcademic, title: "Academic", visible: featureVisibility.academic },
          { icon: Layers, action: onOpenQuantize, title: "Quantizer", visible: featureVisibility.quantize },
          { icon: ArrowDownUp, action: onOpenExport, title: "Export", visible: featureVisibility.exportImport },
          { icon: Settings, action: onOpenSettings, title: "Settings", visible: true },
        ]
          .filter((item) => item.visible)
          .map(({ icon: Icon, action, title }) => (
          <button
            key={title}
            onClick={action}
            className="p-2 rounded-lg transition-smooth hover-btn"
            style={{ color: "var(--text-muted)" }}
            title={title}
          >
            <Icon size={15} />
          </button>
          ))}

        <button
          onClick={onLogout}
          className="p-2 rounded-lg transition-smooth hover-btn"
          style={{ color: "var(--text-muted)" }}
          title="Logout"
        >
          <LogOut size={15} />
        </button>
      </div>
    );
  }

  // ── nav items ──
  const navItems = [
    { icon: Brain, label: "Memory Inspector", action: onOpenGraph },
    { icon: Bot, label: "Agent Mode", action: onOpenAgent, visible: featureVisibility.agent },
    { icon: Activity, label: "Dashboard", action: onOpenDashboard },
    { icon: GraduationCap, label: "Self-Training", action: onOpenTraining, visible: featureVisibility.training },
    { icon: BookOpen, label: "Academic Studio", action: onOpenAcademic, visible: featureVisibility.academic },
    { icon: Layers, label: "Quantizer", action: onOpenQuantize, visible: featureVisibility.quantize },
    { icon: ArrowDownUp, label: "Export / Import", action: onOpenExport, visible: featureVisibility.exportImport },
    { icon: Settings, label: "Settings", action: onOpenSettings },
  ].filter((item) => item.visible ?? true);

  return (
    <div
      className="w-64 h-screen flex flex-col animate-slideInLeft"
      style={{
        background: "var(--bg-secondary)",
        borderRight: "1px solid var(--border-color)",
      }}
    >
      {/* Header */}
      <div
        className="px-3 py-2.5 flex items-center justify-between"
        style={{
          borderBottom: "1px solid var(--border-color)",
        }}
      >
        <div className="flex items-center gap-2">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center"
            style={{
              background: "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
              boxShadow: "0 2px 8px rgba(99, 102, 241, 0.25)",
            }}
          >
            <Cpu size={14} color="white" />
          </div>
          <div>
            <span className="font-semibold text-sm gradient-text">SoloLLM</span>
            <span
              className="block text-[9px] -mt-0.5 tracking-wide"
              style={{ color: "var(--text-muted)" }}
            >
              LOCAL AI
            </span>
          </div>
        </div>
        <button
          onClick={() => setCollapsed(true)}
          className="p-1.5 rounded-lg transition-smooth hover-btn"
          style={{ color: "var(--text-muted)" }}
        >
          <ChevronLeft size={14} />
        </button>
      </div>

      {/* New Chat */}
      <div className="px-2.5 py-2">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-smooth"
          style={{
            background: "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
            color: "white",
            boxShadow: "0 2px 8px rgba(99, 102, 241, 0.2)",
          }}
        >
          <Sparkles size={13} />
          New Chat
        </button>
      </div>

      {/* Conversations */}
      <div className="flex-1 overflow-y-auto px-1.5 py-0.5 min-h-0">
        {conversations.length === 0 ? (
          <div className="text-center py-8 px-3">
            <MessageSquare
              size={24}
              className="mx-auto mb-2"
              style={{ color: "var(--text-muted)", opacity: 0.4 }}
            />
            <p className="text-[11px]" style={{ color: "var(--text-muted)" }}>
              No conversations yet
            </p>
          </div>
        ) : (
          conversations.map((conv, index) => (
            <button
              key={conv.id}
              onClick={() => onSelectConversation(conv.id)}
              className="w-full flex items-start gap-2 px-2.5 py-2 rounded-lg text-xs mb-px group transition-smooth text-left"
              style={{
                background:
                  activeConversationId === conv.id
                    ? "var(--accent-muted)"
                    : "transparent",
                borderLeft:
                  activeConversationId === conv.id
                    ? "2px solid var(--accent)"
                    : "2px solid transparent",
                color:
                  activeConversationId === conv.id
                    ? "var(--text-primary)"
                    : "var(--text-secondary)",
                animation: `fadeIn 0.2s ease-out ${index * 0.02}s both`,
              }}
              onMouseEnter={(e) => {
                if (activeConversationId !== conv.id)
                  e.currentTarget.style.background = "var(--bg-hover)";
              }}
              onMouseLeave={(e) => {
                if (activeConversationId !== conv.id)
                  e.currentTarget.style.background = "transparent";
              }}
            >
              <MessageSquare
                size={12}
                className="shrink-0 mt-0.5"
                style={{
                  color:
                    activeConversationId === conv.id
                      ? "var(--accent)"
                      : "var(--text-muted)",
                }}
              />
              <div className="flex-1 min-w-0">
                <span className="block truncate text-[12px] leading-snug">
                  {conv.title}
                </span>
                <span
                  className="block text-[9px] mt-0.5"
                  style={{ color: "var(--text-muted)" }}
                >
                  {formatTime(conv.updated_at)}
                </span>
              </div>
              <div
                role="button"
                tabIndex={0}
                onClick={(e) => handleDelete(e, conv.id)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleDelete(e as unknown as React.MouseEvent, conv.id); }}
                className="opacity-0 group-hover:opacity-100 p-1 rounded transition-smooth shrink-0 cursor-pointer"
                style={{ color: "var(--text-muted)" }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "rgba(248, 113, 113, 0.1)";
                  e.currentTarget.style.color = "var(--error)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                  e.currentTarget.style.color = "var(--text-muted)";
                }}
                title="Delete conversation"
              >
                <Trash2 size={11} />
              </div>
            </button>
          ))
        )}
      </div>

      {/* Bottom sections */}
      <div
        className="border-t"
        style={{ borderColor: "var(--border-color)" }}
      >
        {/* ─── Models Section (expandable) ─── */}
        <div className="px-2 pt-1.5">
          <button
            onClick={() => { setModelsExpanded(!modelsExpanded); if (!modelsExpanded) loadModels(); }}
            className="w-full flex items-center justify-between px-2 py-1.5 rounded-lg text-[11px] font-semibold uppercase tracking-wider transition-smooth"
            style={{ color: "var(--text-muted)" }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "var(--bg-hover)";
              e.currentTarget.style.color = "var(--text-secondary)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "transparent";
              e.currentTarget.style.color = "var(--text-muted)";
            }}
          >
            <div className="flex items-center gap-1.5">
              <HardDrive size={12} />
              <span>Local Models</span>
              <span
                className="text-[9px] font-normal px-1.5 py-0.5 rounded-full"
                style={{ background: "var(--bg-hover)", color: "var(--text-muted)" }}
              >
                {localModels.length}
              </span>
            </div>
            <ChevronDown
              size={12}
              className="transition-transform duration-200"
              style={{ transform: modelsExpanded ? "rotate(180deg)" : "rotate(0deg)" }}
            />
          </button>

          {modelsExpanded && (
            <div className="mt-1 mb-1 space-y-px max-h-40 overflow-y-auto">
              {localModels.length === 0 ? (
                <p className="text-[10px] px-2 py-2 text-center" style={{ color: "var(--text-muted)" }}>
                  No models installed
                </p>
              ) : (
                localModels.map((model) => (
                  <div
                    key={model.name}
                    className="flex items-center justify-between px-2 py-1.5 rounded-lg group transition-smooth"
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = "var(--bg-hover)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = "transparent";
                    }}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span
                          className="w-1.5 h-1.5 rounded-full shrink-0"
                          style={{ background: "var(--success)" }}
                        />
                        <span
                          className="text-[11px] truncate"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {model.name}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 ml-3 mt-0.5">
                        {model.parameter_size && (
                          <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>
                            {model.parameter_size}
                          </span>
                        )}
                        {model.size && (
                          <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>
                            {formatSize(model.size)}
                          </span>
                        )}
                        {model.quantization_level && (
                          <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>
                            {model.quantization_level}
                          </span>
                        )}
                      </div>
                    </div>
                    <button
                      onClick={(e) => handleDeleteModel(e, model.name)}
                      disabled={deletingModel === model.name}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded transition-smooth shrink-0 disabled:opacity-50"
                      style={{ color: "var(--text-muted)" }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = "rgba(248, 113, 113, 0.15)";
                        e.currentTarget.style.color = "var(--error)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = "transparent";
                        e.currentTarget.style.color = "var(--text-muted)";
                      }}
                      title={`Delete ${model.name}`}
                    >
                      {deletingModel === model.name ? (
                        <div className="w-3 h-3 rounded-full border-2 border-current border-t-transparent animate-spin" />
                      ) : (
                        <X size={12} />
                      )}
                    </button>
                  </div>
                ))
              )}
              <button
                onClick={onOpenModels}
                className="w-full text-[10px] font-medium py-1.5 mt-0.5 rounded-lg transition-smooth"
                style={{ color: "var(--accent)" }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "var(--accent-muted)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                }}
              >
                Manage Models
              </button>
            </div>
          )}
        </div>

        {/* ─── Navigation (collapsible) ─── */}
        <div className="px-2 pb-2">
          <button
            onClick={() => setNavExpanded(!navExpanded)}
            className="w-full flex items-center justify-between px-2 py-1.5 rounded-lg text-[11px] font-semibold uppercase tracking-wider transition-smooth"
            style={{ color: "var(--text-muted)" }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "var(--bg-hover)";
              e.currentTarget.style.color = "var(--text-secondary)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "transparent";
              e.currentTarget.style.color = "var(--text-muted)";
            }}
          >
            <span>Tools & Features</span>
            <ChevronDown
              size={12}
              className="transition-transform duration-200"
              style={{ transform: navExpanded ? "rotate(180deg)" : "rotate(0deg)" }}
            />
          </button>

          {navExpanded && (
            <div className="mt-1 space-y-px">
              {navItems.map(({ icon: Icon, label, action }) => (
                <button
                  key={label}
                  onClick={action}
                  className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-[12px] transition-smooth"
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
                  <Icon size={13} />
                  {label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="px-2.5 py-2" style={{ borderTop: "1px solid var(--border-color)" }}>
        <button
          onClick={onLogout}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-smooth"
          style={{
            background: "var(--bg-tertiary)",
            color: "var(--text-secondary)",
            border: "1px solid var(--border-color)",
          }}
        >
          <LogOut size={13} />
          Logout
        </button>
      </div>
    </div>
  );
}
