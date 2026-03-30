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
  Sparkles,
  Brain,
  Bot,
  Activity,
  ArrowDownUp,
  HardDrive,
  GraduationCap,
  BookOpen,
} from "lucide-react";
import { Conversation } from "@/types";
import { listConversations, deleteConversation } from "@/lib/api";

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
  refreshTrigger: number;
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
  refreshTrigger,
}: SidebarProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [collapsed, setCollapsed] = useState(false);

  const loadConversations = useCallback(async () => {
    try {
      const convos = await listConversations();
      setConversations(convos);
    } catch {
      // Backend might not be running yet
    }
  }, []);

  useEffect(() => {
    loadConversations();
  }, [loadConversations, refreshTrigger]);

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

  // Collapsed sidebar
  if (collapsed) {
    return (
      <div
        className="w-14 h-screen flex flex-col items-center py-4 gap-2 transition-smooth"
        style={{
          background: "var(--bg-secondary)",
          borderRight: "1px solid var(--border-color)",
        }}
      >
        <button
          onClick={() => setCollapsed(false)}
          className="p-2.5 rounded-xl transition-smooth"
          style={{ color: "var(--text-secondary)" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--bg-hover)";
            e.currentTarget.style.color = "var(--text-primary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "var(--text-secondary)";
          }}
          title="Expand sidebar"
        >
          <ChevronRight size={18} />
        </button>

        <button
          onClick={onNewChat}
          className="p-2.5 rounded-xl transition-smooth"
          style={{ background: "var(--accent-muted)", color: "var(--accent)" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--accent)";
            e.currentTarget.style.color = "white";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "var(--accent-muted)";
            e.currentTarget.style.color = "var(--accent)";
          }}
          title="New chat"
        >
          <Plus size={18} />
        </button>

        <div className="flex-1" />

        <button
          onClick={onOpenGraph}
          className="p-2.5 rounded-xl transition-smooth"
          style={{ color: "var(--text-muted)" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--bg-hover)";
            e.currentTarget.style.color = "var(--text-primary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "var(--text-muted)";
          }}
          title="Memory Inspector"
        >
          <Brain size={18} />
        </button>

        <button
          onClick={onOpenAgent}
          className="p-2.5 rounded-xl transition-smooth"
          style={{ color: "var(--text-muted)" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--bg-hover)";
            e.currentTarget.style.color = "var(--text-primary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "var(--text-muted)";
          }}
          title="Agent Mode"
        >
          <Bot size={18} />
        </button>

        <button
          onClick={onOpenDashboard}
          className="p-2.5 rounded-xl transition-smooth"
          style={{ color: "var(--text-muted)" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--bg-hover)";
            e.currentTarget.style.color = "var(--text-primary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "var(--text-muted)";
          }}
          title="Dashboard"
        >
          <Activity size={18} />
        </button>

        <button
          onClick={onOpenModels}
          className="p-2.5 rounded-xl transition-smooth"
          style={{ color: "var(--text-muted)" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--bg-hover)";
            e.currentTarget.style.color = "var(--text-primary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "var(--text-muted)";
          }}
          title="Models"
        >
          <HardDrive size={18} />
        </button>

        <button
          onClick={onOpenTraining}
          className="p-2.5 rounded-xl transition-smooth"
          style={{ color: "var(--text-muted)" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--bg-hover)";
            e.currentTarget.style.color = "var(--text-primary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "var(--text-muted)";
          }}
          title="Self-Training"
        >
          <GraduationCap size={18} />
        </button>

        <button
          onClick={onOpenAcademic}
          className="p-2.5 rounded-xl transition-smooth"
          style={{ color: "var(--text-muted)" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--bg-hover)";
            e.currentTarget.style.color = "var(--text-primary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "var(--text-muted)";
          }}
          title="Academic Studio"
        >
          <BookOpen size={18} />
        </button>

        <button
          onClick={onOpenExport}
          className="p-2.5 rounded-xl transition-smooth"
          style={{ color: "var(--text-muted)" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--bg-hover)";
            e.currentTarget.style.color = "var(--text-primary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "var(--text-muted)";
          }}
          title="Export / Import"
        >
          <ArrowDownUp size={18} />
        </button>

        <button
          onClick={onOpenSettings}
          className="p-2.5 rounded-xl transition-smooth"
          style={{ color: "var(--text-muted)" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--bg-hover)";
            e.currentTarget.style.color = "var(--text-primary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "var(--text-muted)";
          }}
          title="Settings"
        >
          <Settings size={18} />
        </button>
      </div>
    );
  }

  return (
    <div
      className="w-72 h-screen flex flex-col animate-slideInLeft"
      style={{
        background: "var(--bg-secondary)",
        borderRight: "1px solid var(--border-color)",
      }}
    >
      {/* Header */}
      <div
        className="p-4 flex items-center justify-between"
        style={{
          borderBottom: "1px solid var(--border-color)",
          background: "linear-gradient(180deg, rgba(99, 102, 241, 0.04), transparent)",
        }}
      >
        <div className="flex items-center gap-2.5">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{
              background: "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
              boxShadow: "0 2px 12px rgba(99, 102, 241, 0.3)",
            }}
          >
            <Cpu size={16} color="white" />
          </div>
          <div>
            <span className="font-semibold text-sm gradient-text">SoloLLM</span>
            <span
              className="block text-[10px] -mt-0.5"
              style={{ color: "var(--text-muted)" }}
            >
              Local AI Platform
            </span>
          </div>
        </div>
        <button
          onClick={() => setCollapsed(true)}
          className="p-1.5 rounded-lg transition-smooth"
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
          <ChevronLeft size={16} />
        </button>
      </div>

      {/* New Chat Button */}
      <div className="p-3">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-smooth hover-lift"
          style={{
            background: "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
            color: "white",
            boxShadow: "0 2px 12px rgba(99, 102, 241, 0.25)",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.boxShadow = "0 4px 20px rgba(99, 102, 241, 0.4)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.boxShadow = "0 2px 12px rgba(99, 102, 241, 0.25)";
          }}
        >
          <Sparkles size={15} />
          New Chat
        </button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto px-2 py-1">
        {conversations.length === 0 ? (
          <div className="text-center py-12 px-4">
            <MessageSquare
              size={32}
              className="mx-auto mb-3"
              style={{ color: "var(--text-muted)", opacity: 0.5 }}
            />
            <p className="text-sm" style={{ color: "var(--text-muted)" }}>
              No conversations yet
            </p>
            <p className="text-xs mt-1" style={{ color: "var(--text-muted)", opacity: 0.7 }}>
              Start a new chat to begin
            </p>
          </div>
        ) : (
          conversations.map((conv, index) => (
            <button
              key={conv.id}
              onClick={() => onSelectConversation(conv.id)}
              className="w-full flex items-start gap-2.5 px-3 py-2.5 rounded-xl text-sm mb-0.5 group transition-smooth text-left"
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
                animation: `fadeIn 0.3s ease-out ${index * 0.03}s both`,
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
                size={14}
                className="shrink-0 mt-0.5"
                style={{
                  color:
                    activeConversationId === conv.id
                      ? "var(--accent)"
                      : "var(--text-muted)",
                }}
              />
              <div className="flex-1 min-w-0">
                <span className="block truncate text-[13px] leading-snug">
                  {conv.title}
                </span>
                <span
                  className="block text-[10px] mt-0.5"
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
                className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg transition-smooth shrink-0 cursor-pointer"
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
                <Trash2 size={12} />
              </div>
            </button>
          ))
        )}
      </div>

      {/* Bottom */}
      <div
        className="p-3 space-y-1"
        style={{ borderTop: "1px solid var(--border-color)" }}
      >
        <button
          onClick={onOpenGraph}
          className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm transition-smooth"
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
          <Brain size={15} />
          Memory Inspector
        </button>
        <button
          onClick={onOpenAgent}
          className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm transition-smooth"
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
          <Bot size={15} />
          Agent Mode
        </button>
        <button
          onClick={onOpenDashboard}
          className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm transition-smooth"
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
          <Activity size={15} />
          Dashboard
        </button>
        <button
          onClick={onOpenModels}
          className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm transition-smooth"
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
          <HardDrive size={15} />
          Models
        </button>
        <button
          onClick={onOpenTraining}
          className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm transition-smooth"
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
          <GraduationCap size={15} />
          Self-Training
        </button>
        <button
          onClick={onOpenAcademic}
          className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm transition-smooth"
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
          <BookOpen size={15} />
          Academic Studio
        </button>
        <button
          onClick={onOpenExport}
          className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm transition-smooth"
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
          <ArrowDownUp size={15} />
          Export / Import
        </button>
        <button
          onClick={onOpenSettings}
          className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm transition-smooth"
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
          <Settings size={15} />
          Settings
        </button>
      </div>
    </div>
  );
}
