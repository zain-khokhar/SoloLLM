"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Plus,
  MessageSquare,
  Trash2,
  Lock,
  Edit3,
  Check,
  X,
  GitBranch,
} from "lucide-react";
import { Thread } from "@/types";
import {
  listThreads,
  createThread,
  updateThread,
  deleteThread,
} from "@/lib/api";

interface ThreadPanelProps {
  conversationId: string | null;
  activeThreadId: string | null;
  onSelectThread: (threadId: string) => void;
  onThreadCreated?: (thread: Thread) => void;
  refreshTrigger?: number;
}

export default function ThreadPanel({
  conversationId,
  activeThreadId,
  onSelectThread,
  onThreadCreated,
  refreshTrigger = 0,
}: ThreadPanelProps) {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [creating, setCreating] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");

  const loadThreads = useCallback(async () => {
    if (!conversationId) {
      setThreads([]);
      return;
    }
    try {
      const t = await listThreads(conversationId);
      setThreads(t);
    } catch {
      // backend not ready
    }
  }, [conversationId]);

  useEffect(() => {
    loadThreads();
  }, [loadThreads, refreshTrigger]);

  const handleCreate = async () => {
    if (!conversationId || !newTitle.trim()) return;
    try {
      const thread = await createThread(conversationId, newTitle.trim());
      setThreads((prev) => [...prev, thread]);
      setNewTitle("");
      setCreating(false);
      onThreadCreated?.(thread);
      onSelectThread(thread.id);
    } catch {
      // ignore
    }
  };

  const handleDelete = async (e: React.MouseEvent, threadId: string) => {
    e.stopPropagation();
    try {
      await deleteThread(threadId);
      setThreads((prev) => prev.filter((t) => t.id !== threadId));
      if (activeThreadId === threadId && threads.length > 1) {
        const remaining = threads.filter((t) => t.id !== threadId);
        onSelectThread(remaining[0].id);
      }
    } catch {
      // Can't delete default thread
    }
  };

  const handleRename = async (threadId: string) => {
    if (!editTitle.trim()) {
      setEditingId(null);
      return;
    }
    try {
      await updateThread(threadId, { title: editTitle.trim() });
      setThreads((prev) =>
        prev.map((t) =>
          t.id === threadId ? { ...t, title: editTitle.trim() } : t
        )
      );
      setEditingId(null);
    } catch {
      setEditingId(null);
    }
  };

  if (!conversationId) return null;

  return (
    <div
      style={{
        borderBottom: "1px solid var(--border-color)",
        background: "linear-gradient(180deg, rgba(18, 18, 26, 0.6), rgba(10, 10, 15, 0.4))",
      }}
    >
      {/* Thread tabs row */}
      <div className="flex items-center gap-1 px-3 py-2 overflow-x-auto">
        {/* Label */}
        <div className="flex items-center gap-1.5 mr-2 shrink-0">
          <GitBranch size={13} style={{ color: "var(--accent)" }} />
          <span
            className="text-[11px] font-semibold uppercase tracking-wider"
            style={{ color: "var(--text-muted)" }}
          >
            Threads
          </span>
        </div>

        {/* Thread tabs */}
        {threads.map((thread) => {
          const isActive = activeThreadId === thread.id;
          return (
            <div
              key={thread.id}
              onClick={() => onSelectThread(thread.id)}
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs cursor-pointer group transition-all duration-200 shrink-0 relative"
              style={{
                background: isActive
                  ? "linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1))"
                  : "transparent",
                border: isActive
                  ? "1px solid rgba(99, 102, 241, 0.3)"
                  : "1px solid transparent",
                color: isActive ? "var(--text-primary)" : "var(--text-secondary)",
                boxShadow: isActive
                  ? "0 2px 8px rgba(99, 102, 241, 0.15), inset 0 1px 0 rgba(255,255,255,0.05)"
                  : "none",
                minWidth: "100px",
              }}
              onMouseEnter={(e) => {
                if (!isActive) {
                  e.currentTarget.style.background = "var(--bg-hover)";
                  e.currentTarget.style.border = "1px solid var(--border-color)";
                }
              }}
              onMouseLeave={(e) => {
                if (!isActive) {
                  e.currentTarget.style.background = "transparent";
                  e.currentTarget.style.border = "1px solid transparent";
                }
              }}
            >
              {/* Active indicator bar */}
              {isActive && (
                <div
                  className="absolute -bottom-1.25 left-1/2 -translate-x-1/2 w-6 h-0.5 rounded-full"
                  style={{ background: "var(--accent)" }}
                />
              )}

              {thread.context_mode === "isolated" && (
                <Lock size={10} className="shrink-0" style={{ color: isActive ? "var(--accent)" : "var(--text-muted)" }} />
              )}

              {editingId === thread.id ? (
                <div className="flex items-center gap-1 flex-1" onClick={(e) => e.stopPropagation()}>
                  <input
                    type="text"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleRename(thread.id);
                      if (e.key === "Escape") setEditingId(null);
                    }}
                    className="flex-1 bg-transparent outline-none text-xs font-medium"
                    style={{
                      color: "var(--text-primary)",
                      borderBottom: "1px solid var(--accent)",
                      minWidth: "60px",
                    }}
                    autoFocus
                  />
                  <button onClick={() => handleRename(thread.id)} className="p-0.5">
                    <Check size={11} style={{ color: "var(--success)" }} />
                  </button>
                  <button onClick={() => setEditingId(null)} className="p-0.5">
                    <X size={11} style={{ color: "var(--error)" }} />
                  </button>
                </div>
              ) : (
                <>
                  <MessageSquare
                    size={11}
                    style={{
                      color: isActive ? "var(--accent)" : "var(--text-muted)",
                      flexShrink: 0,
                    }}
                  />
                  <span className="truncate font-medium" style={{ maxWidth: "120px" }}>
                    {thread.title}
                  </span>

                  <div className="opacity-0 group-hover:opacity-100 flex items-center gap-0.5 transition-all duration-200 shrink-0 ml-auto">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditingId(thread.id);
                        setEditTitle(thread.title);
                      }}
                      className="p-1 rounded transition-colors"
                      style={{ color: "var(--text-muted)" }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.color = "var(--text-primary)";
                        e.currentTarget.style.background = "var(--bg-hover)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.color = "var(--text-muted)";
                        e.currentTarget.style.background = "transparent";
                      }}
                    >
                      <Edit3 size={10} />
                    </button>
                    <button
                        onClick={(e) => handleDelete(e, thread.id)}
                        className="p-1 rounded transition-colors"
                        style={{ color: "var(--text-muted)" }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.color = "var(--error)";
                          e.currentTarget.style.background = "rgba(248,113,113,0.1)";
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.color = "var(--text-muted)";
                          e.currentTarget.style.background = "transparent";
                        }}
                      >
                        <Trash2 size={10} />
                      </button>
                  </div>
                </>
              )}
            </div>
          );
        })}

        {/* New thread input / button */}
        {creating ? (
          <div
            className="flex items-center gap-2 px-3 py-2 rounded-lg shrink-0"
            style={{
              background: "var(--bg-secondary)",
              border: "1px solid var(--accent)",
              boxShadow: "0 0 8px rgba(99, 102, 241, 0.2)",
            }}
          >
            <MessageSquare size={11} style={{ color: "var(--accent)" }} />
            <input
              type="text"
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleCreate();
                if (e.key === "Escape") {
                  setCreating(false);
                  setNewTitle("");
                }
              }}
              placeholder="Thread name..."
              className="bg-transparent outline-none text-xs font-medium"
              style={{
                color: "var(--text-primary)",
                width: "120px",
              }}
              autoFocus
            />
            <button onClick={handleCreate} className="p-0.5" style={{ color: "var(--success)" }}>
              <Check size={12} />
            </button>
            <button
              onClick={() => { setCreating(false); setNewTitle(""); }}
              className="p-0.5"
              style={{ color: "var(--error)" }}
            >
              <X size={12} />
            </button>
          </div>
        ) : (
          <button
            onClick={() => setCreating(true)}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-[11px] font-semibold transition-all duration-200 shrink-0"
            style={{
              color: "var(--accent)",
              background: "transparent",
              border: "1px dashed rgba(99, 102, 241, 0.3)",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "var(--accent-muted)";
              e.currentTarget.style.borderColor = "var(--accent)";
              e.currentTarget.style.borderStyle = "solid";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "transparent";
              e.currentTarget.style.borderColor = "rgba(99, 102, 241, 0.3)";
              e.currentTarget.style.borderStyle = "dashed";
            }}
          >
            <Plus size={12} />
            New Thread
          </button>
        )}
      </div>
    </div>
  );
}
