"use client";

import { useRef, useEffect } from "react";
import {
  Cpu,
  Zap,
  MessageSquare,
  FileText,
  Shield,
  Brain,
  Activity,
  Target,
  Layers,
  GitBranch,
} from "lucide-react";
import MessageBubble from "./MessageBubble";
import { DistillationMeta } from "@/types";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

interface ChatAreaProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  streamingContent: string;
  distillationMeta?: DistillationMeta | null;
}

export default function ChatArea({
  messages,
  isStreaming,
  streamingContent,
  distillationMeta,
}: ChatAreaProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  // Empty state
  if (messages.length === 0 && !isStreaming) {
    return (
      <div className="flex-1 flex items-center justify-center animate-fadeIn">
        <div className="text-center max-w-lg px-6">
          {/* Animated logo */}
          <div className="relative w-20 h-20 mx-auto mb-6">
            <div
              className="absolute inset-0 rounded-2xl animate-pulse-glow"
              style={{
                background:
                  "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
                opacity: 0.15,
              }}
            />
            <div
              className="relative w-20 h-20 rounded-2xl flex items-center justify-center animate-float"
              style={{
                background:
                  "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
                boxShadow: "0 8px 32px rgba(99, 102, 241, 0.25)",
              }}
            >
              <Cpu size={36} color="white" />
            </div>
          </div>

          <h2 className="text-2xl font-bold mb-2 gradient-text">
            Welcome to SoloLLM
          </h2>
          <p
            className="text-sm mb-8 max-w-sm mx-auto leading-relaxed"
            style={{ color: "var(--text-secondary)" }}
          >
            Your private, local AI assistant. All processing happens on your
            machine — your data never leaves.
          </p>

          <div className="grid grid-cols-2 gap-3 text-left">
            {[
              {
                icon: Zap,
                label: "Fast & Local",
                desc: "Runs entirely on your hardware",
                delay: "0.1s",
              },
              {
                icon: Brain,
                label: "Smart Memory",
                desc: "Tiered memory for precise retrieval",
                delay: "0.15s",
              },
              {
                icon: FileText,
                label: "Document RAG",
                desc: "Deep document intelligence with citations",
                delay: "0.2s",
              },
              {
                icon: Shield,
                label: "100% Private",
                desc: "Your data never leaves your machine",
                delay: "0.25s",
              },
            ].map(({ icon: Icon, label, desc, delay }) => (
              <div
                key={label}
                className="p-4 rounded-xl transition-smooth hover-lift"
                style={{
                  background: "var(--bg-secondary)",
                  border: "1px solid var(--border-color)",
                  animation: `fadeIn 0.5s ease-out ${delay} both`,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "var(--accent)";
                  e.currentTarget.style.background = "var(--accent-muted)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "var(--border-color)";
                  e.currentTarget.style.background = "var(--bg-secondary)";
                }}
              >
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center mb-2.5"
                  style={{
                    background: "var(--accent-muted)",
                  }}
                >
                  <Icon size={16} style={{ color: "var(--accent)" }} />
                </div>
                <p className="text-sm font-medium mb-0.5">{label}</p>
                <p
                  className="text-xs leading-relaxed"
                  style={{ color: "var(--text-muted)" }}
                >
                  {desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-3xl mx-auto py-4">
        {messages.map((msg, index) => (
          <MessageBubble
            key={msg.id}
            role={msg.role}
            content={msg.content}
            index={index}
          />
        ))}

        {/* Streaming message */}
        {isStreaming && (
          <MessageBubble
            role="assistant"
            content={streamingContent}
            isStreaming={true}
            index={messages.length}
          />
        )}

        {/* Distillation info panel */}
        {distillationMeta && !isStreaming && (
          <div
            className="mx-4 mb-3 p-3 rounded-xl animate-fadeIn"
            style={{
              background: "var(--bg-secondary)",
              border: "1px solid var(--border-color)",
            }}
          >
            <div className="flex items-center gap-2 mb-2">
              <Activity size={14} style={{ color: "var(--accent)" }} />
              <span
                className="text-xs font-medium"
                style={{ color: "var(--text-secondary)" }}
              >
                Context Distillation
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {/* Confidence */}
              <div
                className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg"
                style={{ background: "var(--bg-primary)" }}
              >
                <Target
                  size={12}
                  style={{
                    color:
                      distillationMeta.confidence.level === "high"
                        ? "var(--success)"
                        : distillationMeta.confidence.level === "medium"
                        ? "var(--warning, #f59e0b)"
                        : "var(--error)",
                  }}
                />
                <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                  Confidence:{" "}
                  <span
                    className="font-medium"
                    style={{
                      color:
                        distillationMeta.confidence.level === "high"
                          ? "var(--success)"
                          : distillationMeta.confidence.level === "medium"
                          ? "var(--warning, #f59e0b)"
                          : "var(--error)",
                    }}
                  >
                    {Math.round(distillationMeta.confidence.overall * 100)}%
                  </span>
                </span>
              </div>

              {/* Compression */}
              {distillationMeta.compression_ratio < 1 && (
                <div
                  className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg"
                  style={{ background: "var(--bg-primary)" }}
                >
                  <Layers size={12} style={{ color: "var(--accent)" }} />
                  <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                    Compressed:{" "}
                    <span className="font-medium" style={{ color: "var(--text-secondary)" }}>
                      {Math.round((1 - distillationMeta.compression_ratio) * 100)}%
                    </span>
                  </span>
                </div>
              )}

              {/* Query type */}
              <div
                className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg"
                style={{ background: "var(--bg-primary)" }}
              >
                <Brain size={12} style={{ color: "var(--accent)" }} />
                <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                  Type:{" "}
                  <span className="font-medium" style={{ color: "var(--text-secondary)" }}>
                    {distillationMeta.query_type}
                  </span>
                </span>
              </div>

              {/* Multi-hop */}
              {distillationMeta.hops_used > 1 && (
                <div
                  className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg"
                  style={{ background: "var(--bg-primary)" }}
                >
                  <GitBranch size={12} style={{ color: "var(--accent)" }} />
                  <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                    Hops:{" "}
                    <span className="font-medium" style={{ color: "var(--text-secondary)" }}>
                      {distillationMeta.hops_used}
                    </span>
                  </span>
                </div>
              )}
            </div>

            {/* Citations */}
            {distillationMeta.citations && distillationMeta.citations.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {distillationMeta.citations.map((cite) => (
                  <span
                    key={cite.index}
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs"
                    style={{
                      background: "var(--accent-muted)",
                      color: "var(--accent)",
                      border: "1px solid rgba(99, 102, 241, 0.15)",
                    }}
                  >
                    <FileText size={10} />
                    [{cite.index}] {cite.document_title}
                    {cite.page_number ? ` p.${cite.page_number}` : ""}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        <div ref={bottomRef} className="h-4" />
      </div>
    </div>
  );
}
