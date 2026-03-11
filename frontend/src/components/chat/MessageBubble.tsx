"use client";

import ReactMarkdown from "react-markdown";
import { User, Bot, Copy, Check } from "lucide-react";
import { useState } from "react";

interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
  index?: number;
}

export default function MessageBubble({
  role,
  content,
  isStreaming,
  index = 0,
}: MessageBubbleProps) {
  const [copied, setCopied] = useState(false);
  const isUser = role === "user";

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={`flex gap-3 py-3 px-4 group ${isUser ? "justify-end" : ""}`}
      style={{
        animation: isStreaming ? "none" : `fadeIn 0.35s ease-out both`,
      }}
    >
      {!isUser && (
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 mt-1"
          style={{
            background:
              "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
            boxShadow: "0 2px 8px rgba(99, 102, 241, 0.2)",
          }}
        >
          <Bot size={15} color="white" />
        </div>
      )}

      <div className={`flex flex-col max-w-[80%] ${isUser ? "items-end" : ""}`}>
        <div
          className="rounded-2xl px-4 py-3 text-sm leading-relaxed"
          style={{
            background: isUser
              ? "linear-gradient(135deg, var(--gradient-start), var(--gradient-mid))"
              : "var(--bg-secondary)",
            color: isUser ? "#fff" : "var(--text-primary)",
            border: isUser ? "none" : "1px solid var(--border-color)",
            boxShadow: isUser
              ? "0 2px 12px rgba(99, 102, 241, 0.2)"
              : "0 1px 4px rgba(0, 0, 0, 0.15)",
          }}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{content}</p>
          ) : (
            <div className="markdown-content">
              <ReactMarkdown>{content}</ReactMarkdown>
              {isStreaming && (
                <span
                  className="inline-block w-2 h-5 ml-0.5 rounded-sm"
                  style={{
                    background: "var(--accent)",
                    animation: "pulse 1s ease-in-out infinite",
                  }}
                />
              )}
            </div>
          )}
        </div>

        {/* Copy button for assistant messages */}
        {!isUser && content && !isStreaming && (
          <button
            onClick={handleCopy}
            className="opacity-0 group-hover:opacity-100 mt-1.5 px-2.5 py-1 rounded-lg transition-smooth text-xs flex items-center gap-1.5"
            style={{
              color: copied ? "var(--success)" : "var(--text-muted)",
              background: copied ? "rgba(52, 211, 153, 0.08)" : "transparent",
            }}
            onMouseEnter={(e) => {
              if (!copied) {
                e.currentTarget.style.background = "var(--bg-hover)";
                e.currentTarget.style.color = "var(--text-secondary)";
              }
            }}
            onMouseLeave={(e) => {
              if (!copied) {
                e.currentTarget.style.background = "transparent";
                e.currentTarget.style.color = "var(--text-muted)";
              }
            }}
          >
            {copied ? <Check size={12} /> : <Copy size={12} />}
            {copied ? "Copied!" : "Copy"}
          </button>
        )}
      </div>

      {isUser && (
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 mt-1"
          style={{
            background: "var(--bg-tertiary)",
            border: "1px solid var(--border-color)",
          }}
        >
          <User size={15} style={{ color: "var(--text-secondary)" }} />
        </div>
      )}
    </div>
  );
}
