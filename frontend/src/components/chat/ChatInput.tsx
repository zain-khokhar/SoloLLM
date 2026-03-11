"use client";

import { useState, useRef, KeyboardEvent } from "react";
import { Send, Square, Paperclip } from "lucide-react";

interface ChatInputProps {
  onSend: (message: string) => void;
  onStop: () => void;
  isGenerating: boolean;
  disabled?: boolean;
}

export default function ChatInput({
  onSend,
  onStop,
  isGenerating,
  disabled,
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (isGenerating) return;
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    // Auto-resize
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  };

  return (
    <div className="px-4 pb-4 pt-2">
      <div className="max-w-3xl mx-auto">
        <div
          className="glass-input flex items-end gap-2 px-4 py-3"
          style={{
            boxShadow: input.trim()
              ? "var(--shadow-glow)"
              : "0 2px 12px rgba(0,0,0,0.2)",
          }}
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Message SoloLLM..."
            rows={1}
            className="flex-1 bg-transparent resize-none outline-none text-sm leading-relaxed"
            style={{
              color: "var(--text-primary)",
              maxHeight: "200px",
              caretColor: "var(--accent)",
            }}
            disabled={disabled}
          />

          <div className="flex items-center gap-1.5">
            {isGenerating ? (
              <button
                onClick={onStop}
                className="p-2.5 rounded-xl transition-smooth"
                style={{
                  background: "var(--error)",
                  boxShadow: "0 2px 8px rgba(248, 113, 113, 0.3)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.boxShadow =
                    "0 4px 16px rgba(248, 113, 113, 0.4)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.boxShadow =
                    "0 2px 8px rgba(248, 113, 113, 0.3)";
                }}
                title="Stop generating"
              >
                <Square size={14} fill="white" color="white" />
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!input.trim() || disabled}
                className="p-2.5 rounded-xl transition-smooth disabled:opacity-20"
                style={{
                  background: input.trim()
                    ? "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))"
                    : "var(--bg-tertiary)",
                  boxShadow: input.trim()
                    ? "0 2px 12px rgba(99, 102, 241, 0.3)"
                    : "none",
                }}
                onMouseEnter={(e) => {
                  if (input.trim()) {
                    e.currentTarget.style.boxShadow =
                      "0 4px 20px rgba(99, 102, 241, 0.4)";
                    e.currentTarget.style.transform = "translateY(-1px)";
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.boxShadow = input.trim()
                    ? "0 2px 12px rgba(99, 102, 241, 0.3)"
                    : "none";
                  e.currentTarget.style.transform = "translateY(0)";
                }}
                title="Send message"
              >
                <Send size={14} color="white" />
              </button>
            )}
          </div>
        </div>

        <p
          className="text-center text-[11px] mt-2.5 tracking-wide"
          style={{ color: "var(--text-muted)" }}
        >
          SoloLLM — 100% local · 100% private · your data never leaves
        </p>
      </div>
    </div>
  );
}
