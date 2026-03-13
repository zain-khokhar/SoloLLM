"use client";

import { useState, useRef, KeyboardEvent } from "react";
import { Send, Square, Paperclip, Loader2 } from "lucide-react";

interface ChatInputProps {
  onSend: (message: string) => void;
  onStop: () => void;
  isGenerating: boolean;
  disabled?: boolean;
  onUploadDocument?: (file: File) => void;
  isUploading?: boolean;
  contextLimit?: number | null;
}

export default function ChatInput({
  onSend,
  onStop,
  isGenerating,
  disabled,
  onUploadDocument,
  isUploading,
  contextLimit,
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const [isDragOver, setIsDragOver] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && onUploadDocument) {
      onUploadDocument(file);
    }
    if (e.target) e.target.value = "";
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (onUploadDocument) setIsDragOver(true);
  };

  const handleDragLeave = () => setIsDragOver(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && onUploadDocument) onUploadDocument(file);
  };

  return (
    <div className="px-4 pb-4 pt-2">
      <div className="max-w-3xl mx-auto">
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept=".pdf,.doc,.docx,.txt,.md,.html,.csv,.py,.js,.ts,.java,.c,.cpp,.json,.xml"
          onChange={handleFileSelect}
        />

        <div
          className="glass-input flex items-end gap-2 px-4 py-3"
          style={{
            boxShadow: isDragOver
              ? "0 0 20px rgba(99, 102, 241, 0.3)"
              : input.trim()
                ? "var(--shadow-glow)"
                : "0 2px 12px rgba(0,0,0,0.2)",
            border: isDragOver ? "1px solid var(--accent)" : undefined,
          }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {/* Upload button */}
          {onUploadDocument && (
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading || disabled}
              className="p-2 rounded-xl transition-smooth disabled:opacity-30"
              style={{
                color: isUploading ? "var(--accent)" : "var(--text-muted)",
                background: "transparent",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.color = "var(--accent)";
                e.currentTarget.style.background = "var(--accent-muted)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.color = isUploading ? "var(--accent)" : "var(--text-muted)";
                e.currentTarget.style.background = "transparent";
              }}
              title="Attach document to thread"
            >
              {isUploading ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Paperclip size={16} />
              )}
            </button>
          )}

          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={isDragOver ? "Drop file to upload..." : "Message SoloLLM..."}
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

        {/* Context limit indicator */}
        {contextLimit && (
          <div
            className="flex items-center gap-2 mt-1.5 px-1"
            style={{ color: "var(--text-muted)" }}
          >
            <div
              className="flex-1 h-0.5 rounded-full overflow-hidden"
              style={{ background: "var(--bg-tertiary)" }}
            >
              <div
                className="h-0.5 rounded-full transition-all duration-300"
                style={{
                  width: "0%",
                  background: "var(--accent)",
                }}
              />
            </div>
            <span className="text-[10px] shrink-0">
              {Math.round(contextLimit / 1024)}K context
            </span>
          </div>
        )}

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
