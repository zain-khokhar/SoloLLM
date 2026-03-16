"use client";

import { useState, useRef, KeyboardEvent } from "react";
import { Send, Square, Paperclip, Loader2, Globe, FileText } from "lucide-react";

interface ChatInputProps {
  onSend: (message: string, options?: { documentsOnly?: boolean }) => void;
  onStop: () => void;
  isGenerating: boolean;
  disabled?: boolean;
  onUploadDocument?: (file: File) => void;
  isUploading?: boolean;
  contextLimit?: number | null;
  onWebSearch?: (query: string) => void;
  isSearching?: boolean;
  documentsAttached?: boolean;
  availableDocuments?: { document_id: string; filename: string }[];
  selectedDocumentIds?: string[];
  onToggleDocumentReference?: (documentId: string) => void;
}

export default function ChatInput({
  onSend,
  onStop,
  isGenerating,
  disabled,
  onUploadDocument,
  isUploading,
  contextLimit,
  onWebSearch,
  isSearching,
  documentsAttached,
  availableDocuments,
  selectedDocumentIds,
  onToggleDocumentReference,
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const [isDragOver, setIsDragOver] = useState(false);
  const [documentsOnlyMode, setDocumentsOnlyMode] = useState(false);
  const [showDocsMenu, setShowDocsMenu] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed, { documentsOnly: documentsOnlyMode });
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (isGenerating || isSearching) return;
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

  const handleWebSearch = () => {
    const trimmed = input.trim();
    if (!trimmed || disabled || isSearching) return;
    onWebSearch?.(trimmed);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const isBusy = isGenerating || isSearching;

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
              className="inline-flex items-center gap-1.5 px-2.5 py-2 rounded-xl transition-smooth disabled:opacity-30 text-xs font-medium"
              style={{
                color: isUploading ? "var(--accent)" : "var(--text-muted)",
                background: "var(--bg-tertiary)",
                border: "1px solid var(--border-color)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.color = "var(--accent)";
                e.currentTarget.style.background = "var(--accent-muted)";
                e.currentTarget.style.border = "1px solid rgba(99, 102, 241, 0.35)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.color = isUploading ? "var(--accent)" : "var(--text-muted)";
                e.currentTarget.style.background = "var(--bg-tertiary)";
                e.currentTarget.style.border = "1px solid var(--border-color)";
              }}
              title="Upload file"
            >
              {isUploading ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Paperclip size={16} />
              )}
              <span className="hidden sm:inline">Upload</span>
            </button>
          )}

          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={
              isDragOver
                ? "Drop file to upload..."
                : documentsOnlyMode
                  ? "Message SoloLLM... (Docs Only mode ON)"
                  : "Message SoloLLM... (use Upload or Web)"
            }
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
            <button
              onClick={() => setDocumentsOnlyMode((prev) => !prev)}
              disabled={disabled || !documentsAttached}
              className="inline-flex items-center gap-1.5 px-2.5 py-2 rounded-xl transition-smooth disabled:opacity-30 text-xs font-medium"
              style={{
                background: documentsOnlyMode ? "var(--accent-muted)" : "var(--bg-tertiary)",
                color: documentsOnlyMode ? "var(--accent)" : "var(--text-muted)",
                border: documentsOnlyMode
                  ? "1px solid rgba(99, 102, 241, 0.35)"
                  : "1px solid var(--border-color)",
              }}
              title={documentsAttached ? "Answer using only attached thread documents" : "Attach a document to enable Docs Only mode"}
            >
              <FileText size={14} />
              <span className="hidden sm:inline">Docs</span>
            </button>

            <div className="relative">
              <button
                onClick={() => setShowDocsMenu((prev) => !prev)}
                disabled={disabled || !availableDocuments || availableDocuments.length === 0}
                className="inline-flex items-center gap-1.5 px-2.5 py-2 rounded-xl transition-smooth disabled:opacity-30 text-xs font-medium"
                style={{
                  background: showDocsMenu ? "var(--accent-muted)" : "var(--bg-tertiary)",
                  color: showDocsMenu ? "var(--accent)" : "var(--text-muted)",
                  border: showDocsMenu
                    ? "1px solid rgba(99, 102, 241, 0.35)"
                    : "1px solid var(--border-color)",
                }}
                title="Select referenced documents"
              >
                <FileText size={14} />
                <span className="hidden sm:inline">Refs</span>
                <span className="text-[10px]">{selectedDocumentIds?.length || 0}</span>
              </button>
              {showDocsMenu && availableDocuments && availableDocuments.length > 0 && (
                <div
                  className="absolute bottom-12 right-0 w-72 max-h-56 overflow-y-auto rounded-lg p-2 z-20"
                  style={{
                    background: "var(--bg-secondary)",
                    border: "1px solid var(--border-color)",
                    boxShadow: "0 8px 28px rgba(0, 0, 0, 0.35)",
                  }}
                >
                  <div className="text-[10px] px-1 pb-1" style={{ color: "var(--text-muted)" }}>
                    Referenced documents (thread)
                  </div>
                  {availableDocuments.map((doc) => (
                    <label
                      key={doc.document_id}
                      className="flex items-center gap-2 px-1 py-1 rounded text-xs cursor-pointer"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      <input
                        type="checkbox"
                        checked={Boolean(selectedDocumentIds?.includes(doc.document_id))}
                        onChange={() => onToggleDocumentReference?.(doc.document_id)}
                      />
                      <span className="truncate">{doc.filename}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>

            {/* Web Search button */}
            {onWebSearch && (
              <button
                onClick={handleWebSearch}
                disabled={!input.trim() || disabled || isBusy || documentsOnlyMode}
                className="inline-flex items-center gap-1.5 px-2.5 py-2 rounded-xl transition-smooth disabled:opacity-20 text-xs font-medium"
                style={{
                  background: isSearching
                    ? "rgba(59, 130, 246, 0.15)"
                    : input.trim()
                      ? "rgba(59, 130, 246, 0.1)"
                      : "var(--bg-tertiary)",
                  color: isSearching
                    ? "rgb(96, 165, 250)"
                    : input.trim()
                      ? "rgb(96, 165, 250)"
                      : "var(--text-muted)",
                  border: isSearching
                    ? "1px solid rgba(59, 130, 246, 0.3)"
                    : input.trim()
                      ? "1px solid rgba(59, 130, 246, 0.25)"
                      : "1px solid var(--border-color)",
                }}
                onMouseEnter={(e) => {
                  if (input.trim() && !isBusy) {
                    e.currentTarget.style.background = "rgba(59, 130, 246, 0.2)";
                    e.currentTarget.style.border = "1px solid rgba(59, 130, 246, 0.4)";
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = isSearching
                    ? "rgba(59, 130, 246, 0.15)"
                    : input.trim()
                      ? "rgba(59, 130, 246, 0.1)"
                      : "var(--bg-tertiary)";
                  e.currentTarget.style.border = isSearching
                    ? "1px solid rgba(59, 130, 246, 0.3)"
                    : input.trim()
                      ? "1px solid rgba(59, 130, 246, 0.25)"
                      : "1px solid var(--border-color)";
                }}
                title="Web search"
              >
                {isSearching ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <Globe size={14} />
                )}
                <span className="hidden sm:inline">Web</span>
              </button>
            )}

            {isBusy ? (
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
