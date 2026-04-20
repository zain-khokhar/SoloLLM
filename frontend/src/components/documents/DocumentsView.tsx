"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  ArrowLeft,
  Upload,
  FileText,
  Trash2,
  Search,
  FolderOpen,
  Database,
  Loader2,
  CheckCircle2,
  AlertCircle,
  File,
  X,
  BookOpen,
  Hash,
  Layers,
} from "lucide-react";
import { DocumentInfo, Citation, RAGStats } from "@/types";
import {
  uploadDocument,
  listDocuments,
  deleteDocument,
  queryDocuments,
  getRAGStats,
} from "@/lib/api";

interface DocumentsViewProps {
  onBack: () => void;
}

export default function DocumentsView({ onBack }: DocumentsViewProps) {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [stats, setStats] = useState<RAGStats | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Citation[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadData = useCallback(async () => {
    try {
      const [docs, ragStats] = await Promise.all([
        listDocuments().catch(() => []),
        getRAGStats().catch(() => null),
      ]);
      setDocuments(docs);
      setStats(ragStats);
    } catch {
      // Backend might not be running
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    setError(null);
    setSuccess(null);

    for (const file of Array.from(files)) {
      setUploadProgress(`Processing ${file.name}...`);
      try {
        const result = await uploadDocument(file);
        if (result.success) {
          setSuccess(
            `Uploaded "${result.title || file.name}" — ${result.chunk_count} chunks created`
          );
        } else {
          setError(result.errors?.[0] || "Upload failed");
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Upload failed";
        setError(msg);
      }
    }

    setUploading(false);
    setUploadProgress("");
    await loadData();

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleDelete = async (docId: string, filename: string) => {
    try {
      await deleteDocument(docId);
      setSuccess(`Deleted "${filename}"`);
      await loadData();
    } catch {
      setError("Failed to delete document");
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    setError(null);

    try {
      const result = await queryDocuments(searchQuery);
      setSearchResults(result.citations);
    } catch {
      setError("Search failed. Is the backend running?");
    }
    setSearching(false);
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const fileTypeIcon = (type: string) => {
    const icons: Record<string, string> = {
      pdf: "📄",
      docx: "📝",
      markdown: "📋",
      text: "📃",
      html: "🌐",
      csv: "📊",
      code: "💻",
    };
    return icons[type] || "📎";
  };

  return (
    <div
      className="flex-1 overflow-y-auto"
      style={{ background: "var(--bg-primary)" }}
    >
      <div className="max-w-3xl mx-auto p-6 animate-fadeIn">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
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
            <h1 className="text-xl font-bold gradient-text">Documents</h1>
            <p className="text-xs" style={{ color: "var(--text-muted)" }}>
              Upload documents for RAG-powered conversations
            </p>
          </div>
        </div>

        {/* Alerts */}
        {error && (
          <div
            className="mb-4 px-4 py-3 rounded-xl flex items-center gap-2 text-sm animate-slideDown"
            style={{
              background: "rgba(248, 113, 113, 0.08)",
              border: "1px solid rgba(248, 113, 113, 0.15)",
              color: "var(--error)",
            }}
          >
            <AlertCircle size={16} />
            {error}
            <button onClick={() => setError(null)} className="ml-auto">
              <X size={14} />
            </button>
          </div>
        )}
        {success && (
          <div
            className="mb-4 px-4 py-3 rounded-xl flex items-center gap-2 text-sm animate-slideDown"
            style={{
              background: "rgba(52, 211, 153, 0.08)",
              border: "1px solid rgba(52, 211, 153, 0.15)",
              color: "var(--success)",
            }}
          >
            <CheckCircle2 size={16} />
            {success}
            <button onClick={() => setSuccess(null)} className="ml-auto">
              <X size={14} />
            </button>
          </div>
        )}

        {/* Stats Bar */}
        {stats && (
          <div
            className="mb-6 grid grid-cols-3 gap-3"
          >
            <StatCard
              icon={FileText}
              label="Documents"
              value={String(stats.document_count)}
            />
            <StatCard
              icon={Layers}
              label="Chunks"
              value={String(stats.chunk_count)}
            />
            <StatCard
              icon={Database}
              label="Embeddings"
              value={stats.embedding_info.fallback ? "Fallback" : stats.embedding_info.model.split("/").pop() || "Active"}
            />
          </div>
        )}

        {/* Upload Section */}
        <section className="mb-6">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.doc,.txt,.md,.markdown,.html,.htm,.csv,.tsv,.py,.js,.ts,.jsx,.tsx,.java,.c,.cpp,.h,.rs,.go,.rb,.json,.yaml,.yml,.xml,.sql"
            onChange={handleUpload}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="w-full rounded-xl p-6 flex flex-col items-center gap-3 transition-smooth hover-lift disabled:opacity-50"
            style={{
              border: "2px dashed var(--border-color)",
              background: "var(--bg-secondary)",
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
            {uploading ? (
              <>
                <Loader2
                  size={28}
                  className="animate-spin"
                  style={{ color: "var(--accent)" }}
                />
                <span className="text-sm" style={{ color: "var(--accent)" }}>
                  {uploadProgress || "Processing..."}
                </span>
              </>
            ) : (
              <>
                <Upload
                  size={28}
                  style={{ color: "var(--text-muted)" }}
                />
                <div className="text-center">
                  <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                    Drop files here or click to upload
                  </p>
                  <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                    PDF, DOCX, TXT, Markdown, HTML, CSV, and source code
                  </p>
                </div>
              </>
            )}
          </button>
        </section>

        {/* Search Section */}
        <section className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <Search size={16} style={{ color: "var(--accent)" }} />
            <h2
              className="text-sm font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              Search Documents
            </h2>
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSearch();
              }}
              placeholder="Ask a question about your documents..."
              className="flex-1 rounded-xl px-4 py-2.5 text-sm outline-none transition-smooth"
              style={{
                background: "var(--bg-secondary)",
                border: "1px solid var(--border-color)",
                color: "var(--text-primary)",
                caretColor: "var(--accent)",
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = "var(--accent)";
                e.currentTarget.style.boxShadow =
                  "0 0 12px rgba(99, 102, 241, 0.1)";
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = "var(--border-color)";
                e.currentTarget.style.boxShadow = "none";
              }}
            />
            <button
              onClick={handleSearch}
              disabled={searching || !searchQuery.trim()}
              className="px-4 py-2.5 rounded-xl text-sm font-medium transition-smooth disabled:opacity-30"
              style={{
                background:
                  "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
                color: "white",
              }}
            >
              {searching ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                "Search"
              )}
            </button>
          </div>

          {/* Search Results */}
          {searchResults && (
            <div className="mt-3 space-y-2 animate-fadeIn">
              {searchResults.length === 0 ? (
                <p
                  className="text-sm py-4 text-center"
                  style={{ color: "var(--text-muted)" }}
                >
                  No relevant results found
                </p>
              ) : (
                searchResults.map((cite) => (
                  <div
                    key={cite.chunk_id}
                    className="rounded-xl p-3.5 transition-smooth"
                    style={{
                      background: "var(--bg-secondary)",
                      border: "1px solid var(--border-color)",
                    }}
                  >
                    <div className="flex items-start gap-2 mb-1.5">
                      <span
                        className="text-xs font-bold px-1.5 py-0.5 rounded-md shrink-0"
                        style={{
                          background: "var(--accent-muted)",
                          color: "var(--accent)",
                        }}
                      >
                        [{cite.index}]
                      </span>
                      <div className="min-w-0 flex-1">
                        <p
                          className="text-sm font-medium truncate"
                          style={{ color: "var(--text-primary)" }}
                        >
                          {cite.document_title}
                          {cite.section_title && (
                            <span style={{ color: "var(--text-muted)" }}>
                              {" "}
                              › {cite.section_title}
                            </span>
                          )}
                        </p>
                      </div>
                      {cite.page_number && (
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded shrink-0"
                          style={{
                            background: "var(--bg-tertiary)",
                            color: "var(--text-muted)",
                          }}
                        >
                          p. {cite.page_number}
                        </span>
                      )}
                    </div>
                    <p
                      className="text-xs leading-relaxed line-clamp-3"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {cite.excerpt}
                    </p>
                    <div className="mt-1.5 flex items-center gap-2">
                      <span
                        className="text-[10px]"
                        style={{ color: "var(--text-muted)" }}
                      >
                        Relevance: {Math.round(cite.relevance_score * 100)}%
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </section>

        {/* Documents List */}
        <section>
          <div className="flex items-center gap-2 mb-3">
            <FolderOpen size={16} style={{ color: "var(--accent)" }} />
            <h2
              className="text-sm font-semibold"
              style={{ color: "var(--text-primary)" }}
            >
              Uploaded Documents
            </h2>
            <span
              className="text-xs px-2 py-0.5 rounded-full"
              style={{
                background: "var(--accent-muted)",
                color: "var(--accent)",
              }}
            >
              {documents.length}
            </span>
          </div>

          {documents.length === 0 ? (
            <div
              className="text-center py-12 rounded-xl"
              style={{
                background: "var(--bg-secondary)",
                border: "1px solid var(--border-color)",
              }}
            >
              <BookOpen
                size={36}
                className="mx-auto mb-3"
                style={{ color: "var(--text-muted)", opacity: 0.5 }}
              />
              <p className="text-sm" style={{ color: "var(--text-muted)" }}>
                No documents uploaded yet
              </p>
              <p
                className="text-xs mt-1"
                style={{ color: "var(--text-muted)", opacity: 0.7 }}
              >
                Upload documents to enable RAG-powered conversations
              </p>
            </div>
          ) : (
            <div className="space-y-1.5">
              {documents.map((doc, index) => (
                <div
                  key={doc.document_id}
                  className="flex items-center gap-3 px-4 py-3 rounded-xl group transition-smooth"
                  style={{
                    background: "var(--bg-secondary)",
                    border: "1px solid var(--border-color)",
                    animation: `fadeIn 0.3s ease-out ${index * 0.05}s both`,
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor =
                      "rgba(99, 102, 241, 0.2)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "var(--border-color)";
                  }}
                >
                  <span className="text-lg">
                    {fileTypeIcon(doc.file_type)}
                  </span>
                  <div className="flex-1 min-w-0">
                    <p
                      className="text-sm font-medium truncate"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {doc.title || doc.filename}
                    </p>
                    <div
                      className="flex items-center gap-3 mt-0.5 text-[11px]"
                      style={{ color: "var(--text-muted)" }}
                    >
                      <span className="flex items-center gap-1">
                        <Hash size={10} />
                        {doc.chunk_count} chunks
                      </span>
                      {doc.page_count > 0 && (
                        <span>{doc.page_count} pages</span>
                      )}
                      <span>{formatDate(doc.created_at)}</span>
                    </div>
                  </div>
                  <span
                    className="text-[10px] px-2 py-1 rounded-lg font-medium uppercase shrink-0"
                    style={{
                      background: "var(--bg-tertiary)",
                      color: "var(--text-muted)",
                    }}
                  >
                    {doc.file_type}
                  </span>
                  <button
                    onClick={() => handleDelete(doc.document_id, doc.filename)}
                    className="opacity-0 group-hover:opacity-100 p-2 rounded-lg transition-smooth shrink-0"
                    style={{ color: "var(--text-muted)" }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background =
                        "rgba(248, 113, 113, 0.1)";
                      e.currentTarget.style.color = "var(--error)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = "transparent";
                      e.currentTarget.style.color = "var(--text-muted)";
                    }}
                    title="Delete document"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────

function StatCard({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ComponentType<{ size: number; style?: React.CSSProperties }>;
  label: string;
  value: string;
}) {
  return (
    <div
      className="rounded-xl p-3.5 flex items-center gap-3 transition-smooth"
      style={{
        background: "var(--bg-secondary)",
        border: "1px solid var(--border-color)",
      }}
    >
      <div
        className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
        style={{ background: "var(--accent-muted)" }}
      >
        <Icon size={15} style={{ color: "var(--accent)" }} />
      </div>
      <div>
        <p
          className="text-[11px] font-medium"
          style={{ color: "var(--text-muted)" }}
        >
          {label}
        </p>
        <p
          className="text-sm font-semibold"
          style={{ color: "var(--text-primary)" }}
        >
          {value}
        </p>
      </div>
    </div>
  );
}
