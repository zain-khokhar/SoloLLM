"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  ArrowLeft,
  GitBranch,
  Search,
  Database,
  Network,
  Loader2,
  Circle,
  Globe,
  BarChart3,
  Clock,
  Trash2,
  Eye,
  ExternalLink,
  Zap,
  X,
  ChevronDown,
  ChevronRight,
  AlertCircle,
} from "lucide-react";
import {
  getGraphStats,
  searchGraphEntities,
  getEntityNeighbors,
  deleteEntity,
  getGraphVisualization,
  getGraphAnalysis,
  getEntityTimeline,
  clearGraph,
  scrapeUrl,
  scrapePreview,
} from "@/lib/api";
import type {
  GraphStats,
  GraphNode,
  GraphEdge,
  GraphAnalysis,
  EntityNeighbors,
  TimelineEntity,
  ScrapePreview as ScrapePreviewType,
} from "@/types";

interface KnowledgeGraphViewProps {
  onBack: () => void;
}

type Tab = "graph" | "analysis" | "timeline" | "scrape";

export default function KnowledgeGraphView({
  onBack,
}: KnowledgeGraphViewProps) {
  const [activeTab, setActiveTab] = useState<Tab>("graph");
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<GraphNode[]>([]);
  const [searching, setSearching] = useState(false);
  const [loading, setLoading] = useState(true);
  const [selectedEntity, setSelectedEntity] = useState<EntityNeighbors | null>(null);
  const [analysis, setAnalysis] = useState<GraphAnalysis | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [timeline, setTimeline] = useState<TimelineEntity[]>([]);
  const [timelineLoading, setTimelineLoading] = useState(false);

  // Scraping state
  const [scrapeUrlInput, setScrapeUrlInput] = useState("");
  const [scraping, setScraping] = useState(false);
  const [scrapeResult, setScrapeResult] = useState<Record<string, unknown> | null>(null);
  const [scrapeError, setScrapeError] = useState<string | null>(null);
  const [preview, setPreview] = useState<ScrapePreviewType | null>(null);
  const [previewing, setPreviewing] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [statsRes, graphRes] = await Promise.all([
        getGraphStats().catch(() => null),
        getGraphVisualization(150).catch(() => ({ nodes: [], edges: [] })),
      ]);
      setStats(statsRes);
      setNodes(graphRes.nodes || []);
      setEdges(graphRes.edges || []);
    } catch {
      // Backend might not be running
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Render force-directed graph on canvas
  useEffect(() => {
    if (!canvasRef.current || nodes.length === 0) return;
    renderGraph(canvasRef.current, nodes, edges);
  }, [nodes, edges, activeTab]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    try {
      const data = await searchGraphEntities(searchQuery, 20);
      setSearchResults(data.entities || []);
    } catch {
      // Ignore
    }
    setSearching(false);
  };

  const handleEntityClick = async (entityId: string) => {
    try {
      const data = await getEntityNeighbors(entityId);
      setSelectedEntity(data);
    } catch {
      // Ignore
    }
  };

  const handleDeleteEntity = async (entityId: string) => {
    try {
      await deleteEntity(entityId);
      setSelectedEntity(null);
      setSearchResults((prev) => prev.filter((e) => e.id !== entityId));
      loadData();
    } catch {
      // Ignore
    }
  };

  const loadAnalysis = async () => {
    setAnalysisLoading(true);
    try {
      const data = await getGraphAnalysis();
      setAnalysis(data);
    } catch {
      // Ignore
    }
    setAnalysisLoading(false);
  };

  const loadTimeline = async () => {
    setTimelineLoading(true);
    try {
      const data = await getEntityTimeline(50);
      setTimeline(data.entities || []);
    } catch {
      // Ignore
    }
    setTimelineLoading(false);
  };

  useEffect(() => {
    if (activeTab === "analysis" && !analysis) loadAnalysis();
    if (activeTab === "timeline" && timeline.length === 0) loadTimeline();
  }, [activeTab]);

  const handleScrape = async () => {
    if (!scrapeUrlInput.trim()) return;
    setScraping(true);
    setScrapeError(null);
    setScrapeResult(null);
    try {
      const result = await scrapeUrl(scrapeUrlInput);
      setScrapeResult(result as unknown as Record<string, unknown>);
      loadData(); // Refresh graph
    } catch (e: unknown) {
      setScrapeError(e instanceof Error ? e.message : "Scraping failed");
    }
    setScraping(false);
  };

  const handlePreview = async () => {
    if (!scrapeUrlInput.trim()) return;
    setPreviewing(true);
    setScrapeError(null);
    setPreview(null);
    try {
      const data = await scrapePreview(scrapeUrlInput);
      setPreview(data);
    } catch (e: unknown) {
      setScrapeError(e instanceof Error ? e.message : "Preview failed");
    }
    setPreviewing(false);
  };

  const handleClearGraph = async () => {
    try {
      await clearGraph();
      loadData();
      setSelectedEntity(null);
      setSearchResults([]);
      setAnalysis(null);
      setTimeline([]);
    } catch {
      // Ignore
    }
  };

  const typeColors: Record<string, string> = {
    person: "#818cf8",
    organization: "#34d399",
    concept: "#f59e0b",
    date: "#f87171",
    code_ref: "#22d3ee",
    quoted: "#c084fc",
    url: "#60a5fa",
    email: "#fb923c",
    technology: "#a78bfa",
  };

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: "graph", label: "Graph", icon: <Network size={14} /> },
    { id: "analysis", label: "Analysis", icon: <BarChart3 size={14} /> },
    { id: "timeline", label: "Timeline", icon: <Clock size={14} /> },
    { id: "scrape", label: "Web Scrape", icon: <Globe size={14} /> },
  ];

  return (
    <div
      className="flex-1 overflow-y-auto"
      style={{ background: "var(--bg-primary)" }}
    >
      <div className="max-w-5xl mx-auto p-6 animate-fadeIn">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
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
              <h1 className="text-xl font-bold gradient-text">
                Memory Inspector
              </h1>
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                Knowledge graph, entity analysis & web scraping
              </p>
            </div>
          </div>
          {stats && stats.entity_count > 0 && (
            <button
              onClick={handleClearGraph}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-smooth"
              style={{
                color: "var(--text-muted)",
                border: "1px solid var(--border-color)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "#f87171";
                e.currentTarget.style.color = "#f87171";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "var(--border-color)";
                e.currentTarget.style.color = "var(--text-muted)";
              }}
            >
              <Trash2 size={12} /> Clear
            </button>
          )}
        </div>

        {/* Stats */}
        {stats && (
          <div className="grid grid-cols-3 gap-3 mb-6">
            <StatCard icon={GitBranch} label="Entities" value={String(stats.entity_count)} />
            <StatCard icon={Network} label="Relationships" value={String(stats.relationship_count)} />
            <StatCard icon={Database} label="Types" value={String(Object.keys(stats.entity_types).length)} />
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-1 mb-6 p-1 rounded-xl" style={{ background: "var(--bg-secondary)" }}>
          {tabs.map(({ id, label, icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-medium transition-smooth flex-1 justify-center"
              style={{
                background: activeTab === id ? "var(--bg-tertiary)" : "transparent",
                color: activeTab === id ? "var(--text-primary)" : "var(--text-muted)",
                border: activeTab === id ? "1px solid var(--border-color)" : "1px solid transparent",
              }}
            >
              {icon} {label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === "graph" && (
          <GraphTab
            nodes={nodes}
            edges={edges}
            loading={loading}
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            searchResults={searchResults}
            searching={searching}
            handleSearch={handleSearch}
            handleEntityClick={handleEntityClick}
            selectedEntity={selectedEntity}
            setSelectedEntity={setSelectedEntity}
            handleDeleteEntity={handleDeleteEntity}
            typeColors={typeColors}
            stats={stats}
            canvasRef={canvasRef}
          />
        )}
        {activeTab === "analysis" && (
          <AnalysisTab
            analysis={analysis}
            loading={analysisLoading}
            onRefresh={loadAnalysis}
            typeColors={typeColors}
          />
        )}
        {activeTab === "timeline" && (
          <TimelineTab
            timeline={timeline}
            loading={timelineLoading}
            onRefresh={loadTimeline}
            typeColors={typeColors}
            handleEntityClick={handleEntityClick}
          />
        )}
        {activeTab === "scrape" && (
          <ScrapeTab
            urlInput={scrapeUrlInput}
            setUrlInput={setScrapeUrlInput}
            scraping={scraping}
            handleScrape={handleScrape}
            scrapeResult={scrapeResult}
            scrapeError={scrapeError}
            preview={preview}
            previewing={previewing}
            handlePreview={handlePreview}
          />
        )}
      </div>
    </div>
  );
}

// ── Force-directed graph renderer ──────────────────────────

function renderGraph(
  canvas: HTMLCanvasElement,
  nodes: GraphNode[],
  edges: GraphEdge[]
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;

  const typeColors: Record<string, string> = {
    person: "#818cf8",
    organization: "#34d399",
    concept: "#f59e0b",
    date: "#f87171",
    code_ref: "#22d3ee",
    quoted: "#c084fc",
    url: "#60a5fa",
    email: "#fb923c",
    technology: "#a78bfa",
  };

  // Assign positions using simple force layout
  const positions: Record<string, { x: number; y: number }> = {};
  const nodeMap: Record<string, GraphNode> = {};
  nodes.forEach((n, i) => {
    nodeMap[n.id] = n;
    // Arrange in a circle initially
    const angle = (2 * Math.PI * i) / nodes.length;
    const radius = Math.min(W, H) * 0.35;
    positions[n.id] = {
      x: W / 2 + radius * Math.cos(angle),
      y: H / 2 + radius * Math.sin(angle),
    };
  });

  // Simple force simulation (a few iterations)
  for (let iter = 0; iter < 80; iter++) {
    // Repulsion between nodes
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = positions[nodes[i].id];
        const b = positions[nodes[j].id];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
        const force = 800 / (dist * dist);
        a.x -= (dx / dist) * force;
        a.y -= (dy / dist) * force;
        b.x += (dx / dist) * force;
        b.y += (dy / dist) * force;
      }
    }

    // Attraction along edges
    for (const edge of edges) {
      const a = positions[edge.source_entity_id];
      const b = positions[edge.target_entity_id];
      if (!a || !b) continue;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      const force = (dist - 100) * 0.01;
      a.x += (dx / dist) * force;
      a.y += (dy / dist) * force;
      b.x -= (dx / dist) * force;
      b.y -= (dy / dist) * force;
    }

    // Center gravity
    for (const n of nodes) {
      const p = positions[n.id];
      p.x += (W / 2 - p.x) * 0.01;
      p.y += (H / 2 - p.y) * 0.01;
      // Keep in bounds
      p.x = Math.max(30, Math.min(W - 30, p.x));
      p.y = Math.max(30, Math.min(H - 30, p.y));
    }
  }

  // Draw
  ctx.clearRect(0, 0, W, H);

  // Edges
  ctx.lineWidth = 0.5;
  for (const edge of edges) {
    const a = positions[edge.source_entity_id];
    const b = positions[edge.target_entity_id];
    if (!a || !b) continue;
    ctx.strokeStyle = "rgba(100, 100, 140, 0.25)";
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  // Nodes
  for (const node of nodes) {
    const p = positions[node.id];
    if (!p) continue;
    const color = typeColors[node.entity_type] || "#6b7280";
    const radius = Math.max(4, Math.min(12, 3 + node.mention_count * 1.5));

    // Glow
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius + 3, 0, Math.PI * 2);
    ctx.fillStyle = color + "20";
    ctx.fill();

    // Node circle
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    // Label
    ctx.fillStyle = "#d1d5db";
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    const label = node.name.length > 18 ? node.name.slice(0, 16) + "…" : node.name;
    ctx.fillText(label, p.x, p.y + radius + 14);
  }
}

// ── Graph Tab ──────────────────────────────────────────────

function GraphTab({
  nodes,
  edges,
  loading,
  searchQuery,
  setSearchQuery,
  searchResults,
  searching,
  handleSearch,
  handleEntityClick,
  selectedEntity,
  setSelectedEntity,
  handleDeleteEntity,
  typeColors,
  stats,
  canvasRef,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
  loading: boolean;
  searchQuery: string;
  setSearchQuery: (q: string) => void;
  searchResults: GraphNode[];
  searching: boolean;
  handleSearch: () => void;
  handleEntityClick: (id: string) => void;
  selectedEntity: EntityNeighbors | null;
  setSelectedEntity: (e: EntityNeighbors | null) => void;
  handleDeleteEntity: (id: string) => void;
  typeColors: Record<string, string>;
  stats: GraphStats | null;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
}) {
  return (
    <div className="space-y-4 animate-fadeIn">
      {/* Entity Type Legend */}
      {stats && Object.keys(stats.entity_types).length > 0 && (
        <div
          className="p-4 rounded-xl"
          style={{
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
          }}
        >
          <p className="text-xs font-semibold mb-2" style={{ color: "var(--text-muted)" }}>
            Entity Types
          </p>
          <div className="flex flex-wrap gap-3">
            {Object.entries(stats.entity_types).map(([type, count]) => (
              <span
                key={type}
                className="flex items-center gap-1.5 text-xs"
                style={{ color: "var(--text-secondary)" }}
              >
                <Circle size={8} fill={typeColors[type] || "#6b7280"} color={typeColors[type] || "#6b7280"} />
                {type} <span style={{ color: "var(--text-muted)" }}>({count})</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Search */}
      <div className="flex gap-2">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          placeholder="Search entities..."
          className="flex-1 rounded-xl px-4 py-2.5 text-sm outline-none transition-smooth"
          style={{
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
            color: "var(--text-primary)",
          }}
          onFocus={(e) => { e.currentTarget.style.borderColor = "var(--accent)"; }}
          onBlur={(e) => { e.currentTarget.style.borderColor = "var(--border-color)"; }}
        />
        <button
          onClick={handleSearch}
          disabled={searching || !searchQuery.trim()}
          className="px-4 py-2.5 rounded-xl text-sm font-medium transition-smooth disabled:opacity-30"
          style={{
            background: "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
            color: "white",
          }}
        >
          {searching ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
        </button>
      </div>

      {/* Search Results */}
      {searchResults.length > 0 && (
        <div className="space-y-1.5 animate-fadeIn">
          {searchResults.map((entity) => (
            <div
              key={entity.id}
              onClick={() => handleEntityClick(entity.id)}
              className="flex items-center gap-3 px-4 py-2.5 rounded-xl cursor-pointer transition-smooth"
              style={{
                background: "var(--bg-secondary)",
                border: "1px solid var(--border-color)",
              }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--accent)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border-color)"; }}
            >
              <Circle size={8} fill={typeColors[entity.entity_type] || "#6b7280"} color={typeColors[entity.entity_type] || "#6b7280"} />
              <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                {entity.name}
              </span>
              <span
                className="text-[10px] px-2 py-0.5 rounded-lg uppercase"
                style={{ background: "var(--bg-tertiary)", color: "var(--text-muted)" }}
              >
                {entity.entity_type}
              </span>
              <span className="text-xs ml-auto" style={{ color: "var(--text-muted)" }}>
                {entity.mention_count}×
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Entity Detail Panel */}
      {selectedEntity && selectedEntity.entity && (
        <EntityDetailPanel
          data={selectedEntity}
          onClose={() => setSelectedEntity(null)}
          typeColors={typeColors}
          onDelete={handleDeleteEntity}
        />
      )}

      {/* Graph Visualization */}
      <div
        className="rounded-xl overflow-hidden"
        style={{
          background: "var(--bg-secondary)",
          border: "1px solid var(--border-color)",
          minHeight: "400px",
        }}
      >
        {loading ? (
          <div className="flex items-center justify-center h-96">
            <Loader2 size={24} className="animate-spin" style={{ color: "var(--accent)" }} />
          </div>
        ) : nodes.length === 0 ? (
          <div className="text-center py-16">
            <Network size={40} className="mx-auto mb-3" style={{ color: "var(--text-muted)", opacity: 0.4 }} />
            <p className="text-sm" style={{ color: "var(--text-muted)" }}>No entities yet</p>
            <p className="text-xs mt-1" style={{ color: "var(--text-muted)", opacity: 0.7 }}>
              Upload documents or scrape web pages to build the knowledge graph
            </p>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            style={{ width: "100%", height: "400px", display: "block" }}
          />
        )}
      </div>

      {/* Entity chips below graph */}
      {nodes.length > 0 && (
        <div
          className="rounded-xl p-4"
          style={{
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
          }}
        >
          <p className="text-xs font-semibold mb-3" style={{ color: "var(--text-muted)" }}>
            Top Entities ({nodes.length})
          </p>
          <div className="flex flex-wrap gap-2">
            {nodes.slice(0, 50).map((node) => (
              <span
                key={node.id}
                onClick={() => handleEntityClick(node.id)}
                className="px-3 py-1.5 rounded-lg text-xs font-medium transition-smooth hover-lift cursor-pointer"
                style={{
                  background: `${typeColors[node.entity_type] || "#6b7280"}20`,
                  color: typeColors[node.entity_type] || "#9ca3af",
                  border: `1px solid ${typeColors[node.entity_type] || "#6b7280"}30`,
                }}
              >
                {node.name}
                {node.mention_count > 1 && (
                  <span className="ml-1 opacity-60">×{node.mention_count}</span>
                )}
              </span>
            ))}
          </div>
          {edges.length > 0 && (
            <p className="text-xs mt-4" style={{ color: "var(--text-muted)" }}>
              {edges.length} relationships detected
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// ── Entity Detail Panel ────────────────────────────────────

function EntityDetailPanel({
  data,
  onClose,
  typeColors,
  onDelete,
}: {
  data: EntityNeighbors;
  onClose: () => void;
  typeColors: Record<string, string>;
  onDelete: (id: string) => void;
}) {
  const entity = data.entity;
  if (!entity) return null;

  return (
    <div
      className="rounded-xl p-4 animate-fadeIn"
      style={{
        background: "var(--bg-secondary)",
        border: "1px solid var(--accent)",
      }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Circle
            size={10}
            fill={typeColors[entity.entity_type] || "#6b7280"}
            color={typeColors[entity.entity_type] || "#6b7280"}
          />
          <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            {entity.name}
          </span>
          <span
            className="text-[10px] px-2 py-0.5 rounded-lg uppercase"
            style={{ background: "var(--bg-tertiary)", color: "var(--text-muted)" }}
          >
            {entity.entity_type}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => onDelete(entity.id)}
            className="p-1.5 rounded-lg transition-smooth"
            style={{ color: "var(--text-muted)" }}
            onMouseEnter={(e) => { e.currentTarget.style.color = "#f87171"; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text-muted)"; }}
            title="Delete entity"
          >
            <Trash2 size={13} />
          </button>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg transition-smooth"
            style={{ color: "var(--text-muted)" }}
            onMouseEnter={(e) => { e.currentTarget.style.color = "var(--text-primary)"; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text-muted)"; }}
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {data.neighbors.length > 0 ? (
        <div className="space-y-1.5">
          <p className="text-xs font-medium" style={{ color: "var(--text-muted)" }}>
            Connected to ({data.neighbors.length}):
          </p>
          {data.neighbors.map((n, i) => (
            <div
              key={i}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs"
              style={{ background: "var(--bg-tertiary)" }}
            >
              <Circle
                size={6}
                fill={typeColors[n.target_type] || "#6b7280"}
                color={typeColors[n.target_type] || "#6b7280"}
              />
              <span style={{ color: "var(--text-primary)" }}>{n.target_name}</span>
              <span
                className="px-1.5 py-0.5 rounded"
                style={{
                  background: "var(--bg-secondary)",
                  color: "var(--text-muted)",
                  fontSize: "9px",
                }}
              >
                {n.relation_type}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
          No connections found
        </p>
      )}
    </div>
  );
}

// ── Analysis Tab ───────────────────────────────────────────

function AnalysisTab({
  analysis,
  loading,
  onRefresh,
  typeColors,
}: {
  analysis: GraphAnalysis | null;
  loading: boolean;
  onRefresh: () => void;
  typeColors: Record<string, string>;
}) {
  const [expandedCommunity, setExpandedCommunity] = useState<number | null>(null);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 size={24} className="animate-spin" style={{ color: "var(--accent)" }} />
      </div>
    );
  }

  if (!analysis || analysis.node_count === 0) {
    return (
      <div className="text-center py-16 animate-fadeIn">
        <BarChart3 size={40} className="mx-auto mb-3" style={{ color: "var(--text-muted)", opacity: 0.4 }} />
        <p className="text-sm" style={{ color: "var(--text-muted)" }}>No graph data to analyze</p>
        <button
          onClick={onRefresh}
          className="mt-3 px-4 py-2 rounded-xl text-xs font-medium transition-smooth"
          style={{
            background: "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
            color: "white",
          }}
        >
          Run Analysis
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4 animate-fadeIn">
      {/* Overview */}
      <div className="grid grid-cols-3 gap-3">
        <StatCard icon={Network} label="Nodes" value={String(analysis.node_count ?? 0)} />
        <StatCard icon={GitBranch} label="Edges" value={String(analysis.edge_count ?? 0)} />
        <StatCard icon={Zap} label="Density" value={(analysis.density ?? 0).toFixed(4)} />
      </div>

      {/* Hub Entities (PageRank) */}
      {(analysis.hub_entities ?? []).length > 0 && (
        <div
          className="rounded-xl p-4"
          style={{
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
          }}
        >
          <p className="text-xs font-semibold mb-3" style={{ color: "var(--text-muted)" }}>
            Hub Entities (by PageRank)
          </p>
          <div className="space-y-2">
            {(analysis.hub_entities ?? []).map((hub) => (
              <div key={hub.id} className="flex items-center gap-3">
                <Circle
                  size={8}
                  fill={typeColors[hub.entity_type] || "#6b7280"}
                  color={typeColors[hub.entity_type] || "#6b7280"}
                />
                <span className="text-sm font-medium flex-1" style={{ color: "var(--text-primary)" }}>
                  {hub.name}
                </span>
                <span className="text-[10px] font-mono" style={{ color: "var(--accent)" }}>
                  PR: {(hub.pagerank ?? 0).toFixed(4)}
                </span>
                <span className="text-[10px] font-mono" style={{ color: "var(--text-muted)" }}>
                  BC: {(hub.centrality ?? 0).toFixed(4)}
                </span>
                <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                  deg: {hub.degree}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Communities */}
      {(analysis.communities ?? []).length > 0 && (
        <div
          className="rounded-xl p-4"
          style={{
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
          }}
        >
          <p className="text-xs font-semibold mb-3" style={{ color: "var(--text-muted)" }}>
            Communities ({(analysis.communities ?? []).length})
          </p>
          <div className="space-y-1.5">
            {(analysis.communities ?? []).map((community) => (
              <div key={community.id}>
                <button
                  onClick={() =>
                    setExpandedCommunity(
                      expandedCommunity === community.id ? null : community.id
                    )
                  }
                  className="flex items-center gap-2 w-full px-3 py-2 rounded-lg text-xs transition-smooth"
                  style={{ background: "var(--bg-tertiary)", color: "var(--text-primary)" }}
                >
                  {expandedCommunity === community.id ? (
                    <ChevronDown size={12} />
                  ) : (
                    <ChevronRight size={12} />
                  )}
                  <span className="font-medium">Community {community.id + 1}</span>
                  <span style={{ color: "var(--text-muted)" }}>{community.size} members</span>
                </button>
                {expandedCommunity === community.id && (
                  <div className="flex flex-wrap gap-1.5 mt-1.5 ml-6 animate-fadeIn">
                    {community.members.map((m) => (
                      <span
                        key={m.id}
                        className="px-2 py-1 rounded text-[10px]"
                        style={{
                          background: "var(--bg-secondary)",
                          color: "var(--text-secondary)",
                          border: "1px solid var(--border-color)",
                        }}
                      >
                        {m.name}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <button
        onClick={onRefresh}
        className="w-full py-2.5 rounded-xl text-xs font-medium transition-smooth"
        style={{
          background: "var(--bg-secondary)",
          border: "1px solid var(--border-color)",
          color: "var(--text-secondary)",
        }}
        onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--accent)"; }}
        onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border-color)"; }}
      >
        Refresh Analysis
      </button>
    </div>
  );
}

// ── Timeline Tab ───────────────────────────────────────────

function TimelineTab({
  timeline,
  loading,
  onRefresh,
  typeColors,
  handleEntityClick,
}: {
  timeline: TimelineEntity[];
  loading: boolean;
  onRefresh: () => void;
  typeColors: Record<string, string>;
  handleEntityClick: (id: string) => void;
}) {
  const formatTime = (dateStr: string) => {
    try {
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
    } catch {
      return dateStr;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 size={24} className="animate-spin" style={{ color: "var(--accent)" }} />
      </div>
    );
  }

  return (
    <div className="space-y-3 animate-fadeIn">
      {timeline.length === 0 ? (
        <div className="text-center py-16">
          <Clock size={40} className="mx-auto mb-3" style={{ color: "var(--text-muted)", opacity: 0.4 }} />
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>No entities recorded yet</p>
        </div>
      ) : (
        <div className="space-y-1.5">
          {timeline.map((entity) => (
            <div
              key={entity.id}
              onClick={() => handleEntityClick(entity.id)}
              className="flex items-center gap-3 px-4 py-3 rounded-xl cursor-pointer transition-smooth"
              style={{
                background: "var(--bg-secondary)",
                border: "1px solid var(--border-color)",
              }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--accent)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border-color)"; }}
            >
              <Circle
                size={8}
                fill={typeColors[entity.entity_type] || "#6b7280"}
                color={typeColors[entity.entity_type] || "#6b7280"}
              />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate" style={{ color: "var(--text-primary)" }}>
                  {entity.name}
                </p>
                <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>
                  {entity.entity_type} · {entity.mention_count}× mentions
                </p>
              </div>
              <span className="text-[10px] whitespace-nowrap" style={{ color: "var(--text-muted)" }}>
                {formatTime(entity.updated_at)}
              </span>
            </div>
          ))}
        </div>
      )}

      <button
        onClick={onRefresh}
        className="w-full py-2.5 rounded-xl text-xs font-medium transition-smooth"
        style={{
          background: "var(--bg-secondary)",
          border: "1px solid var(--border-color)",
          color: "var(--text-secondary)",
        }}
        onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--accent)"; }}
        onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border-color)"; }}
      >
        Refresh Timeline
      </button>
    </div>
  );
}

// ── Scrape Tab ─────────────────────────────────────────────

function ScrapeTab({
  urlInput,
  setUrlInput,
  scraping,
  handleScrape,
  scrapeResult,
  scrapeError,
  preview,
  previewing,
  handlePreview,
}: {
  urlInput: string;
  setUrlInput: (v: string) => void;
  scraping: boolean;
  handleScrape: () => void;
  scrapeResult: Record<string, unknown> | null;
  scrapeError: string | null;
  preview: ScrapePreviewType | null;
  previewing: boolean;
  handlePreview: () => void;
}) {
  return (
    <div className="space-y-4 animate-fadeIn">
      {/* URL Input */}
      <div
        className="rounded-xl p-4"
        style={{
          background: "var(--bg-secondary)",
          border: "1px solid var(--border-color)",
        }}
      >
        <p className="text-xs font-semibold mb-3" style={{ color: "var(--text-muted)" }}>
          Web Page Ingestion
        </p>
        <p className="text-xs mb-3" style={{ color: "var(--text-muted)", opacity: 0.7 }}>
          Scrape a web page and ingest its content into your knowledge base and graph.
        </p>
        <div className="flex gap-2">
          <input
            type="url"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleScrape()}
            placeholder="https://example.com/article"
            className="flex-1 rounded-xl px-4 py-2.5 text-sm outline-none transition-smooth"
            style={{
              background: "var(--bg-tertiary)",
              border: "1px solid var(--border-color)",
              color: "var(--text-primary)",
            }}
            onFocus={(e) => { e.currentTarget.style.borderColor = "var(--accent)"; }}
            onBlur={(e) => { e.currentTarget.style.borderColor = "var(--border-color)"; }}
          />
          <button
            onClick={handlePreview}
            disabled={previewing || !urlInput.trim()}
            className="px-3 py-2.5 rounded-xl text-xs font-medium transition-smooth disabled:opacity-30"
            style={{
              background: "var(--bg-tertiary)",
              border: "1px solid var(--border-color)",
              color: "var(--text-secondary)",
            }}
            title="Preview content"
          >
            {previewing ? <Loader2 size={14} className="animate-spin" /> : <Eye size={14} />}
          </button>
          <button
            onClick={handleScrape}
            disabled={scraping || !urlInput.trim()}
            className="px-4 py-2.5 rounded-xl text-sm font-medium transition-smooth disabled:opacity-30"
            style={{
              background: "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
              color: "white",
            }}
          >
            {scraping ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <span className="flex items-center gap-1.5">
                <Globe size={14} /> Scrape
              </span>
            )}
          </button>
        </div>
      </div>

      {/* Error */}
      {scrapeError ? (
        <div
          className="flex items-center gap-2 px-4 py-3 rounded-xl text-xs animate-fadeIn"
          style={{
            background: "rgba(248, 113, 113, 0.08)",
            border: "1px solid rgba(248, 113, 113, 0.15)",
            color: "#f87171",
          }}
        >
          <AlertCircle size={14} />
          {scrapeError}
        </div>
      ) : null}

      {/* Preview */}
      {preview ? (
        <div
          className="rounded-xl p-4 space-y-3 animate-fadeIn"
          style={{
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
          }}
        >
          <div className="flex items-center justify-between">
            <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              {preview.title}
            </p>
            <a
              href={preview.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-xs"
              style={{ color: "var(--accent)" }}
            >
              <ExternalLink size={11} /> Open
            </a>
          </div>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div style={{ color: "var(--text-muted)" }}>
              Content: <span style={{ color: "var(--text-secondary)" }}>{preview.content_length.toLocaleString()} chars</span>
            </div>
            <div style={{ color: "var(--text-muted)" }}>
              Links: <span style={{ color: "var(--text-secondary)" }}>{preview.link_count}</span>
            </div>
          </div>
          <div
            className="rounded-lg p-3 text-xs leading-relaxed max-h-48 overflow-y-auto"
            style={{
              background: "var(--bg-tertiary)",
              color: "var(--text-secondary)",
              whiteSpace: "pre-wrap",
            }}
          >
            {preview.content_preview}
          </div>
        </div>
      ) : null}

      {/* Success Result */}
      {scrapeResult && Boolean((scrapeResult as Record<string, unknown>).success) ? (
        <div
          className="rounded-xl p-4 animate-fadeIn"
          style={{
            background: "rgba(52, 211, 153, 0.08)",
            border: "1px solid rgba(52, 211, 153, 0.2)",
          }}
        >
          <p className="text-sm font-semibold mb-2" style={{ color: "#34d399" }}>
            Successfully ingested!
          </p>
          <div className="grid grid-cols-2 gap-2 text-xs" style={{ color: "var(--text-secondary)" }}>
            <div>Title: <span className="font-medium">{String(scrapeResult.title || "")}</span></div>
            <div>Chunks: <span className="font-medium">{String(scrapeResult.chunk_count || 0)}</span></div>
            <div>Entities: <span className="font-medium">{String(scrapeResult.entities_extracted || 0)}</span></div>
            <div>Relationships: <span className="font-medium">{String(scrapeResult.relationships_extracted || 0)}</span></div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

// ── Shared Components ──────────────────────────────────────

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
      className="rounded-xl p-3.5 flex items-center gap-3"
      style={{
        background: "var(--bg-secondary)",
        border: "1px solid var(--border-color)",
      }}
    >
      <div
        className="w-8 h-8 rounded-lg flex items-center justify-center"
        style={{ background: "var(--accent-muted)" }}
      >
        <Icon size={15} style={{ color: "var(--accent)" }} />
      </div>
      <div>
        <p className="text-[11px] font-medium" style={{ color: "var(--text-muted)" }}>{label}</p>
        <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>{value}</p>
      </div>
    </div>
  );
}
