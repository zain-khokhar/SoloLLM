"use client";

import { useState, useEffect, useCallback } from "react";
import {
  ArrowLeft,
  Activity,
  Clock,
  Zap,
  BarChart3,
  Cpu,
  HardDrive,
  RefreshCw,
  Loader2,
} from "lucide-react";
import { getDashboardSummary } from "@/lib/api";
import type { DashboardSummary } from "@/types";

interface DashboardViewProps {
  onBack: () => void;
}

export default function DashboardView({ onBack }: DashboardViewProps) {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadData = useCallback(async () => {
    try {
      setSummary(await getDashboardSummary());
    } catch {
      // Backend might not be running
    }
    setLoading(false);
    setRefreshing(false);
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 10000); // Auto-refresh every 10s
    return () => clearInterval(interval);
  }, [loadData]);

  const handleRefresh = () => {
    setRefreshing(true);
    loadData();
  };

  const formatUptime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  };

  const formatNumber = (n: number) =>
    n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);

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
          <div className="flex-1">
            <h1 className="text-xl font-bold gradient-text">
              Performance Dashboard
            </h1>
            <p className="text-xs" style={{ color: "var(--text-muted)" }}>
              Real-time metrics and system health
            </p>
          </div>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="p-2 rounded-xl transition-smooth"
            style={{
              background: "var(--bg-secondary)",
              border: "1px solid var(--border-color)",
              color: "var(--text-secondary)",
            }}
          >
            {refreshing ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <RefreshCw size={16} />
            )}
          </button>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2
              size={24}
              className="animate-spin"
              style={{ color: "var(--accent)" }}
            />
          </div>
        ) : !summary ? (
          <div className="text-center py-20">
            <Activity
              size={40}
              className="mx-auto mb-3"
              style={{ color: "var(--text-muted)", opacity: 0.4 }}
            />
            <p className="text-sm" style={{ color: "var(--text-muted)" }}>
              Dashboard unavailable — start the backend
            </p>
          </div>
        ) : (
          <>
            {/* Main Stats Grid */}
            <div className="grid grid-cols-4 gap-3 mb-6">
              <MetricCard
                icon={Clock}
                label="Uptime"
                value={formatUptime(summary.uptime_seconds)}
              />
              <MetricCard
                icon={Zap}
                label="Requests"
                value={formatNumber(summary.total_requests)}
              />
              <MetricCard
                icon={BarChart3}
                label="Tokens"
                value={formatNumber(summary.total_tokens)}
              />
              <MetricCard
                icon={Activity}
                label="Avg Latency"
                value={`${summary.avg_latency_ms}ms`}
              />
            </div>

            {/* System Health */}
            {summary.system && (
              <section className="mb-6">
                <h2
                  className="text-sm font-semibold mb-3 flex items-center gap-2"
                  style={{ color: "var(--text-primary)" }}
                >
                  <Cpu size={14} style={{ color: "var(--accent)" }} />
                  System Health
                </h2>
                <div className="grid grid-cols-2 gap-3">
                  <BarCard
                    label="CPU Usage"
                    value={summary.system.cpu_percent}
                    maxLabel="100%"
                  />
                  <BarCard
                    label="Memory"
                    value={summary.system.memory_percent}
                    maxLabel={`${summary.system.memory_used_mb}/${summary.system.memory_total_mb} MB`}
                  />
                </div>
              </section>
            )}

            {/* Latency Stats */}
            <section className="mb-6">
              <h2
                className="text-sm font-semibold mb-3"
                style={{ color: "var(--text-primary)" }}
              >
                Latency Breakdown
              </h2>
              <div
                className="grid grid-cols-4 gap-3 rounded-xl p-4"
                style={{
                  background: "var(--bg-secondary)",
                  border: "1px solid var(--border-color)",
                }}
              >
                <MiniStat label="Min" value={`${summary.min_latency_ms || 0}ms`} />
                <MiniStat label="Avg" value={`${summary.avg_latency_ms}ms`} />
                <MiniStat label="P95" value={`${summary.p95_latency_ms || 0}ms`} />
                <MiniStat label="Max" value={`${summary.max_latency_ms || 0}ms`} />
              </div>
            </section>

            {/* Model Stats */}
            {Object.keys(summary.model_stats).length > 0 && (
              <section className="mb-6">
                <h2
                  className="text-sm font-semibold mb-3"
                  style={{ color: "var(--text-primary)" }}
                >
                  Model Usage
                </h2>
                <div className="space-y-1.5">
                  {Object.entries(summary.model_stats).map(([model, stats]) => (
                    <div
                      key={model}
                      className="flex items-center gap-3 px-4 py-3 rounded-xl"
                      style={{
                        background: "var(--bg-secondary)",
                        border: "1px solid var(--border-color)",
                      }}
                    >
                      <span
                        className="text-sm font-medium flex-1"
                        style={{ color: "var(--text-primary)" }}
                      >
                        {model}
                      </span>
                      <span
                        className="text-xs"
                        style={{ color: "var(--text-muted)" }}
                      >
                        {stats.requests} req
                      </span>
                      <span
                        className="text-xs"
                        style={{ color: "var(--text-muted)" }}
                      >
                        {formatNumber(stats.tokens_in + stats.tokens_out)} tokens
                      </span>
                      <span
                        className="text-xs"
                        style={{ color: "var(--text-muted)" }}
                      >
                        ~{stats.avg_latency_ms}ms
                      </span>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Error Rate + RPM */}
            <div className="grid grid-cols-2 gap-3">
              <div
                className="rounded-xl p-4"
                style={{
                  background: "var(--bg-secondary)",
                  border: "1px solid var(--border-color)",
                }}
              >
                <p
                  className="text-[11px] font-medium mb-1"
                  style={{ color: "var(--text-muted)" }}
                >
                  Error Rate
                </p>
                <p
                  className="text-lg font-bold"
                  style={{
                    color:
                      summary.error_rate > 5
                        ? "var(--error)"
                        : "var(--success)",
                  }}
                >
                  {summary.error_rate}%
                </p>
              </div>
              <div
                className="rounded-xl p-4"
                style={{
                  background: "var(--bg-secondary)",
                  border: "1px solid var(--border-color)",
                }}
              >
                <p
                  className="text-[11px] font-medium mb-1"
                  style={{ color: "var(--text-muted)" }}
                >
                  Requests / Min
                </p>
                <p
                  className="text-lg font-bold"
                  style={{ color: "var(--text-primary)" }}
                >
                  {summary.requests_per_minute}
                </p>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────

function MetricCard({
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
      className="rounded-xl p-3.5"
      style={{
        background: "var(--bg-secondary)",
        border: "1px solid var(--border-color)",
      }}
    >
      <Icon
        size={14}
        style={{ color: "var(--accent)", marginBottom: "6px" }}
      />
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
  );
}

function BarCard({
  label,
  value,
  maxLabel,
}: {
  label: string;
  value: number;
  maxLabel: string;
}) {
  const barColor =
    value > 80 ? "var(--error)" : value > 60 ? "#f59e0b" : "var(--accent)";

  return (
    <div
      className="rounded-xl p-4"
      style={{
        background: "var(--bg-secondary)",
        border: "1px solid var(--border-color)",
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <p
          className="text-xs font-medium"
          style={{ color: "var(--text-secondary)" }}
        >
          {label}
        </p>
        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
          {maxLabel}
        </p>
      </div>
      <div
        className="h-2 rounded-full overflow-hidden"
        style={{ background: "var(--bg-tertiary)" }}
      >
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${Math.min(value, 100)}%`, background: barColor }}
        />
      </div>
      <p
        className="text-lg font-bold mt-1"
        style={{ color: barColor }}
      >
        {Math.round(value)}%
      </p>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <p
        className="text-[10px] font-medium"
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
  );
}
