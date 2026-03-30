"use client";

import React, { useEffect, useState } from "react";
import { academicGetOverviewMetrics } from "@/lib/api";
import type { AcademicOverviewStats } from "@/types/academic";

const CARD_STYLE: React.CSSProperties = {
  background: "rgba(255,255,255,0.04)", borderRadius: "16px",
  border: "1px solid rgba(255,255,255,0.06)", padding: "24px",
  backdropFilter: "blur(8px)", transition: "transform 0.2s ease",
};

export default function AcademicOverview() {
  const [stats, setStats] = useState<AcademicOverviewStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const data = await academicGetOverviewMetrics();
        setStats(data);
      } catch (e) {
        console.error("Failed to load overview:", e);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) {
    return <div style={{ textAlign: "center", padding: "60px", color: "#888" }}>
      <div style={{ fontSize: "32px", marginBottom: "12px" }}>⏳</div>
      Loading overview...
    </div>;
  }

  const cards = [
    { label: "Total Courses", value: stats?.total_courses ?? 0, icon: "📚", color: "#667eea" },
    { label: "Total Reviews", value: stats?.total_reviews ?? 0, icon: "💬", color: "#764ba2" },
    { label: "Active Jobs", value: stats?.active_jobs ?? 0, icon: "⚡", color: "#f093fb" },
    { label: "Completed Jobs", value: stats?.completed_jobs ?? 0, icon: "✅", color: "#4facfe" },
    { label: "Failed Jobs", value: stats?.failed_jobs ?? 0, icon: "❌", color: "#f5576c" },
    { label: "Total Outputs", value: stats?.total_outputs ?? 0, icon: "📄", color: "#43e97b" },
  ];

  return (
    <div>
      <h2 style={{ fontSize: "20px", fontWeight: 600, marginBottom: "20px" }}>
        System Overview
      </h2>

      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
        gap: "16px", marginBottom: "32px",
      }}>
        {cards.map((card) => (
          <div key={card.label} style={CARD_STYLE}>
            <div style={{ fontSize: "24px", marginBottom: "8px" }}>{card.icon}</div>
            <div style={{ fontSize: "32px", fontWeight: 700, color: card.color }}>
              {card.value}
            </div>
            <div style={{ fontSize: "13px", color: "#888", marginTop: "4px" }}>
              {card.label}
            </div>
          </div>
        ))}
      </div>

      {/* Status indicator */}
      <div style={{
        ...CARD_STYLE, display: "flex", alignItems: "center", gap: "12px",
      }}>
        <div style={{
          width: "12px", height: "12px", borderRadius: "50%",
          background: stats?.system_status === "operational" ? "#43e97b" : "#f5576c",
          boxShadow: `0 0 8px ${stats?.system_status === "operational" ? "#43e97b" : "#f5576c"}`,
        }} />
        <span style={{ fontSize: "14px", fontWeight: 500 }}>
          System Status: {stats?.system_status ?? "Unknown"}
        </span>
      </div>
    </div>
  );
}
