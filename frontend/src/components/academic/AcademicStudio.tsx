"use client";

import React, { useState } from "react";
import AcademicOverview from "./AcademicOverview";
import CourseLibrary from "./CourseLibrary";
import ReviewsCenter from "./ReviewsCenter";
import GenerationCenter from "./GenerationCenter";
import OutputsArchive from "./OutputsArchive";
import EvaluationPanel from "./EvaluationPanel";
import type { AcademicTab } from "@/types/academic";

const TABS: { id: AcademicTab; label: string; icon: string }[] = [
  { id: "overview", label: "Overview", icon: "📊" },
  { id: "courses", label: "Course Library", icon: "📚" },
  { id: "reviews", label: "Reviews", icon: "💬" },
  { id: "generation", label: "Generation", icon: "⚡" },
  { id: "outputs", label: "Outputs", icon: "📄" },
  { id: "evaluation", label: "Evaluation", icon: "🎯" },
];

export default function AcademicStudio() {
  const [activeTab, setActiveTab] = useState<AcademicTab>("overview");

  return (
    <div style={{
      width: "100%", minHeight: "100vh",
      background: "linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%)",
      color: "#e0e0e0", fontFamily: "'Inter', system-ui, sans-serif",
    }}>
      {/* Header */}
      <div style={{
        padding: "24px 32px", borderBottom: "1px solid rgba(255,255,255,0.06)",
        background: "rgba(0,0,0,0.2)", backdropFilter: "blur(12px)",
      }}>
        <h1 style={{
          fontSize: "28px", fontWeight: 700, margin: 0,
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
        }}>
          🎓 Academic Studio
        </h1>
        <p style={{ margin: "4px 0 0", color: "#888", fontSize: "14px" }}>
          VU Academic Content Auto-Generation System
        </p>
      </div>

      {/* Tab Navigation */}
      <div style={{
        display: "flex", gap: "4px", padding: "12px 32px",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        background: "rgba(0,0,0,0.1)", overflowX: "auto",
      }}>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: "10px 20px", border: "none", borderRadius: "8px",
              cursor: "pointer", fontSize: "14px", fontWeight: 500,
              transition: "all 0.2s ease",
              background: activeTab === tab.id
                ? "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                : "rgba(255,255,255,0.04)",
              color: activeTab === tab.id ? "#fff" : "#aaa",
              whiteSpace: "nowrap",
            }}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div style={{ padding: "24px 32px" }}>
        {activeTab === "overview" && <AcademicOverview />}
        {activeTab === "courses" && <CourseLibrary />}
        {activeTab === "reviews" && <ReviewsCenter />}
        {activeTab === "generation" && <GenerationCenter />}
        {activeTab === "outputs" && <OutputsArchive />}
        {activeTab === "evaluation" && <EvaluationPanel />}
      </div>
    </div>
  );
}
