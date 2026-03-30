"use client";

import React, { useEffect, useState, useCallback } from "react";
import {
  academicListCourses, academicGetCourseMetrics,
  academicGetScoringWeights, academicUpdateScoringWeights,
} from "@/lib/api";
import type { AcademicCourse, CourseMetrics, ScoringWeights } from "@/types/academic";

const CARD_STYLE: React.CSSProperties = {
  background: "rgba(255,255,255,0.04)", borderRadius: "16px",
  border: "1px solid rgba(255,255,255,0.06)", padding: "24px",
  backdropFilter: "blur(8px)",
};

const BTN_STYLE: React.CSSProperties = {
  padding: "10px 20px", border: "none", borderRadius: "10px",
  cursor: "pointer", fontWeight: 600, fontSize: "14px",
  background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
  color: "#fff", transition: "opacity 0.2s",
};

export default function EvaluationPanel() {
  const [courses, setCourses] = useState<AcademicCourse[]>([]);
  const [selectedCourse, setSelectedCourse] = useState("");
  const [metrics, setMetrics] = useState<CourseMetrics | null>(null);
  const [weights, setWeights] = useState<ScoringWeights | null>(null);
  const [editWeights, setEditWeights] = useState<Partial<ScoringWeights>>({});
  const [message, setMessage] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const data = await academicListCourses();
        setCourses(data.courses);
        const wData = await academicGetScoringWeights();
        setWeights(wData.weights);
        setEditWeights(wData.weights);
      } catch (e) { console.error(e); }
    })();
  }, []);

  const loadMetrics = useCallback(async () => {
    if (!selectedCourse) { setMetrics(null); return; }
    try {
      const data = await academicGetCourseMetrics(selectedCourse);
      setMetrics(data.metrics);
    } catch (e) { console.error(e); setMetrics(null); }
  }, [selectedCourse]);

  useEffect(() => { loadMetrics(); }, [loadMetrics]);

  const handleSaveWeights = async () => {
    try {
      const data = await academicUpdateScoringWeights(editWeights);
      setWeights(data.weights);
      setEditWeights(data.weights);
      setMessage("✅ Scoring weights updated");
    } catch (err: any) {
      setMessage(`❌ ${err.message}`);
    }
  };

  const WEIGHT_LABELS: Record<string, { label: string; color: string }> = {
    review_frequency: { label: "Review Frequency", color: "#667eea" },
    urgency_signal: { label: "Urgency Signal", color: "#f5576c" },
    consensus: { label: "Cross-Review Consensus", color: "#43e97b" },
    syllabus_importance: { label: "Syllabus Importance", color: "#4facfe" },
    recency: { label: "Recency Weight", color: "#f093fb" },
    llm_confidence: { label: "LLM Confidence", color: "#ffc107" },
  };

  return (
    <div>
      <h2 style={{ fontSize: "20px", fontWeight: 600, marginBottom: "20px" }}>
        🎯 Evaluation Panel
      </h2>

      {/* Course Metrics */}
      <div style={{ ...CARD_STYLE, marginBottom: "20px" }}>
        <h3 style={{ fontSize: "15px", fontWeight: 600, marginBottom: "12px" }}>
          Course Quality Metrics
        </h3>
        <input
          list="course-list-eval"
          value={selectedCourse}
          onChange={(e) => setSelectedCourse(e.target.value.toUpperCase())}
          placeholder="Enter Course Code (e.g. CS101)"
          style={{
            padding: "10px 14px", borderRadius: "8px", width: "300px",
            border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.04)",
            color: "#e0e0e0", fontSize: "14px", outline: "none",
            marginBottom: "16px",
          }}
        />
        <datalist id="course-list-eval">
          {courses.map((c) => (
            <option key={c.code} value={c.code}>
              {c.code}
            </option>
          ))}
        </datalist>

        {metrics && (
          <div style={{
            display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
            gap: "12px",
          }}>
            {[
              { label: "Total Topics", value: metrics.topic_count, color: "#667eea" },
              { label: "High Confidence", value: metrics.high_confidence, color: "#43e97b" },
              { label: "Medium Confidence", value: metrics.medium_confidence, color: "#ffc107" },
              { label: "Low Confidence", value: metrics.low_confidence, color: "#f5576c" },
              { label: "Avg Exam Prob", value: `${(metrics.avg_exam_probability * 100).toFixed(1)}%`, color: "#4facfe" },
              { label: "Output Count", value: metrics.output_count, color: "#f093fb" },
              { label: "Avg Rating", value: metrics.avg_feedback_rating.toFixed(1), color: "#43e97b" },
              { label: "Feedback Count", value: metrics.feedback_count, color: "#888" },
            ].map((item) => (
              <div key={item.label} style={{
                padding: "14px", borderRadius: "10px",
                background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)",
                textAlign: "center",
              }}>
                <div style={{ fontSize: "24px", fontWeight: 700, color: item.color }}>
                  {item.value}
                </div>
                <div style={{ fontSize: "11px", color: "#888", marginTop: "4px" }}>
                  {item.label}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Scoring Weights Editor */}
      <div style={CARD_STYLE}>
        <h3 style={{ fontSize: "15px", fontWeight: 600, marginBottom: "16px" }}>
          ⚖️ Prediction Scoring Weights
        </h3>
        <p style={{ fontSize: "12px", color: "#888", marginBottom: "16px" }}>
          Adjust the weights used in the exam topic prediction formula. All weights auto-normalize to sum to 1.0.
        </p>

        <div style={{
          display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(250px, 1fr))",
          gap: "14px", marginBottom: "16px",
        }}>
          {Object.entries(WEIGHT_LABELS).map(([key, meta]) => (
            <div key={key} style={{
              padding: "14px", borderRadius: "10px",
              background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)",
            }}>
              <label style={{ fontSize: "13px", fontWeight: 500, color: meta.color }}>
                {meta.label}
              </label>
              <input
                type="range"
                min="0" max="100"
                value={((editWeights as any)[key] || 0) * 100}
                onChange={(e) => setEditWeights(prev => ({ ...prev, [key]: Number(e.target.value) / 100 }))}
                style={{ width: "100%", marginTop: "8px", accentColor: meta.color }}
              />
              <div style={{ fontSize: "12px", color: "#aaa", textAlign: "right" }}>
                {(((editWeights as any)[key] || 0) * 100).toFixed(0)}%
              </div>
            </div>
          ))}
        </div>

        {message && (
          <div style={{
            padding: "8px 12px", borderRadius: "8px", marginBottom: "12px",
            background: message.startsWith("✅") ? "rgba(67,233,123,0.1)" : "rgba(245,87,108,0.1)",
            fontSize: "13px",
          }}>{message}</div>
        )}

        <button style={BTN_STYLE} onClick={handleSaveWeights}>
          💾 Save Weights
        </button>
      </div>
    </div>
  );
}
