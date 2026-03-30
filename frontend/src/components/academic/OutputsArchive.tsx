"use client";

import React, { useEffect, useState, useCallback } from "react";
import {
  academicListCourses, academicListOutputs, academicDownloadUrl,
  academicSubmitFeedback, academicDeleteOutput,
} from "@/lib/api";
import type { AcademicCourse, AcademicOutput } from "@/types/academic";

const CARD_STYLE: React.CSSProperties = {
  background: "rgba(255,255,255,0.04)", borderRadius: "16px",
  border: "1px solid rgba(255,255,255,0.06)", padding: "24px",
  backdropFilter: "blur(8px)",
};

const BTN_STYLE: React.CSSProperties = {
  padding: "8px 16px", border: "none", borderRadius: "8px",
  cursor: "pointer", fontWeight: 600, fontSize: "13px",
  background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
  color: "#fff",
};

const TYPE_LABELS: Record<string, { label: string; icon: string; color: string }> = {
  highlighted_handout: { label: "Highlighted Handout", icon: "📝", color: "#667eea" },
  midterm_notes: { label: "Midterm Notes", icon: "📋", color: "#4facfe" },
  final_notes: { label: "Final Notes", icon: "📖", color: "#43e97b" },
  mcqs: { label: "MCQ Bank", icon: "❓", color: "#f093fb" },
};

export default function OutputsArchive() {
  const [courses, setCourses] = useState<AcademicCourse[]>([]);
  const [selectedCourse, setSelectedCourse] = useState("");
  const [outputs, setOutputs] = useState<AcademicOutput[]>([]);
  const [loading, setLoading] = useState(false);
  const [feedbackMap, setFeedbackMap] = useState<Record<string, number>>({});

  useEffect(() => {
    (async () => {
      try {
        const data = await academicListCourses();
        setCourses(data.courses);
      } catch (e) { console.error(e); }
    })();
  }, []);

  const loadOutputs = useCallback(async () => {
    setLoading(true);
    try {
      const data = await academicListOutputs(selectedCourse || undefined);
      setOutputs(data.outputs);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, [selectedCourse]);

  useEffect(() => { loadOutputs(); }, [loadOutputs]);

  const handleFeedback = async (outputId: string, rating: number) => {
    try {
      await academicSubmitFeedback(outputId, rating);
      setFeedbackMap(prev => ({ ...prev, [outputId]: rating }));
    } catch (e) { console.error(e); }
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div>
      <h2 style={{ fontSize: "20px", fontWeight: 600, marginBottom: "20px" }}>
        📄 Outputs Archive
      </h2>

      {/* Course Filter */}
      <div style={{ ...CARD_STYLE, marginBottom: "20px", display: "flex", gap: "12px", alignItems: "center" }}>
        <label style={{ fontWeight: 500, fontSize: "14px" }}>Filter by course:</label>
        <input
          list="course-list-out"
          value={selectedCourse}
          onChange={(e) => setSelectedCourse(e.target.value.toUpperCase())}
          placeholder="All Courses (or type course code)"
          style={{
            padding: "10px 14px", borderRadius: "8px", width: "300px",
            border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.04)",
            color: "#e0e0e0", fontSize: "14px", outline: "none",
          }}
        />
        <datalist id="course-list-out">
          {courses.map((c) => (
            <option key={c.code} value={c.code}>
              {c.code}
            </option>
          ))}
        </datalist>
      </div>

      {/* Outputs Grid */}
      {loading ? (
        <div style={{ textAlign: "center", padding: "40px", color: "#888" }}>Loading outputs...</div>
      ) : outputs.length === 0 ? (
        <div style={{ ...CARD_STYLE, textAlign: "center", padding: "48px" }}>
          <div style={{ fontSize: "48px", marginBottom: "12px" }}>📭</div>
          <p style={{ color: "#888" }}>No outputs yet. Generate content in the Generation tab.</p>
        </div>
      ) : (
        <div style={{
          display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
          gap: "16px",
        }}>
          {outputs.map((output) => {
            const meta = TYPE_LABELS[output.output_type] || { label: output.output_type, icon: "📄", color: "#888" };
            const currentRating = feedbackMap[output.id] || 0;

            return (
              <div key={output.id} style={CARD_STYLE}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start" }}>
                  <div>
                    <span style={{ fontSize: "24px" }}>{meta.icon}</span>
                    <h4 style={{
                      fontSize: "16px", fontWeight: 600, margin: "6px 0 4px",
                      color: meta.color,
                    }}>
                      {meta.label}
                    </h4>
                    <div style={{ fontSize: "12px", color: "#888" }}>
                      Version {output.version} • {output.topic_count} topics • {formatSize(output.file_size)}
                    </div>
                  </div>
                  <a
                    href={academicDownloadUrl(output.id)}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={BTN_STYLE}
                  >
                    ⬇️ Download
                  </a>
                </div>

                <div style={{ marginTop: "10px" }}>
                  <button
                    onClick={async () => {
                      await academicDeleteOutput(output.id);
                      await loadOutputs();
                    }}
                    style={{ ...BTN_STYLE, background: "rgba(245,87,108,0.28)", padding: "6px 12px" }}
                  >
                    🗑 Delete PDF
                  </button>
                </div>

                {/* Star Rating */}
                <div style={{ marginTop: "14px", display: "flex", gap: "4px", alignItems: "center" }}>
                  <span style={{ fontSize: "12px", color: "#888", marginRight: "8px" }}>Rate:</span>
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button
                      key={star}
                      onClick={() => handleFeedback(output.id, star)}
                      style={{
                        border: "none", background: "none", cursor: "pointer",
                        fontSize: "20px", opacity: star <= currentRating ? 1 : 0.3,
                        transition: "opacity 0.2s",
                      }}
                    >
                      ⭐
                    </button>
                  ))}
                </div>

                <div style={{ fontSize: "11px", color: "#555", marginTop: "8px" }}>
                  Created: {new Date(output.created_at).toLocaleDateString()}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
