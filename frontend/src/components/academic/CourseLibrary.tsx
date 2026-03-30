"use client";

import React, { useEffect, useState, useCallback } from "react";
import { academicListCourses, academicBulkImport, academicUploadPdf } from "@/lib/api";
import type { AcademicCourse, BulkImportResult } from "@/types/academic";

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

export default function CourseLibrary() {
  const [courses, setCourses] = useState<AcademicCourse[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [lastResult, setLastResult] = useState<BulkImportResult | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const loadCourses = useCallback(async () => {
    try {
      const data = await academicListCourses();
      setCourses(data.courses);
    } catch (e) {
      console.error("Failed to load courses:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadCourses(); }, [loadCourses]);

  const handleFiles = async (files: FileList | File[]) => {
    const pdfFiles = Array.from(files).filter(f => f.name.endsWith(".pdf"));
    if (pdfFiles.length === 0) return;

    setUploading(true);
    setLastResult(null);
    try {
      const result = await academicBulkImport(pdfFiles);
      setLastResult(result);
      await loadCourses();
    } catch (e: any) {
      setLastResult({ total_files: pdfFiles.length, success_count: 0, failed_count: pdfFiles.length, results: [{ filename: "error", error: e.message }] });
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length > 0) handleFiles(e.dataTransfer.files);
  };

  return (
    <div>
      <h2 style={{ fontSize: "20px", fontWeight: 600, marginBottom: "20px" }}>
        📚 Course Library
      </h2>

      {/* Drag & Drop Upload Zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        style={{
          ...CARD_STYLE, textAlign: "center", padding: "48px 24px",
          border: dragOver
            ? "2px dashed #667eea"
            : "2px dashed rgba(255,255,255,0.12)",
          background: dragOver ? "rgba(102,126,234,0.05)" : "rgba(255,255,255,0.02)",
          transition: "all 0.3s ease", marginBottom: "24px", cursor: "pointer",
        }}
        onClick={() => {
          const input = document.createElement("input");
          input.type = "file"; input.multiple = true; input.accept = ".pdf";
          input.onchange = (e) => {
            const files = (e.target as HTMLInputElement).files;
            if (files) handleFiles(files);
          };
          input.click();
        }}
      >
        <div style={{ fontSize: "48px", marginBottom: "12px" }}>
          {uploading ? "⏳" : "📂"}
        </div>
        <p style={{ fontSize: "16px", fontWeight: 500, margin: "0 0 8px" }}>
          {uploading ? "Uploading & Vectorizing..." : "Drop PDFs here or click to browse"}
        </p>
        <p style={{ fontSize: "13px", color: "#888", margin: 0 }}>
          Course code will be auto-detected from filenames (e.g. CS201_Handout.pdf)
        </p>
        <p style={{ fontSize: "12px", color: "#666", margin: "4px 0 0" }}>
          Each PDF is stored in an isolated per-course vector workspace
        </p>
      </div>

      {/* Upload Results */}
      {lastResult && (
        <div style={{ ...CARD_STYLE, marginBottom: "24px" }}>
          <h3 style={{ fontSize: "16px", fontWeight: 600, marginBottom: "12px" }}>
            Upload Results — {lastResult.success_count}/{lastResult.total_files} successful
          </h3>
          {lastResult.results.map((r, i) => (
            <div key={i} style={{
              padding: "10px 14px", borderRadius: "8px", marginBottom: "6px",
              background: r.error ? "rgba(245,87,108,0.1)" : "rgba(67,233,123,0.1)",
              border: `1px solid ${r.error ? "rgba(245,87,108,0.2)" : "rgba(67,233,123,0.2)"}`,
              fontSize: "13px",
            }}>
              <strong>{r.filename}</strong>
              {r.course_code && <span style={{ color: "#667eea" }}> → {r.course_code}</span>}
              {r.error && <span style={{ color: "#f5576c" }}> — {r.error}</span>}
              {r.needs_manual_mapping && <span style={{ color: "#ffc107" }}> ⚠️ Needs manual mapping</span>}
            </div>
          ))}
        </div>
      )}

      {/* Course Grid */}
      {loading ? (
        <div style={{ textAlign: "center", padding: "40px", color: "#888" }}>Loading courses...</div>
      ) : courses.length === 0 ? (
        <div style={{ ...CARD_STYLE, textAlign: "center", padding: "48px" }}>
          <div style={{ fontSize: "48px", marginBottom: "12px" }}>📭</div>
          <p style={{ color: "#888" }}>No courses yet. Upload PDFs to get started.</p>
        </div>
      ) : (
        <div style={{
          display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
          gap: "16px",
        }}>
          {courses.map((course) => (
            <div key={course.id} style={CARD_STYLE}>
              <div style={{
                fontSize: "20px", fontWeight: 700, marginBottom: "8px",
                background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              }}>
                {course.code}
              </div>
              <div style={{ fontSize: "14px", color: "#aaa", marginBottom: "12px" }}>
                {course.title || "Untitled Course"}
              </div>
              <div style={{
                display: "flex", gap: "16px", fontSize: "12px", color: "#888",
              }}>
                <span>📝 {course.review_count ?? 0} reviews</span>
                <span>🏷️ {course.department}</span>
              </div>
              <div style={{
                marginTop: "8px", fontSize: "11px", color: "#555",
                fontFamily: "monospace",
              }}>
                workspace: {course.workspace_id}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
