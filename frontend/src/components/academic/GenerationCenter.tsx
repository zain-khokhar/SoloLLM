"use client";

import React, { useEffect, useState, useCallback } from "react";
import {
  academicListCourses, academicGenerate, academicListJobs,
  academicRetryJob, academicCancelJob, academicPauseJob,
  academicResumeJob, academicStopJob, academicDeleteJob,
  academicListGenerationModels, academicPreviewDownloadUrl,
} from "@/lib/api";
import type { AcademicCourse, AcademicJob } from "@/types/academic";
import type { ModelInfo } from "@/types";

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

const OUTPUT_TYPES = [
  { id: "highlighted_handout", label: "Highlighted Handout", icon: "📝" },
  { id: "midterm_notes", label: "Midterm Short Notes", icon: "📋" },
  { id: "final_notes", label: "Final Short Notes", icon: "📖" },
  { id: "mcqs", label: "MCQ Bank", icon: "❓" },
];

const STATUS_COLORS: Record<string, string> = {
  pending: "#ffc107",
  running: "#4facfe",
  paused: "#ff9800",
  completed: "#43e97b",
  failed: "#f5576c",
  cancelled: "#888",
};

export default function GenerationCenter() {
  const [courses, setCourses] = useState<AcademicCourse[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedCourse, setSelectedCourse] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedTypes, setSelectedTypes] = useState<string[]>(["highlighted_handout", "midterm_notes", "final_notes", "mcqs"]);
  const [batchPages, setBatchPages] = useState(5);
  const [maxBatchChars, setMaxBatchChars] = useState(7000);
  const [maxReviewEvidenceChars, setMaxReviewEvidenceChars] = useState(2500);
  const [maxSpansPerBatch, setMaxSpansPerBatch] = useState(25);
  const [syllabusScope, setSyllabusScope] = useState<"all" | "midterm" | "final">("all");
  const [splitMode, setSplitMode] = useState<"auto" | "manual" | "none">("auto");
  const [manualMidtermEndPage, setManualMidtermEndPage] = useState<number>(237);
  const [jobs, setJobs] = useState<AcademicJob[]>([]);
  const [generating, setGenerating] = useState(false);
  const [message, setMessage] = useState("");
  const [lastGenerationPayload, setLastGenerationPayload] = useState<Record<string, unknown> | null>(null);
  const [lastSubmittedJobId, setLastSubmittedJobId] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const data = await academicListCourses();
        setCourses(data.courses);
      } catch (e) { console.error(e); }
    })();
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const m = await academicListGenerationModels();
        setModels(m);
        if (m.length > 0 && !selectedModel) {
          setSelectedModel(m[0].name);
        }
      } catch (e) { console.error(e); }
    })();
  }, [selectedModel]);

  const loadJobs = useCallback(async () => {
    try {
      const data = await academicListJobs(selectedCourse || undefined, undefined, 20);
      setJobs(data.jobs);
    } catch (e) { console.error(e); }
  }, [selectedCourse]);

  useEffect(() => { loadJobs(); const id = setInterval(loadJobs, 2000); return () => clearInterval(id); }, [loadJobs]);

  const handleGenerate = async () => {
    if (!selectedCourse) { setMessage("⚠️ Select a course first!"); return; }
    if (selectedTypes.length === 0) { setMessage("⚠️ Select at least one output type!"); return; }
    setGenerating(true); setMessage("");
    try {
      const options = {
        generation_model: selectedModel || undefined,
        batch_pages: batchPages,
        max_batch_chars: maxBatchChars,
        max_review_evidence_chars: maxReviewEvidenceChars,
        max_spans_per_batch: maxSpansPerBatch,
        syllabus_scope: syllabusScope,
        split_mode: splitMode,
        manual_midterm_end_page: splitMode === "manual" ? manualMidtermEndPage : undefined,
      };
      const requestPayload: Record<string, unknown> = {
        course_code: selectedCourse,
        output_types: selectedTypes,
        ...options,
      };
      setLastGenerationPayload(requestPayload);
      const job = await academicGenerate(selectedCourse, selectedTypes, options);
      setLastSubmittedJobId(job.id);
      setMessage(`✅ Job submitted: ${job.id}`);
      loadJobs();
    } catch (err: any) {
      setMessage(`❌ ${err.message}`);
    } finally {
      setGenerating(false);
    }
  };

  const toggleType = (type: string) => {
    setSelectedTypes(prev =>
      prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type]
    );
  };

  const liveJob =
    jobs.find((j) => j.id === lastSubmittedJobId)
    || jobs.find((j) => j.status === "running" || j.status === "pending" || j.status === "paused")
    || jobs[0];
  const liveAiLogs = liveJob?.checkpoint_json?.ai_logs || [];

  return (
    <div>
      <h2 style={{ fontSize: "20px", fontWeight: 600, marginBottom: "20px" }}>
        ⚡ Generation Center
      </h2>

      {/* Course Selection (MANDATORY) */}
      <div style={{ ...CARD_STYLE, marginBottom: "20px" }}>
        <h3 style={{ fontSize: "15px", fontWeight: 600, marginBottom: "12px" }}>
          1️⃣ Select Course <span style={{ color: "#f5576c", fontSize: "12px" }}>(required — isolation enforced)</span>
        </h3>
        <input
          list="course-list-gen"
          value={selectedCourse}
          onChange={(e) => setSelectedCourse(e.target.value.toUpperCase())}
          placeholder="Enter Course Code (e.g. CS101)"
          style={{
            padding: "12px 16px", borderRadius: "10px", width: "100%",
            border: "1px solid rgba(102,126,234,0.3)", background: "rgba(255,255,255,0.04)",
            color: "#e0e0e0", fontSize: "16px", outline: "none",
          }}
        />
        <datalist id="course-list-gen">
          {courses.map((c) => (
            <option key={c.code} value={c.code}>
              {c.title || "Untitled"} ({c.review_count ?? 0} reviews)
            </option>
          ))}
        </datalist>
        <p style={{ margin: "8px 0 0", fontSize: "12px", color: "#888" }}>
          ⚠️ Only the selected course's PDFs and reviews will be used. No cross-course data.
        </p>
      </div>

      {/* Output Type Selection */}
      <div style={{ ...CARD_STYLE, marginBottom: "20px" }}>
        <h3 style={{ fontSize: "15px", fontWeight: 600, marginBottom: "12px" }}>
          2️⃣ Select Output Types
        </h3>
        <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
          {OUTPUT_TYPES.map((t) => (
            <button
              key={t.id}
              onClick={() => toggleType(t.id)}
              style={{
                padding: "10px 18px", borderRadius: "10px", fontSize: "14px",
                border: selectedTypes.includes(t.id)
                  ? "2px solid #667eea" : "2px solid rgba(255,255,255,0.08)",
                background: selectedTypes.includes(t.id) ? "rgba(102,126,234,0.15)" : "rgba(255,255,255,0.02)",
                color: selectedTypes.includes(t.id) ? "#fff" : "#aaa",
                cursor: "pointer", transition: "all 0.2s",
              }}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Highlight Controls */}
      <div style={{ ...CARD_STYLE, marginBottom: "20px" }}>
        <h3 style={{ fontSize: "15px", fontWeight: 600, marginBottom: "12px" }}>
          3️⃣ Highlight Settings (for proper yellow highlights)
        </h3>
        <div style={{ display: "grid", gap: "10px", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
          <div>
            <label style={{ fontSize: "12px", color: "#aaa" }}>Generation model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{ width: "100%", marginTop: "6px", padding: "10px", borderRadius: "8px", background: "rgba(255,255,255,0.04)", color: "#fff", border: "1px solid rgba(255,255,255,0.12)" }}
            >
              {models.map((m) => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
            {(() => {
              const model = models.find(m => m.name === selectedModel);
              const paramSize = model?.parameter_size;
              if (paramSize) {
                const sizeNum = parseFloat(paramSize.replace(/[^0-9.]/g, ''));
                if (sizeNum > 0 && sizeNum < 7) {
                  return (
                    <div style={{
                      marginTop: "6px", padding: "8px 12px", borderRadius: "8px",
                      background: "rgba(255,193,7,0.15)", border: "1px solid rgba(255,193,7,0.3)",
                      fontSize: "12px", color: "#ffc107",
                    }}>
                      {selectedModel} ({paramSize}) is too small. Models under 7B produce garbage highlights. Use 7B+ (e.g., qwen2.5:7b, llama3:8b, deepseek-r1:7b).
                    </div>
                  );
                }
              }
              return null;
            })()}
          </div>

          <div>
            <label style={{ fontSize: "12px", color: "#aaa" }}>Batch pages (sequential)</label>
            <input
              type="number"
              min={1}
              max={10}
              value={batchPages}
              onChange={(e) => setBatchPages(Number(e.target.value) || 5)}
              style={{ width: "100%", marginTop: "6px", padding: "10px", borderRadius: "8px", background: "rgba(255,255,255,0.04)", color: "#fff", border: "1px solid rgba(255,255,255,0.12)" }}
            />
          </div>

          <div>
            <label style={{ fontSize: "12px", color: "#aaa" }}>Max chars per batch</label>
            <input
              type="number"
              min={1200}
              max={30000}
              value={maxBatchChars}
              onChange={(e) => setMaxBatchChars(Number(e.target.value) || 7000)}
              style={{ width: "100%", marginTop: "6px", padding: "10px", borderRadius: "8px", background: "rgba(255,255,255,0.04)", color: "#fff", border: "1px solid rgba(255,255,255,0.12)" }}
            />
          </div>

          <div>
            <label style={{ fontSize: "12px", color: "#aaa" }}>Review evidence chars</label>
            <input
              type="number"
              min={500}
              max={12000}
              value={maxReviewEvidenceChars}
              onChange={(e) => setMaxReviewEvidenceChars(Number(e.target.value) || 2500)}
              style={{ width: "100%", marginTop: "6px", padding: "10px", borderRadius: "8px", background: "rgba(255,255,255,0.04)", color: "#fff", border: "1px solid rgba(255,255,255,0.12)" }}
            />
          </div>

          <div>
            <label style={{ fontSize: "12px", color: "#aaa" }}>Max spans per batch</label>
            <input
              type="number"
              min={5}
              max={60}
              value={maxSpansPerBatch}
              onChange={(e) => setMaxSpansPerBatch(Number(e.target.value) || 20)}
              style={{ width: "100%", marginTop: "6px", padding: "10px", borderRadius: "8px", background: "rgba(255,255,255,0.04)", color: "#fff", border: "1px solid rgba(255,255,255,0.12)" }}
            />
          </div>

          <div>
            <label style={{ fontSize: "12px", color: "#aaa" }}>Syllabus scope</label>
            <select
              value={syllabusScope}
              onChange={(e) => setSyllabusScope(e.target.value as any)}
              style={{ width: "100%", marginTop: "6px", padding: "10px", borderRadius: "8px", background: "rgba(255,255,255,0.04)", color: "#fff", border: "1px solid rgba(255,255,255,0.12)" }}
            >
              <option value="all">All pages</option>
              <option value="midterm">Midterm pages only</option>
              <option value="final">Final pages only</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: "12px", color: "#aaa" }}>Mid/final split mode</label>
            <select
              value={splitMode}
              onChange={(e) => setSplitMode(e.target.value as any)}
              style={{ width: "100%", marginTop: "6px", padding: "10px", borderRadius: "8px", background: "rgba(255,255,255,0.04)", color: "#fff", border: "1px solid rgba(255,255,255,0.12)" }}
            >
              <option value="auto">Auto formula (500 → 237 mids)</option>
              <option value="manual">Manual page split</option>
              <option value="none">No split</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: "12px", color: "#aaa" }}>Manual midterm end page</label>
            <input
              type="number"
              min={1}
              value={manualMidtermEndPage}
              disabled={splitMode !== "manual"}
              onChange={(e) => setManualMidtermEndPage(Number(e.target.value) || 1)}
              style={{ width: "100%", marginTop: "6px", padding: "10px", borderRadius: "8px", background: "rgba(255,255,255,0.04)", color: "#fff", border: "1px solid rgba(255,255,255,0.12)", opacity: splitMode === "manual" ? 1 : 0.5 }}
            />
          </div>
        </div>
      </div>

      {/* Generate Button */}
      <div style={{ marginBottom: "20px" }}>
        <button
          onClick={handleGenerate}
          disabled={generating || !selectedCourse}
          style={{
            ...BTN_STYLE, fontSize: "16px", padding: "14px 32px",
            opacity: (generating || !selectedCourse) ? 0.5 : 1,
            width: "100%",
          }}
        >
          {generating ? "⏳ Generating..." : "🚀 Generate Content"}
        </button>
      </div>

      {message && (
        <div style={{
          padding: "10px 14px", borderRadius: "8px", marginBottom: "16px",
          background: message.startsWith("✅") ? "rgba(67,233,123,0.1)" : message.startsWith("⚠") ? "rgba(255,193,7,0.1)" : "rgba(245,87,108,0.1)",
          fontSize: "13px",
        }}>{message}</div>
      )}

      {(lastGenerationPayload || liveJob) && (
        <div style={{ ...CARD_STYLE, marginBottom: "20px", border: "1px solid rgba(79,172,254,0.35)", background: "rgba(79,172,254,0.08)" }}>
          <h3 style={{ fontSize: "15px", fontWeight: 700, marginBottom: "10px", color: "#b7dfff" }}>
            Live Generation Logs
          </h3>

          <div style={{ fontSize: "12px", color: "#c9d6e2", marginBottom: "8px" }}>
            Job: {liveJob?.id || "(waiting for job id)"}
            {liveJob && <span style={{ marginLeft: "12px" }}>Status: {liveJob.status}</span>}
            {liveJob?.current_stage && <span style={{ marginLeft: "12px" }}>Stage: {liveJob.current_stage}</span>}
          </div>

          <div style={{ fontSize: "12px", color: "#9ad0ff", marginBottom: "4px" }}>Request sent to backend:</div>
          <pre style={{ margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: "11px", color: "#d8f0ff", background: "rgba(24,35,49,0.75)", borderRadius: "6px", padding: "8px", maxHeight: "200px", overflowY: "auto" }}>
            {lastGenerationPayload ? JSON.stringify(lastGenerationPayload, null, 2) : "(not submitted yet)"}
          </pre>

          <div style={{ fontSize: "12px", color: "#9af5b6", margin: "10px 0 6px" }}>AI request/response stream:</div>
          <div style={{ display: "grid", gap: "8px", maxHeight: "420px", overflowY: "auto" }}>
            {liveAiLogs.length === 0 && (
              <div style={{ fontSize: "12px", color: "#9aa", background: "rgba(0,0,0,0.18)", borderRadius: "6px", padding: "8px" }}>
                Waiting for AI calls... logs will appear here as soon as generation starts model inference.
              </div>
            )}

            {liveAiLogs.map((log, index) => (
              <div key={`live-ai-log-${index}`} style={{ border: "1px solid rgba(255,255,255,0.08)", borderRadius: "8px", padding: "8px", background: "rgba(0,0,0,0.18)" }}>
                <div style={{ fontSize: "11px", color: "#ddd", marginBottom: "6px" }}>
                  #{index + 1} • {log.content_type} • {log.topic_name || "n/a"} • {log.model || "default model"}
                </div>

                <div style={{ fontSize: "11px", color: "#9ad0ff", marginBottom: "4px" }}>Request sent to AI:</div>
                <pre style={{ margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: "11px", color: "#d8f0ff", background: "rgba(24,35,49,0.75)", borderRadius: "6px", padding: "8px", maxHeight: "180px", overflowY: "auto" }}>
                  {log.request || "(empty request)"}
                </pre>

                <div style={{ fontSize: "11px", color: "#9af5b6", margin: "8px 0 4px" }}>Response received from AI:</div>
                <pre style={{ margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: "11px", color: "#d6ffe2", background: "rgba(21,44,30,0.7)", borderRadius: "6px", padding: "8px", maxHeight: "180px", overflowY: "auto" }}>
                  {log.response || "(no response captured yet)"}
                </pre>

                {log.error && (
                  <div style={{ marginTop: "6px", fontSize: "11px", color: "#ff9aa2" }}>
                    Error: {log.error}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Job Queue */}
      <div style={CARD_STYLE}>
        <h3 style={{ fontSize: "15px", fontWeight: 600, marginBottom: "12px" }}>
          📋 Job Queue ({jobs.length})
        </h3>
        {jobs.length === 0 ? (
          <div style={{ color: "#888", textAlign: "center", padding: "20px" }}>
            No jobs yet. Generate content above.
          </div>
        ) : (
          <div style={{ maxHeight: "400px", overflowY: "auto" }}>
            {jobs.map((job) => (
              <div key={job.id} style={{
                padding: "14px 16px", borderRadius: "10px", marginBottom: "8px",
                background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)",
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <span style={{
                      display: "inline-block", padding: "2px 10px", borderRadius: "12px",
                      fontSize: "11px", fontWeight: 600,
                      background: `${STATUS_COLORS[job.status] || "#888"}22`,
                      color: STATUS_COLORS[job.status] || "#888",
                      border: `1px solid ${STATUS_COLORS[job.status] || "#888"}44`,
                    }}>
                      {job.status.toUpperCase()}
                    </span>
                    <span style={{ marginLeft: "12px", fontSize: "13px", color: "#aaa" }}>
                      {job.id.slice(0, 8)}...
                    </span>
                  </div>
                  <div style={{ display: "flex", gap: "6px" }}>
                    {job.status === "failed" && (
                      <button onClick={() => academicRetryJob(job.id).then(loadJobs)}
                        style={{ ...BTN_STYLE, padding: "4px 12px", fontSize: "11px" }}>🔄 Retry</button>
                    )}
                    {job.status === "running" && (
                      <>
                        <button onClick={() => academicPauseJob(job.id).then(loadJobs)}
                          style={{ ...BTN_STYLE, padding: "4px 12px", fontSize: "11px", background: "rgba(255,152,0,0.35)" }}>⏸ Pause</button>
                        <button onClick={() => academicStopJob(job.id).then(loadJobs)}
                          style={{ ...BTN_STYLE, padding: "4px 12px", fontSize: "11px", background: "rgba(245,87,108,0.35)" }}>⏹ Stop</button>
                      </>
                    )}
                    {job.status === "paused" && (
                      <>
                        <button onClick={() => academicResumeJob(job.id).then(loadJobs)}
                          style={{ ...BTN_STYLE, padding: "4px 12px", fontSize: "11px", background: "rgba(67,233,123,0.25)" }}>▶ Resume</button>
                        <button onClick={() => academicStopJob(job.id).then(loadJobs)}
                          style={{ ...BTN_STYLE, padding: "4px 12px", fontSize: "11px", background: "rgba(245,87,108,0.35)" }}>⏹ Stop</button>
                      </>
                    )}
                    {(job.status === "failed" || job.status === "cancelled" || job.status === "completed") && (
                      <button onClick={() => academicDeleteJob(job.id).then(loadJobs)}
                        style={{ ...BTN_STYLE, padding: "4px 12px", fontSize: "11px", background: "rgba(130,130,130,0.3)" }}>🗑 Delete</button>
                    )}
                    {(job.status === "running" || job.status === "paused" || job.status === "completed") && (
                      <a
                        href={academicPreviewDownloadUrl(job.id)}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ ...BTN_STYLE, padding: "4px 12px", fontSize: "11px", background: "rgba(79,172,254,0.3)", textDecoration: "none" }}
                      >
                        👁 Preview
                      </a>
                    )}
                  </div>
                </div>

                {/* Progress Bar */}
                {(job.status === "running" || job.status === "pending") && (
                  <div style={{
                    marginTop: "10px", height: "4px", borderRadius: "4px",
                    background: "rgba(255,255,255,0.06)", overflow: "hidden",
                  }}>
                    <div style={{
                      height: "100%", borderRadius: "4px",
                      background: "linear-gradient(90deg, #667eea, #764ba2)",
                      width: `${(job.progress || 0) * 100}%`,
                      transition: "width 0.5s ease",
                    }} />
                  </div>
                )}

                <div style={{ marginTop: "8px", fontSize: "11px", color: "#666" }}>
                  {job.current_stage && `Stage: ${job.current_stage}`}
                  {job.error && <span style={{ color: "#f5576c" }}> — {job.error}</span>}
                  <span style={{ marginLeft: "12px" }}>Types: {job.output_types?.join(", ")}</span>
                </div>

                {job.checkpoint_json?.preview && (
                  <div style={{ marginTop: "8px", fontSize: "11px", color: "#9aa" }}>
                    <div>
                      Preview: pages processed {(job.checkpoint_json.preview.processed_pages || []).length} • highlights {job.checkpoint_json.preview.highlights_applied || 0}
                    </div>
                    <div style={{ marginTop: "2px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      Pages: {(job.checkpoint_json.preview.processed_pages || []).slice(0, 20).join(", ") || "-"}
                    </div>
                  </div>
                )}

                {job.checkpoint_json?.render_stats && (
                  <div style={{ marginTop: "4px", fontSize: "11px", color: "#9aa" }}>
                    Match rate: {((job.checkpoint_json.render_stats.match_rate || 0) * 100).toFixed(1)}%
                    {" | "}Highlights: {job.checkpoint_json.render_stats.highlight_count || 0}
                    {" | "}Unmatched: {job.checkpoint_json.render_stats.unmatched_spans || 0}
                    {" | "}Failed batches: {job.checkpoint_json.render_stats.failed_batches || 0}
                  </div>
                )}

                {job.checkpoint_json?.model_warning && (
                  <div style={{ marginTop: "4px", fontSize: "11px", color: "#ffc107", background: "rgba(255,193,7,0.1)", padding: "4px 8px", borderRadius: "4px" }}>
                    {job.checkpoint_json.model_warning}
                  </div>
                )}

                {job.checkpoint_json?.quality_warning && (
                  <div style={{ marginTop: "4px", fontSize: "11px", color: "#f5576c", background: "rgba(245,87,108,0.1)", padding: "4px 8px", borderRadius: "4px" }}>
                    {job.checkpoint_json.quality_warning}
                  </div>
                )}

                {!!(job.checkpoint_json?.ai_logs && job.checkpoint_json.ai_logs.length > 0) && (
                  <details style={{ marginTop: "10px", border: "1px solid rgba(79,172,254,0.25)", borderRadius: "8px", padding: "8px", background: "rgba(79,172,254,0.06)" }}>
                    <summary style={{ cursor: "pointer", fontSize: "12px", fontWeight: 600, color: "#b7dfff" }}>
                      AI Logs ({job.checkpoint_json?.ai_logs?.length || 0})
                    </summary>
                    <div style={{ marginTop: "8px", display: "grid", gap: "8px", maxHeight: "380px", overflowY: "auto" }}>
                      {(job.checkpoint_json?.ai_logs || []).map((log, index) => (
                        <div key={`${job.id}-ai-log-${index}`} style={{ border: "1px solid rgba(255,255,255,0.08)", borderRadius: "8px", padding: "8px", background: "rgba(0,0,0,0.18)" }}>
                          <div style={{ fontSize: "11px", color: "#ddd", marginBottom: "6px" }}>
                            #{index + 1} • {log.content_type} • {log.topic_name || "n/a"} • {log.model || "default model"}
                          </div>

                          <div style={{ fontSize: "11px", color: "#9ad0ff", marginBottom: "4px" }}>Request sent to AI:</div>
                          <pre style={{ margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: "11px", color: "#d8f0ff", background: "rgba(24,35,49,0.75)", borderRadius: "6px", padding: "8px", maxHeight: "180px", overflowY: "auto" }}>
                            {log.request || "(empty request)"}
                          </pre>

                          <div style={{ fontSize: "11px", color: "#9af5b6", margin: "8px 0 4px" }}>Response received from AI:</div>
                          <pre style={{ margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: "11px", color: "#d6ffe2", background: "rgba(21,44,30,0.7)", borderRadius: "6px", padding: "8px", maxHeight: "180px", overflowY: "auto" }}>
                            {log.response || "(no response captured)"}
                          </pre>

                          {log.error && (
                            <div style={{ marginTop: "6px", fontSize: "11px", color: "#ff9aa2" }}>
                              Error: {log.error}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
