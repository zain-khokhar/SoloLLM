"use client";

import React, { useEffect, useState, useCallback } from "react";
import {
  academicListCourses, academicListReviews, academicUploadReviews,
  academicAddManualReview, academicReprocessReviews,
} from "@/lib/api";
import type { AcademicCourse, AcademicReview } from "@/types/academic";

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

const INPUT_STYLE: React.CSSProperties = {
  padding: "10px 14px", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.1)",
  background: "rgba(255,255,255,0.04)", color: "#e0e0e0", fontSize: "14px",
  width: "100%", outline: "none",
};

export default function ReviewsCenter() {
  const [courses, setCourses] = useState<AcademicCourse[]>([]);
  const [selectedCourse, setSelectedCourse] = useState<string>("");
  const [reviews, setReviews] = useState<AcademicReview[]>([]);
  const [loading, setLoading] = useState(false);
  const [manualText, setManualText] = useState("");
  const [manualSemester, setManualSemester] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const data = await academicListCourses();
        setCourses(data.courses);
        if (data.courses.length > 0) setSelectedCourse(data.courses[0].code);
      } catch (e) { console.error(e); }
    })();
  }, []);

  const loadReviews = useCallback(async () => {
    if (!selectedCourse) return;
    setLoading(true);
    try {
      const data = await academicListReviews(selectedCourse, false, 100);
      setReviews(data.reviews);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, [selectedCourse]);

  useEffect(() => { loadReviews(); }, [loadReviews]);

  const handleFileUpload = async () => {
    if (!selectedCourse) return;
    const input = document.createElement("input");
    input.type = "file"; input.accept = ".csv,.json,.txt";
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      try {
        const result = await academicUploadReviews(file, selectedCourse);
        setMessage(`✅ Added ${result.total_added} reviews from ${result.filename}`);
        loadReviews();
      } catch (err: any) {
        setMessage(`❌ ${err.message}`);
      }
    };
    input.click();
  };

  const handleManualAdd = async () => {
    if (!selectedCourse || manualText.length < 10) return;
    try {
      await academicAddManualReview(selectedCourse, manualText, manualSemester);
      setManualText(""); setManualSemester("");
      setMessage("✅ Review added");
      loadReviews();
    } catch (err: any) {
      setMessage(`❌ ${err.message}`);
    }
  };

  const handleReprocess = async () => {
    if (!selectedCourse) return;
    try {
      const result = await academicReprocessReviews(selectedCourse);
      setMessage(`✅ Scored ${result.topics_scored} topics`);
    } catch (err: any) {
      setMessage(`❌ ${err.message}`);
    }
  };

  return (
    <div>
      <h2 style={{ fontSize: "20px", fontWeight: 600, marginBottom: "20px" }}>
        💬 Reviews Center
      </h2>

      {/* Course Selector */}
      <div style={{ ...CARD_STYLE, marginBottom: "20px", display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
        <label style={{ fontWeight: 500, fontSize: "14px" }}>Course:</label>
        <input
          list="course-list"
          value={selectedCourse}
          onChange={(e) => setSelectedCourse(e.target.value.toUpperCase())}
          placeholder="Enter Course Code (e.g. CS101)"
          style={{ ...INPUT_STYLE, width: "220px" }}
        />
        <datalist id="course-list">
          {courses.map((c) => (
            <option key={c.code} value={c.code}>
              {c.title || "Untitled"}
            </option>
          ))}
        </datalist>
        <button style={BTN_STYLE} onClick={handleFileUpload}>📎 Upload CSV/JSON</button>
        <button style={{ ...BTN_STYLE, background: "rgba(255,255,255,0.08)" }} onClick={handleReprocess}>
          🔄 Reprocess Features
        </button>
      </div>

      {message && (
        <div style={{ padding: "10px 14px", borderRadius: "8px", marginBottom: "16px",
          background: message.startsWith("✅") ? "rgba(67,233,123,0.1)" : "rgba(245,87,108,0.1)",
          fontSize: "13px",
        }}>{message}</div>
      )}

      {/* Manual Review Entry */}
      <div style={{ ...CARD_STYLE, marginBottom: "20px" }}>
        <h3 style={{ fontSize: "15px", fontWeight: 600, marginBottom: "12px" }}>Add Manual Review</h3>
        <textarea
          value={manualText}
          onChange={(e) => setManualText(e.target.value)}
          placeholder="Write a student review here... (min 10 chars)"
          style={{ ...INPUT_STYLE, height: "80px", resize: "vertical", marginBottom: "8px" }}
        />
        <div style={{ display: "flex", gap: "8px" }}>
          <input
            value={manualSemester}
            onChange={(e) => setManualSemester(e.target.value)}
            placeholder="Semester (optional)"
            style={{ ...INPUT_STYLE, width: "200px" }}
          />
          <button style={BTN_STYLE} onClick={handleManualAdd} disabled={manualText.length < 10}>
            ➕ Add
          </button>
        </div>
      </div>

      {/* Reviews List */}
      <div style={CARD_STYLE}>
        <h3 style={{ fontSize: "15px", fontWeight: 600, marginBottom: "12px" }}>
          Reviews ({reviews.length})
        </h3>
        {loading ? (
          <div style={{ color: "#888", padding: "20px", textAlign: "center" }}>Loading...</div>
        ) : reviews.length === 0 ? (
          <div style={{ color: "#888", padding: "20px", textAlign: "center" }}>
            No reviews yet. Upload a CSV or add manually above.
          </div>
        ) : (
          <div style={{ maxHeight: "500px", overflowY: "auto" }}>
            {reviews.map((review) => (
              <div key={review.id} style={{
                padding: "12px 16px", borderRadius: "8px", marginBottom: "8px",
                background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)",
              }}>
                <p style={{ margin: "0 0 8px", fontSize: "14px", lineHeight: 1.5 }}>
                  {review.review_text}
                </p>
                <div style={{ display: "flex", gap: "16px", fontSize: "11px", color: "#888" }}>
                  <span>🔥 Urgency: {(review.urgency_score * 100).toFixed(0)}%</span>
                  <span>💡 Sentiment: {(review.sentiment_score * 100).toFixed(0)}%</span>
                  {review.semester && <span>📅 {review.semester}</span>}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
