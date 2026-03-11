"use client";

import { AlertTriangle, ArrowRight, Zap } from "lucide-react";

interface ContinuationBannerProps {
  reason: string;
  confidence: number;
  onContinue: () => void;
  isLoading: boolean;
}

export default function ContinuationBanner({
  reason,
  confidence,
  onContinue,
  isLoading,
}: ContinuationBannerProps) {
  const reasonLabels: Record<string, string> = {
    token_limit: "Response hit the token limit",
    incomplete_sentence: "Response ended mid-sentence",
    incomplete_code: "Code block was not completed",
    incomplete_section: "Section was cut off",
    near_limit: "Response used most of the token budget",
  };

  return (
    <div
      className="mx-4 mb-2 rounded-xl px-4 py-3.5 flex items-center gap-3 animate-slideDown"
      style={{
        background:
          "linear-gradient(135deg, rgba(99, 102, 241, 0.06), rgba(139, 92, 246, 0.04))",
        border: "1px solid rgba(99, 102, 241, 0.2)",
        boxShadow: "0 2px 16px rgba(99, 102, 241, 0.08)",
      }}
    >
      <div
        className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
        style={{
          background: "rgba(251, 191, 36, 0.1)",
        }}
      >
        <AlertTriangle size={16} style={{ color: "var(--warning)" }} />
      </div>

      <div className="flex-1 min-w-0">
        <p
          className="text-sm font-medium"
          style={{ color: "var(--text-primary)" }}
        >
          {reasonLabels[reason] || "Response may be incomplete"}
        </p>
        <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
          {Math.round(confidence * 100)}% confidence · Click to resume from
          where it stopped
        </p>
      </div>

      <button
        onClick={onContinue}
        disabled={isLoading}
        className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-medium transition-smooth disabled:opacity-50 shrink-0"
        style={{
          background:
            "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
          color: "white",
          boxShadow: "0 2px 8px rgba(99, 102, 241, 0.25)",
        }}
        onMouseEnter={(e) => {
          if (!isLoading) {
            e.currentTarget.style.boxShadow =
              "0 4px 16px rgba(99, 102, 241, 0.4)";
            e.currentTarget.style.transform = "translateY(-1px)";
          }
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.boxShadow =
            "0 2px 8px rgba(99, 102, 241, 0.25)";
          e.currentTarget.style.transform = "translateY(0)";
        }}
      >
        {isLoading ? (
          <>
            <Zap size={14} className="animate-pulse" />
            Continuing...
          </>
        ) : (
          <>
            <ArrowRight size={14} />
            Continue
          </>
        )}
      </button>
    </div>
  );
}
