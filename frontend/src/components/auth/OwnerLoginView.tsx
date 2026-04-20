"use client";

import { FormEvent, useEffect, useState } from "react";
import { AlertCircle, Lock, LogIn, Shield } from "lucide-react";

interface OwnerLoginViewProps {
  ownerUsername: string;
  loading: boolean;
  error: string | null;
  publicBaseUrl?: string;
  onLogin: (username: string, password: string) => Promise<void>;
}

export default function OwnerLoginView({
  ownerUsername,
  loading,
  error,
  publicBaseUrl,
  onLogin,
}: OwnerLoginViewProps) {
  const [username, setUsername] = useState(ownerUsername);
  const [password, setPassword] = useState("");

  useEffect(() => {
    setUsername(ownerUsername);
  }, [ownerUsername]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    await onLogin(username, password);
    setPassword("");
  };

  return (
    <div
      className="flex-1 flex items-center justify-center px-4"
      style={{ background: "var(--bg-primary)" }}
    >
      <div className="w-full max-w-md animate-fadeIn">
        <div
          className="rounded-3xl p-8"
          style={{
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-color)",
            boxShadow: "0 24px 60px rgba(0, 0, 0, 0.22)",
          }}
        >
          <div className="flex items-center gap-3 mb-6">
            <div
              className="w-12 h-12 rounded-2xl flex items-center justify-center"
              style={{
                background: "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
                boxShadow: "0 10px 24px rgba(99, 102, 241, 0.25)",
              }}
            >
              <Shield size={22} color="white" />
            </div>
            <div>
              <h1 className="text-xl font-bold gradient-text">Private SoloLLM</h1>
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                Owner access is required before the app will call the backend.
              </p>
            </div>
          </div>

          {error && (
            <div
              className="mb-5 px-4 py-3 rounded-2xl flex items-start gap-2 text-sm"
              style={{
                background: "rgba(248, 113, 113, 0.08)",
                border: "1px solid rgba(248, 113, 113, 0.16)",
                color: "var(--error)",
              }}
            >
              <AlertCircle size={16} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <form className="space-y-4" onSubmit={handleSubmit}>
            <div>
              <label className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                Username
              </label>
              <input
                type="text"
                autoComplete="username"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                className="w-full mt-2 rounded-2xl px-4 py-3 text-sm outline-none transition-smooth"
                style={{
                  background: "var(--bg-primary)",
                  border: "1px solid var(--border-color)",
                  color: "var(--text-primary)",
                }}
                spellCheck={false}
              />
            </div>

            <div>
              <label className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                Password
              </label>
              <div className="relative mt-2">
                <Lock
                  size={15}
                  className="absolute left-4 top-1/2 -translate-y-1/2"
                  style={{ color: "var(--text-muted)" }}
                />
                <input
                  type="password"
                  autoComplete="current-password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  className="w-full rounded-2xl pl-11 pr-4 py-3 text-sm outline-none transition-smooth"
                  style={{
                    background: "var(--bg-primary)",
                    border: "1px solid var(--border-color)",
                    color: "var(--text-primary)",
                  }}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading || !password.trim()}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-2xl text-sm font-medium transition-smooth disabled:opacity-50"
              style={{
                background: "linear-gradient(135deg, var(--gradient-start), var(--gradient-end))",
                color: "white",
                boxShadow: "0 12px 28px rgba(99, 102, 241, 0.28)",
              }}
            >
              <LogIn size={15} />
              {loading ? "Signing in..." : "Unlock SoloLLM"}
            </button>
          </form>

          <div
            className="mt-5 px-4 py-3 rounded-2xl text-xs"
            style={{
              background: "rgba(99, 102, 241, 0.08)",
              border: "1px solid rgba(99, 102, 241, 0.14)",
              color: "var(--text-secondary)",
            }}
          >
            Backend: {publicBaseUrl || "Configured via NEXT_PUBLIC_API_BASE_URL"}
          </div>
        </div>
      </div>
    </div>
  );
}