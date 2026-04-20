import {
  Conversation,
  Message,
  ModelInfo,
  ModelDetailInfo,
  AppSettings,
  SystemProfile,
  DistillationMeta,
  DistillationSettings,
  GraphStats,
  GraphVisualization,
  GraphAnalysis,
  EntityNeighbors,
  TimelineEntity,
  ScrapeResult,
  ScrapePreview,
  GraphNode,
  AgentTool,
  AgentMemory,
  AgentRun,
  DashboardSummary,
  ImportResult,
  RuntimeStatus,
  AuthConfig,
  AuthSession,
  BackendCapabilities,
  ModelCatalogResponse,
  Thread,
  ThreadSettings,
  ThreadContextPage,
  DocumentInfo,
  Citation,
  RAGStats,
  TrainingStatus,
  TrainingDataPreview,
  TrainingCapabilities,
  FinetunedModel,
} from "@/types";

function normalizeBaseUrl(value: string): string {
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

function deriveOpenAICompatBaseUrl(apiBase: string): string {
  const normalized = normalizeBaseUrl(apiBase);
  if (normalized.endsWith("/api")) {
    return `${normalized.slice(0, -4)}/v1`;
  }
  return normalized;
}

export const API_BASE = normalizeBaseUrl(
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000/api"
);
export const OPENAI_COMPAT_BASE_URL = normalizeBaseUrl(
  process.env.NEXT_PUBLIC_OPENAI_COMPAT_BASE_URL ||
    deriveOpenAICompatBaseUrl(API_BASE)
);
const AUTH_STORAGE_KEY = "solollm.ownerSessionToken";
const AUTH_REQUIRED_EVENT = "solollm-auth-required";

const UPLOAD_API_BASE = API_BASE;

export class UnauthorizedError extends Error {
  status: number;

  constructor(message: string = "Authentication required") {
    super(message);
    this.name = "UnauthorizedError";
    this.status = 401;
  }
}

function isBrowser(): boolean {
  return typeof window !== "undefined";
}

function getStoredAuthToken(): string {
  if (!isBrowser()) {
    return "";
  }
  return window.localStorage.getItem(AUTH_STORAGE_KEY) || "";
}

export function setStoredAuthToken(token: string | null): void {
  if (!isBrowser()) {
    return;
  }
  if (token) {
    window.localStorage.setItem(AUTH_STORAGE_KEY, token);
  } else {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
  }
}

export function clearStoredAuthToken(): void {
  setStoredAuthToken(null);
}

export function addAuthRequiredListener(listener: () => void): () => void {
  if (!isBrowser()) {
    return () => {};
  }

  const handler = () => listener();
  window.addEventListener(AUTH_REQUIRED_EVENT, handler);
  return () => window.removeEventListener(AUTH_REQUIRED_EVENT, handler);
}

function notifyAuthRequired(): void {
  clearStoredAuthToken();
  if (isBrowser()) {
    window.dispatchEvent(new CustomEvent(AUTH_REQUIRED_EVENT));
  }
}

function buildAuthHeaders(
  headers?: HeadersInit,
  includeJsonContentType: boolean = false
): Headers {
  const merged = new Headers(headers);
  if (includeJsonContentType && !merged.has("Content-Type")) {
    merged.set("Content-Type", "application/json");
  }

  const token = getStoredAuthToken();
  if (token && !merged.has("Authorization")) {
    merged.set("Authorization", `Bearer ${token}`);
  }

  return merged;
}

async function readErrorMessage(
  res: Response,
  fallback: string
): Promise<string> {
  const error = await res.json().catch(() => ({ detail: fallback }));
  return error.detail || fallback;
}

async function fetchWithAuth(
  input: string,
  options?: RequestInit,
  includeJsonContentType: boolean = false
): Promise<Response> {
  const res = await fetch(input, {
    ...options,
    headers: buildAuthHeaders(options?.headers, includeJsonContentType),
  });

  if (res.status === 401) {
    const detail = await readErrorMessage(res, "Authentication required");
    notifyAuthRequired();
    throw new UnauthorizedError(detail);
  }

  return res;
}

function buildAuthorizedUrl(url: string): string {
  const token = getStoredAuthToken();
  if (!token || !isBrowser()) {
    return url;
  }

  const resolved = new URL(url, window.location.origin);
  resolved.searchParams.set("access_token", token);
  return resolved.toString();
}

// ── REST helpers ───────────────────────────────────────────

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetchWithAuth(`${API_BASE}${url}`, options, true);
  if (!res.ok) {
    throw new Error(await readErrorMessage(res, `Request failed: ${res.status}`));
  }
  return res.json();
}

export async function getAuthConfig(): Promise<AuthConfig> {
  const res = await fetch(`${API_BASE}/auth/config`);
  if (!res.ok) {
    throw new Error(await readErrorMessage(res, `Request failed: ${res.status}`));
  }
  return res.json();
}

export async function getAuthSession(): Promise<AuthSession> {
  const res = await fetchWithAuth(`${API_BASE}/auth/session`);
  if (!res.ok) {
    throw new Error(await readErrorMessage(res, `Request failed: ${res.status}`));
  }
  return res.json();
}

export async function loginOwner(
  username: string,
  password: string
): Promise<AuthSession & { token: string }> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    throw new Error(await readErrorMessage(res, "Login failed"));
  }
  const data = (await res.json()) as AuthSession & { token: string };
  if (data.token) {
    setStoredAuthToken(data.token);
  }
  return data;
}

export async function logoutOwner(): Promise<void> {
  try {
    await fetchWithAuth(`${API_BASE}/auth/logout`, { method: "POST" });
  } catch {
    // Ignore logout transport failures; local token cleanup still matters.
  }
  clearStoredAuthToken();
}

export async function getBackendCapabilities(): Promise<BackendCapabilities> {
  return fetchJSON("/capabilities");
}

// ── Health ─────────────────────────────────────────────────

export async function checkHealth(): Promise<{
  status: string;
  ollama_connected: boolean;
}> {
  return fetchJSON("/health");
}

// ── Models ─────────────────────────────────────────────────

export async function listModels(): Promise<ModelInfo[]> {
  const data = await fetchJSON<{ models: ModelInfo[] }>("/models");
  return data.models;
}

export async function deleteModel(name: string): Promise<void> {
  await fetchJSON(`/models/${encodeURIComponent(name)}`, { method: "DELETE" });
}

export async function getModelInfo(modelName: string): Promise<ModelDetailInfo> {
  const data = await fetchJSON<{ info: ModelDetailInfo }>(
    `/models/${encodeURIComponent(modelName)}/info`
  );
  return data.info;
}

// ── Conversations ──────────────────────────────────────────

export async function listConversations(): Promise<Conversation[]> {
  const data = await fetchJSON<{ conversations: Conversation[] }>(
    "/conversations"
  );
  return data.conversations;
}

export async function getConversation(
  id: string
): Promise<{ conversation: Conversation; messages: Message[]; threads: Thread[]; default_thread_id: string | null }> {
  return fetchJSON(`/conversations/${id}`);
}

export async function updateConversation(
  id: string,
  data: Partial<Pick<Conversation, "title" | "model" | "system_prompt">>
): Promise<Conversation> {
  const res = await fetchJSON<{ conversation: Conversation }>(
    `/conversations/${id}`,
    {
      method: "PUT",
      body: JSON.stringify(data),
    }
  );
  return res.conversation;
}

export async function deleteConversation(id: string): Promise<void> {
  await fetchJSON(`/conversations/${id}`, { method: "DELETE" });
}

// ── Settings ───────────────────────────────────────────────

export async function getSettings(): Promise<AppSettings> {
  const data = await fetchJSON<{ settings: AppSettings }>("/settings");
  return data.settings;
}

export async function updateSettings(
  settings: Partial<AppSettings>
): Promise<AppSettings> {
  const data = await fetchJSON<{ settings: AppSettings }>("/settings", {
    method: "PUT",
    body: JSON.stringify(settings),
  });
  return data.settings;
}

// ── System ─────────────────────────────────────────────────

export async function getSystemProfile(): Promise<SystemProfile> {
  const data = await fetchJSON<{ profile: SystemProfile }>("/system/profile");
  return data.profile;
}

export async function runProfiler(): Promise<SystemProfile> {
  const data = await fetchJSON<{ profile: SystemProfile }>("/system/profile", {
    method: "POST",
  });
  return data.profile;
}

// ── Runtime / Ollama Manager ──────────────────────────────

export async function getRuntimeStatus(): Promise<RuntimeStatus> {
  return fetchJSON("/runtime/status");
}

export async function setupRuntime(): Promise<{ success: boolean; ollama: Record<string, unknown> }> {
  return fetchJSON("/runtime/setup", { method: "POST" });
}

export interface SetupProgress {
  stage: "idle" | "downloading" | "extracting" | "starting" | "ready" | "error";
  downloaded_bytes: number;
  total_bytes: number;
  percent: number;
  error: string | null;
}

export function streamSetupProgress(
  callbacks: {
    onProgress: (data: SetupProgress) => void;
    onError: (error: string) => void;
  }
): AbortController {
  const controller = new AbortController();

  fetchWithAuth(`${API_BASE}/runtime/setup/progress`, {
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        callbacks.onError(`Request failed: ${res.status}`);
        return;
      }
      const reader = res.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data:")) {
            const dataStr = line.slice(5).trim();
            if (!dataStr) continue;
            try {
              callbacks.onProgress(JSON.parse(dataStr));
            } catch {
              // skip
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        callbacks.onError(err.message);
      }
    });

  return controller;
}

export async function restartRuntime(): Promise<{ success: boolean }> {
  return fetchJSON("/runtime/restart", { method: "POST" });
}

export async function getModelCatalog(): Promise<ModelCatalogResponse> {
  return fetchJSON("/runtime/models/catalog");
}

export function streamModelPull(
  modelName: string,
  callbacks: {
    onProgress: (data: { status: string; progress: number; total: number; completed: number }) => void;
    onDone: (model: string) => void;
    onError: (error: string) => void;
  }
): AbortController {
  const controller = new AbortController();

  fetchWithAuth(
    `${API_BASE}/models/pull`,
    {
      method: "POST",
      body: JSON.stringify({ name: modelName }),
      signal: controller.signal,
    },
    true
  )
    .then(async (res) => {
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        callbacks.onError(err.detail || `Request failed: ${res.status}`);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        callbacks.onError("No response body");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let eventType = "";
        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            const dataStr = line.slice(5).trim();
            if (!dataStr) continue;
            try {
              const data = JSON.parse(dataStr);
              if (eventType === "progress") {
                callbacks.onProgress(data);
              } else if (eventType === "done") {
                callbacks.onDone(data.model);
              } else if (eventType === "error") {
                callbacks.onError(data.error);
              }
            } catch {
              // skip
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        callbacks.onError(err.message);
      }
    });

  return controller;
}

// ── SSE Streaming Chat ─────────────────────────────────────

export interface ChatStreamCallbacks {
  onToken: (content: string) => void;
  onDone: (data: {
    message_id: string;
    conversation_id: string;
    thread_id: string | null;
    tokens_used: number;
  }) => void;
  onTruncated: (data: {
    message_id: string;
    conversation_id: string;
    thread_id: string | null;
    tokens_used: number;
    reason: string;
    confidence: number;
  }) => void;
  onError: (error: string) => void;
  onDistillation?: (data: DistillationMeta) => void;
}

export function streamChat(
  params: {
    message: string;
    conversation_id?: string;
    thread_id?: string;
    model?: string;
    system_prompt?: string;
    temperature?: number;
    max_tokens?: number;
    documents_only?: boolean;
  },
  callbacks: ChatStreamCallbacks
): AbortController {
  const controller = new AbortController();

  fetchWithAuth(
    `${API_BASE}/chat`,
    {
      method: "POST",
      body: JSON.stringify(params),
      signal: controller.signal,
    },
    true
  )
    .then(async (res) => {
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        callbacks.onError(err.detail || `Request failed: ${res.status}`);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        callbacks.onError("No response body");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let eventType = "";
        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            const dataStr = line.slice(5).trim();
            if (!dataStr) continue;
            try {
              const data = JSON.parse(dataStr);
              switch (eventType) {
                case "token":
                  callbacks.onToken(data.content);
                  break;
                case "done":
                  callbacks.onDone(data);
                  break;
                case "truncated":
                  callbacks.onTruncated(data);
                  break;
                case "distillation":
                  callbacks.onDistillation?.(data);
                  break;
                case "error":
                  callbacks.onError(data.error);
                  break;
              }
            } catch {
              // Skip malformed JSON
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        callbacks.onError(err.message);
      }
    });

  return controller;
}

export function streamContinuation(
  params: { conversation_id: string; message_id: string },
  callbacks: ChatStreamCallbacks
): AbortController {
  const controller = new AbortController();

  fetchWithAuth(
    `${API_BASE}/chat/continue`,
    {
      method: "POST",
      body: JSON.stringify(params),
      signal: controller.signal,
    },
    true
  )
    .then(async (res) => {
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        callbacks.onError(err.detail || `Request failed: ${res.status}`);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        callbacks.onError("No response body");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let eventType = "";
        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            const dataStr = line.slice(5).trim();
            if (!dataStr) continue;
            try {
              const data = JSON.parse(dataStr);
              switch (eventType) {
                case "token":
                  callbacks.onToken(data.content);
                  break;
                case "done":
                  callbacks.onDone(data);
                  break;
                case "truncated":
                  callbacks.onTruncated(data);
                  break;
                case "error":
                  callbacks.onError(data.error);
                  break;
              }
            } catch {
              // Skip malformed JSON
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        callbacks.onError(err.message);
      }
    });

  return controller;
}

// ── Web Search Streaming ──────────────────────────────────────

export interface WebSearchCallbacks {
  onStatus: (phase: string, message: string) => void;
  onSources: (results: { title: string; snippet: string; url: string }[]) => void;
  onToken: (content: string) => void;
  onDone: (data: { message_id: string; conversation_id: string; thread_id: string | null }) => void;
  onError: (error: string) => void;
}

export function streamWebSearch(
  params: {
    query: string;
    model?: string;
    conversation_id?: string;
    thread_id?: string;
    num_results?: number;
  },
  callbacks: WebSearchCallbacks
): AbortController {
  const controller = new AbortController();

  fetchWithAuth(
    `${API_BASE}/chat/web-search`,
    {
      method: "POST",
      body: JSON.stringify(params),
      signal: controller.signal,
    },
    true
  )
    .then(async (res) => {
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        callbacks.onError(err.detail || `Request failed: ${res.status}`);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        callbacks.onError("No response body");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let eventType = "";
        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            const dataStr = line.slice(5).trim();
            if (!dataStr) continue;
            try {
              const data = JSON.parse(dataStr);
              switch (eventType) {
                case "status":
                  callbacks.onStatus(data.phase, data.message);
                  break;
                case "sources":
                  callbacks.onSources(data.results);
                  break;
                case "token":
                  callbacks.onToken(data.content);
                  break;
                case "done":
                  callbacks.onDone(data);
                  break;
                case "error":
                  callbacks.onError(data.content || data.error || "Unknown error");
                  break;
              }
            } catch {
              // Skip malformed JSON
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        callbacks.onError(err.message);
      }
    });

  return controller;
}

// ── Distillation API (Phase 3) ─────────────────────────────

export async function getDistillationSettings(): Promise<DistillationSettings> {
  const data = await fetchJSON<{ settings: DistillationSettings }>(
    "/distillation/settings"
  );
  return data.settings;
}

export async function updateDistillationSettings(
  update: Partial<DistillationSettings>
): Promise<DistillationSettings> {
  const data = await fetchJSON<{ settings: DistillationSettings }>(
    "/distillation/settings",
    {
      method: "PUT",
      body: JSON.stringify(update),
    }
  );
  return data.settings;
}

export async function getDistillationMetrics(
  conversationId?: string
): Promise<{
  metrics: Array<Record<string, unknown>>;
  summary: {
    total_queries: number;
    avg_confidence: number;
    avg_compression_ratio: number;
    total_verified: number;
    query_type_distribution: Record<string, number>;
  };
}> {
  const url = conversationId
    ? `/distillation/metrics?conversation_id=${conversationId}`
    : "/distillation/metrics";
  return fetchJSON(url);
}

export async function runChainOfDensity(
  content: string,
  iterations: number = 2,
  model?: string
): Promise<{ summary: string; iterations: number }> {
  return fetchJSON("/distillation/chain-of-density", {
    method: "POST",
    body: JSON.stringify({ content, iterations, model }),
  });
}

export async function verifyResponse(
  response: string,
  context: string,
  query: string,
  model?: string
): Promise<{
  verified: boolean;
  corrected_response: string | null;
  feedback: string;
}> {
  return fetchJSON("/distillation/verify", {
    method: "POST",
    body: JSON.stringify({ response, context, query, model }),
  });
}

// ── Knowledge Graph API (Phase 4) ─────────────────────────

export async function getGraphStats(): Promise<GraphStats> {
  return fetchJSON("/graph/stats");
}

export async function searchGraphEntities(
  query: string,
  limit: number = 20
): Promise<{ entities: GraphNode[]; count: number }> {
  return fetchJSON("/graph/search", {
    method: "POST",
    body: JSON.stringify({ query, limit }),
  });
}

export async function getEntityNeighbors(
  entityId: string
): Promise<EntityNeighbors> {
  return fetchJSON(`/graph/entity/${encodeURIComponent(entityId)}`);
}

export async function deleteEntity(entityId: string): Promise<void> {
  await fetchJSON(`/graph/entity/${encodeURIComponent(entityId)}`, {
    method: "DELETE",
  });
}

export async function getGraphVisualization(
  limit: number = 200
): Promise<GraphVisualization> {
  return fetchJSON(`/graph/visualization?limit=${limit}`);
}

export async function getGraphAnalysis(): Promise<GraphAnalysis> {
  return fetchJSON("/graph/analysis");
}

export async function getEntityTimeline(
  limit: number = 50
): Promise<{ entities: TimelineEntity[]; count: number }> {
  return fetchJSON(`/graph/timeline?limit=${limit}`);
}

export async function clearGraph(): Promise<void> {
  await fetchJSON("/graph/clear", { method: "DELETE" });
}

// ── Web Scraping API (Phase 4) ─────────────────────────────

export async function scrapeUrl(
  url: string,
  workspaceId: string = "default"
): Promise<ScrapeResult> {
  return fetchJSON("/graph/scrape", {
    method: "POST",
    body: JSON.stringify({ url, workspace_id: workspaceId }),
  });
}

export async function scrapePreview(
  url: string
): Promise<ScrapePreview> {
  return fetchJSON("/graph/scrape/preview", {
    method: "POST",
    body: JSON.stringify({ url }),
  });
}

// ── Agent API (Phase 5) ───────────────────────────────────

export async function listAgentTools(): Promise<AgentTool[]> {
  const data = await fetchJSON<{ tools: AgentTool[] }>("/agent/tools");
  return data.tools;
}

export async function executeAgentTool(
  toolName: string,
  args: Record<string, unknown>
): Promise<{
  tool_name: string;
  success: boolean;
  output: string;
  error: string | null;
}> {
  return fetchJSON("/agent/execute", {
    method: "POST",
    body: JSON.stringify({ tool_name: toolName, arguments: args }),
  });
}

export interface AgentStreamCallbacks {
  onThought: (step: number, content: string) => void;
  onAction: (step: number, tool: string, input: Record<string, unknown>) => void;
  onObservation: (step: number, content: string) => void;
  onAnswer: (content: string, totalSteps: number, toolsUsed: string[]) => void;
  onError: (error: string) => void;
  onThinking?: (step: number, content: string) => void;
}

export function streamAgentRun(
  params: { query: string; model?: string; max_steps?: number; reasoning_model?: string },
  callbacks: AgentStreamCallbacks
): AbortController {
  const controller = new AbortController();

  fetchWithAuth(
    `${API_BASE}/agent/run/stream`,
    {
      method: "POST",
      body: JSON.stringify(params),
      signal: controller.signal,
    },
    true
  )
    .then(async (res) => {
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        callbacks.onError(err.detail || `Request failed: ${res.status}`);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        callbacks.onError("No response body");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let eventType = "";
        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            const dataStr = line.slice(5).trim();
            if (!dataStr) continue;
            try {
              const data = JSON.parse(dataStr);
              switch (eventType) {
                case "thinking":
                  callbacks.onThinking?.(data.step, data.content);
                  break;
                case "thought":
                  callbacks.onThought(data.step, data.content);
                  break;
                case "action":
                  callbacks.onAction(data.step, data.tool, data.input);
                  break;
                case "observation":
                  callbacks.onObservation(data.step, data.content);
                  break;
                case "answer":
                  callbacks.onAnswer(
                    data.content,
                    data.total_steps,
                    data.tools_used
                  );
                  break;
                case "error":
                  callbacks.onError(data.content);
                  break;
              }
            } catch {
              // Skip malformed JSON
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        callbacks.onError(err.message);
      }
    });

  return controller;
}

export async function runAgent(
  query: string,
  model?: string,
  maxSteps?: number
): Promise<{
  answer: string;
  success: boolean;
  error: string | null;
  total_steps: number;
  tools_used: string[];
  steps: Array<Record<string, unknown>>;
}> {
  return fetchJSON("/agent/run", {
    method: "POST",
    body: JSON.stringify({ query, model, max_steps: maxSteps }),
  });
}

export async function getAgentRuns(
  limit: number = 20
): Promise<{ runs: AgentRun[] }> {
  return fetchJSON(`/agent/runs?limit=${limit}`);
}

export async function getAgentMemories(
  category?: string
): Promise<{ memories: AgentMemory[]; count: number }> {
  const url = category
    ? `/agent/memory?category=${encodeURIComponent(category)}`
    : "/agent/memory";
  return fetchJSON(url);
}

export async function addAgentMemory(
  content: string,
  category: string = "general"
): Promise<{ memory: AgentMemory }> {
  return fetchJSON("/agent/memory", {
    method: "POST",
    body: JSON.stringify({ content, category }),
  });
}

export async function deleteAgentMemory(
  memoryId: string
): Promise<void> {
  await fetchJSON(`/agent/memory/${encodeURIComponent(memoryId)}`, {
    method: "DELETE",
  });
}

// ── Dashboard API (Phase 6) ──────────────────────────────

export async function getDashboardSummary(): Promise<DashboardSummary> {
  return fetchJSON("/dashboard/summary");
}

export async function getDashboardRecent(
  count: number = 20
): Promise<{ metrics: Array<Record<string, unknown>> }> {
  return fetchJSON(`/dashboard/recent?count=${count}`);
}

// ── Export / Import API (Phase 6) ─────────────────────────

export async function exportConversations(): Promise<void> {
  const res = await fetchWithAuth(`${API_BASE}/export/conversations`);
  if (!res.ok) throw new Error("Export failed");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `solollm_conversations_${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function importConversations(file: File): Promise<ImportResult> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetchWithAuth(`${API_BASE}/export/conversations/import`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Import failed");
  }
  return res.json();
}

export async function exportSettings(): Promise<void> {
  const res = await fetchWithAuth(`${API_BASE}/export/settings`);
  if (!res.ok) throw new Error("Export failed");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "solollm_settings.json";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function importSettings(file: File): Promise<{ success: boolean; imported_keys: string[] }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetchWithAuth(`${API_BASE}/export/settings/import`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Import failed");
  }
  return res.json();
}

export async function clearAgentMemories(): Promise<{ cleared: number }> {
  return fetchJSON("/agent/memory", { method: "DELETE" });
}

// ── Threads API ────────────────────────────────────────────

export async function listThreads(conversationId: string): Promise<Thread[]> {
  const data = await fetchJSON<{ threads: Thread[] }>(
    `/conversations/${conversationId}/threads`
  );
  return data.threads;
}

export async function createThread(
  conversationId: string,
  title: string = "New Thread",
  systemPrompt: string = "",
  contextMode: "isolated" | "shared" = "isolated"
): Promise<Thread> {
  const data = await fetchJSON<{ thread: Thread }>(
    `/conversations/${conversationId}/threads`,
    {
      method: "POST",
      body: JSON.stringify({
        title,
        system_prompt: systemPrompt,
        context_mode: contextMode,
      }),
    }
  );
  return data.thread;
}

export async function getThread(
  threadId: string
): Promise<{ thread: Thread; messages: Message[]; settings: ThreadSettings }> {
  return fetchJSON(`/threads/${threadId}`);
}

export async function updateThread(
  threadId: string,
  data: Partial<Pick<Thread, "title" | "system_prompt" | "context_mode">>
): Promise<Thread> {
  const res = await fetchJSON<{ thread: Thread }>(`/threads/${threadId}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
  return res.thread;
}

export async function deleteThread(threadId: string): Promise<void> {
  await fetchJSON(`/threads/${threadId}`, { method: "DELETE" });
}

export async function getThreadSettings(
  threadId: string
): Promise<ThreadSettings> {
  const data = await fetchJSON<{ settings: ThreadSettings }>(
    `/threads/${threadId}/settings`
  );
  return data.settings;
}

export async function updateThreadSettings(
  threadId: string,
  settings: Partial<ThreadSettings>
): Promise<ThreadSettings> {
  const data = await fetchJSON<{ settings: ThreadSettings }>(
    `/threads/${threadId}/settings`,
    {
      method: "PUT",
      body: JSON.stringify(settings),
    }
  );
  return data.settings;
}

export async function attachDocumentToThread(
  threadId: string,
  documentId: string
): Promise<void> {
  await fetchJSON(`/threads/${threadId}/documents/${documentId}`, {
    method: "POST",
  });
}

export async function getThreadDocuments(
  threadId: string
): Promise<{ documents: { id: string; thread_id: string; document_id: string; attached_at: string }[] }> {
  return fetchJSON(`/threads/${threadId}/documents`);
}

export async function detachDocumentFromThread(
  threadId: string,
  documentId: string
): Promise<void> {
  await fetchJSON(`/threads/${threadId}/documents/${documentId}`, {
    method: "DELETE",
  });
}

export async function getThreadContext(
  threadId: string,
  activeOnly: boolean = true
): Promise<{ pages: ThreadContextPage[]; thread_id: string }> {
  return fetchJSON(
    `/threads/${threadId}/context?active_only=${activeOnly}`
  );
}

// ── Documents API ──────────────────────────────────────────

export async function uploadDocument(
  file: File,
  workspaceId: string = "default"
): Promise<{ success: boolean; document_id: string; filename: string; title: string; file_type: string; chunk_count: number; page_count: number; content_length: number; workspace_id: string; errors?: string[] }> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("workspace_id", workspaceId);
  const res = await fetchWithAuth(`${UPLOAD_API_BASE}/documents/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export async function listDocuments(
  workspaceId: string = "default"
): Promise<DocumentInfo[]> {
  const data = await fetchJSON<{ documents: DocumentInfo[] }>(
    `/documents/list?workspace_id=${encodeURIComponent(workspaceId)}`
  );
  // Backend persists primary key as `id`; normalize for frontend contract (`document_id`).
  return data.documents.map((doc) => {
    const normalized = doc as DocumentInfo & { id?: string };
    return {
      ...normalized,
      document_id: normalized.document_id ?? normalized.id ?? "",
    };
  });
}

export async function deleteDocument(documentId: string): Promise<void> {
  await fetchJSON(`/documents/${documentId}`, { method: "DELETE" });
}

export async function queryDocuments(
  query: string,
  workspaceId: string = "default",
  topK: number = 5
): Promise<{ citations: Citation[]; context_text: string; source_count: number }> {
  return fetchJSON(`/documents/query`, {
    method: "POST",
    body: JSON.stringify({
      query,
      workspace_id: workspaceId,
      top_k: topK,
    }),
  });
}

export async function getRAGStats(
  workspaceId: string = "default"
): Promise<RAGStats> {
  return fetchJSON(`/documents/stats/${encodeURIComponent(workspaceId)}`);
}

// ── Training API ──────────────────────────────────────────

export async function startTraining(params: {
  model: string;
  output_name?: string;
  conversation_ids?: string[];
  document_ids?: string[];
  source_mode?: "conversation" | "documents" | "mixed";
  workspace_id?: string;
  lora_rank?: number;
  num_epochs?: number;
  learning_rate?: number;
  max_seq_length?: number;
  validation_split?: number;
  quality_loss_threshold?: number;
}): Promise<{
  status: string;
  examples: number;
  training_examples: number;
  validation_examples: number;
  base_model: string;
  source_mode: "conversation" | "documents" | "mixed";
  documents_used: string[];
}> {
  return fetchJSON("/training/start", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function getTrainingStatus(): Promise<TrainingStatus> {
  return fetchJSON("/training/status");
}

export async function getTrainingCapabilities(model: string): Promise<TrainingCapabilities> {
  const params = new URLSearchParams();
  params.set("model", model);
  return fetchJSON(`/training/capabilities?${params.toString()}`);
}

export async function cancelTraining(): Promise<{ status: string }> {
  return fetchJSON("/training/cancel", { method: "POST" });
}

export async function previewTrainingData(
  conversationIds?: string[],
  documentIds?: string[],
  sourceMode?: "conversation" | "documents" | "mixed",
  workspaceId: string = "default"
): Promise<TrainingDataPreview> {
  const params = new URLSearchParams();
  if (conversationIds?.length) {
    params.set("conversation_ids", conversationIds.join(","));
  }
  if (documentIds?.length) {
    params.set("document_ids", documentIds.join(","));
  }
  if (sourceMode) {
    params.set("source_mode", sourceMode);
  }
  params.set("workspace_id", workspaceId);
  const query = params.toString();
  const url = query ? `/training/data/preview?${query}` : "/training/data/preview";
  return fetchJSON(url);
}

// ── Fine-tuned Models API ──────────────────────────────────

export async function listFinetunedModels(): Promise<FinetunedModel[]> {
  const data = await fetchJSON<{ models: FinetunedModel[] }>("/training/models");
  return data.models;
}

export async function getFinetunedModel(name: string): Promise<FinetunedModel> {
  const data = await fetchJSON<{ model: FinetunedModel }>(
    `/training/models/${encodeURIComponent(name)}`
  );
  return data.model;
}

export async function registerFinetunedModel(name: string): Promise<{ status: string; model: string }> {
  return fetchJSON("/training/models/register", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export async function unregisterFinetunedModel(name: string): Promise<{ status: string; model: string }> {
  return fetchJSON("/training/models/unregister", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export async function deleteFinetunedModel(name: string): Promise<{ status: string; model: string }> {
  return fetchJSON(`/training/models/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

// ── Academic Auto-Generation API ─────────────────────────────

import type {
  AcademicCourse, AcademicReview, AcademicTopicScore, AcademicJob,
  AcademicOutput, AcademicFeedback, AcademicOverviewStats,
  CourseMetrics, ScoringWeights, BulkImportResult, ReviewIngestResult,
  AcademicGenerateOptions,
} from "@/types/academic";

export async function academicBulkImport(files: File[]): Promise<BulkImportResult> {
  const formData = new FormData();
  files.forEach((f) => formData.append("files", f));
  const res = await fetchWithAuth(`${UPLOAD_API_BASE}/academic/courses/bulk-import`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export async function academicUploadPdf(file: File, courseCode: string): Promise<any> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("course_code", courseCode);
  const res = await fetchWithAuth(`${UPLOAD_API_BASE}/academic/courses/upload-pdf`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export async function academicListCourses(): Promise<{ courses: AcademicCourse[]; total: number }> {
  return fetchJSON("/academic/courses");
}

export async function academicGetCourse(courseCode: string): Promise<any> {
  return fetchJSON(`/academic/courses/${encodeURIComponent(courseCode)}`);
}

export async function academicUploadReviews(file: File, courseCode: string): Promise<ReviewIngestResult> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("course_code", courseCode);
  const res = await fetchWithAuth(`${UPLOAD_API_BASE}/academic/reviews/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export async function academicAddManualReview(
  courseCode: string, reviewText: string, semester?: string
): Promise<any> {
  return fetchJSON("/academic/reviews/manual", {
    method: "POST",
    body: JSON.stringify({ course_code: courseCode, review_text: reviewText, semester: semester || "" }),
  });
}

export async function academicListReviews(
  courseCode: string, includeSpam?: boolean, limit?: number
): Promise<{ course_code: string; reviews: AcademicReview[]; total: number }> {
  const params = new URLSearchParams({ course_code: courseCode });
  if (includeSpam) params.set("include_spam", "true");
  if (limit) params.set("limit", limit.toString());
  return fetchJSON(`/academic/reviews?${params}`);
}

export async function academicReprocessReviews(courseCode: string): Promise<any> {
  return fetchJSON(`/academic/reviews/reprocess?course_code=${encodeURIComponent(courseCode)}`, {
    method: "POST",
  });
}

export async function academicGenerate(
  courseCode: string, outputTypes: string[], options?: AcademicGenerateOptions
): Promise<AcademicJob> {
  return fetchJSON("/academic/generate", {
    method: "POST",
    body: JSON.stringify({ course_code: courseCode, output_types: outputTypes, ...(options || {}) }),
  });
}

export async function academicBulkGenerate(
  courseCodes: string[], outputTypes: string[], options?: AcademicGenerateOptions
): Promise<{ jobs: any[]; total: number }> {
  return fetchJSON("/academic/generate/bulk", {
    method: "POST",
    body: JSON.stringify({ course_codes: courseCodes, output_types: outputTypes, ...(options || {}) }),
  });
}

export async function academicGetJob(jobId: string): Promise<AcademicJob> {
  return fetchJSON(`/academic/jobs/${jobId}`);
}

export async function academicListJobs(
  courseCode?: string, status?: string, limit?: number
): Promise<{ jobs: AcademicJob[]; total: number }> {
  const params = new URLSearchParams();
  if (courseCode) params.set("course_code", courseCode);
  if (status) params.set("status", status);
  if (limit) params.set("limit", limit.toString());
  return fetchJSON(`/academic/jobs?${params}`);
}

export async function academicRetryJob(jobId: string): Promise<any> {
  return fetchJSON(`/academic/jobs/${jobId}/retry`, { method: "POST" });
}

export async function academicPauseJob(jobId: string): Promise<any> {
  return fetchJSON(`/academic/jobs/${jobId}/pause`, { method: "POST" });
}

export async function academicResumeJob(jobId: string): Promise<any> {
  return fetchJSON(`/academic/jobs/${jobId}/resume`, { method: "POST" });
}

export async function academicStopJob(jobId: string): Promise<any> {
  return fetchJSON(`/academic/jobs/${jobId}/stop`, { method: "POST" });
}

export async function academicCancelJob(jobId: string): Promise<any> {
  return fetchJSON(`/academic/jobs/${jobId}/cancel`, { method: "POST" });
}

export async function academicDeleteJob(jobId: string): Promise<any> {
  return fetchJSON(`/academic/jobs/${jobId}`, { method: "DELETE" });
}

export async function academicGetJobPreview(jobId: string): Promise<any> {
  return fetchJSON(`/academic/jobs/${jobId}/preview`);
}

export function academicPreviewDownloadUrl(jobId: string): string {
  return buildAuthorizedUrl(`${UPLOAD_API_BASE}/academic/jobs/${jobId}/preview/download`);
}

export async function academicListOutputs(
  courseCode?: string, outputType?: string, limit?: number
): Promise<{ outputs: AcademicOutput[]; total: number }> {
  const params = new URLSearchParams();
  if (courseCode) params.set("course_code", courseCode);
  if (outputType) params.set("output_type", outputType);
  if (limit) params.set("limit", limit.toString());
  return fetchJSON(`/academic/outputs?${params}`);
}

export function academicDownloadUrl(outputId: string): string {
  return buildAuthorizedUrl(`${UPLOAD_API_BASE}/academic/outputs/${outputId}/download`);
}

export async function academicDeleteOutput(outputId: string): Promise<any> {
  return fetchJSON(`/academic/outputs/${outputId}`, { method: "DELETE" });
}

export async function academicListGenerationModels(): Promise<ModelInfo[]> {
  const data = await fetchJSON<{ models: ModelInfo[] }>("/academic/models");
  return data.models;
}

export async function academicSubmitFeedback(
  outputId: string, rating: number, comment?: string, topicAccuracy?: number
): Promise<any> {
  return fetchJSON(`/academic/outputs/${outputId}/feedback`, {
    method: "POST",
    body: JSON.stringify({ rating, comment: comment || "", topic_accuracy_pct: topicAccuracy }),
  });
}

export async function academicGetOverviewMetrics(): Promise<AcademicOverviewStats> {
  return fetchJSON("/academic/metrics/overview");
}

export async function academicGetCourseMetrics(courseCode: string): Promise<{
  course_code: string; course_id: string; metrics: CourseMetrics;
}> {
  return fetchJSON(`/academic/metrics/course/${encodeURIComponent(courseCode)}`);
}

export async function academicGetScoringWeights(): Promise<{ weights: ScoringWeights }> {
  return fetchJSON("/academic/scoring-weights");
}

export async function academicUpdateScoringWeights(weights: Partial<ScoringWeights>): Promise<{ weights: ScoringWeights }> {
  return fetchJSON("/academic/scoring-weights", {
    method: "PUT",
    body: JSON.stringify(weights),
  });
}

// ── Quantize API ─────────────────────────────────────────────

export interface QuantizeToolsStatus {
  ready: boolean;
  tools_path: string;
  has_quantize: boolean;
  has_convert: boolean;
}

export interface QuantType {
  description: string;
  size_ratio: number;
  quality_note: string;
  bits_per_weight: number;
}

export interface QuantizeJob {
  id: string;
  status: "pending" | "running" | "complete" | "error" | "cancelled";
  source_type: "local_gguf" | "huggingface";
  source: string;
  quant_level: string;
  output_name: string;
  import_to_ollama: boolean;
  progress: number;
  stage: string;
  message: string;
  error: string | null;
  output_path: string | null;
  output_size: number | null;
  created_at: string;
  completed_at: string | null;
}

export async function getQuantizeToolsStatus(): Promise<QuantizeToolsStatus> {
  return fetchJSON("/quantize/tools-status");
}

export async function getQuantTypes(): Promise<Record<string, QuantType>> {
  return fetchJSON("/quantize/quant-types");
}

export async function startQuantize(params: {
  source_type: "local_gguf" | "huggingface";
  source: string;
  quant_level: string;
  output_name: string;
  import_to_ollama: boolean;
}): Promise<{ job_id: string; job: QuantizeJob }> {
  return fetchJSON("/quantize/start", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function listQuantizeJobs(): Promise<QuantizeJob[]> {
  return fetchJSON("/quantize/jobs");
}

export async function getQuantizeJob(jobId: string): Promise<QuantizeJob> {
  return fetchJSON(`/quantize/jobs/${jobId}`);
}

export async function cancelQuantizeJob(jobId: string): Promise<{ message: string }> {
  return fetchJSON(`/quantize/jobs/${jobId}/cancel`, { method: "POST" });
}

export async function importQuantizeToOllama(jobId: string): Promise<{ message: string }> {
  return fetchJSON(`/quantize/import-ollama/${jobId}`, { method: "POST" });
}

export async function deleteQuantizeJob(jobId: string): Promise<{ message: string }> {
  return fetchJSON(`/quantize/jobs/${jobId}`, { method: "DELETE" });
}

export async function importGGUFDirect(params: {
  gguf_path: string;
  model_name: string;
}): Promise<{ model_name: string; gguf_path: string; size: number }> {
  return fetchJSON("/quantize/import-gguf", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function uploadGGUF(
  file: File,
  modelName?: string,
): Promise<{ model_name: string; gguf_path: string; size: number }> {
  const formData = new FormData();
  formData.append("file", file);
  if (modelName) formData.append("model_name", modelName);

  const res = await fetchWithAuth(`${UPLOAD_API_BASE}/quantize/upload-gguf`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || `Upload failed (${res.status})`);
  }
  return res.json();
}

export function streamQuantizeSetup(
  callbacks: {
    onProgress: (data: { stage: string; message: string; percent: number; downloaded_bytes?: number; total_bytes?: number }) => void;
    onDone: () => void;
    onError: (error: string) => void;
  }
): AbortController {
  const controller = new AbortController();

  fetchWithAuth(`${API_BASE}/quantize/setup-tools`, {
    method: "POST",
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        callbacks.onError(err.detail || `Request failed: ${res.status}`);
        return;
      }
      const reader = res.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let eventType = "";
        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            const dataStr = line.slice(5).trim();
            if (!dataStr) continue;
            try {
              const data = JSON.parse(dataStr);
              if (eventType === "progress") {
                callbacks.onProgress(data);
              } else if (eventType === "done") {
                callbacks.onDone();
              } else if (eventType === "error") {
                callbacks.onError(data.message || "Unknown error");
              }
            } catch {
              // skip
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        callbacks.onError(err.message);
      }
    });

  return controller;
}

export function streamQuantizeJob(
  jobId: string,
  callbacks: {
    onStatus: (job: QuantizeJob) => void;
    onDone: () => void;
    onError: (error: string) => void;
  }
): AbortController {
  const controller = new AbortController();

  fetchWithAuth(`${API_BASE}/quantize/jobs/${jobId}/stream`, {
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        callbacks.onError(err.detail || `Request failed: ${res.status}`);
        return;
      }
      const reader = res.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let eventType = "";
        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            const dataStr = line.slice(5).trim();
            if (!dataStr) continue;
            try {
              const data = JSON.parse(dataStr);
              if (eventType === "status") {
                callbacks.onStatus(data);
              } else if (eventType === "done") {
                callbacks.onDone();
              } else if (eventType === "error") {
                callbacks.onError(data.message || "Unknown error");
              }
            } catch {
              // skip
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        callbacks.onError(err.message);
      }
    });

  return controller;
}
