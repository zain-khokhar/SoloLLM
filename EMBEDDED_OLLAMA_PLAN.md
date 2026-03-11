# SoloLLM — Embedded Model Runner Plan

## Goal
Eliminate the requirement for users to separately install Ollama. The app should be fully self-contained: start the app → pick a model → chat. Zero technical setup.

---

## Current Architecture

```
Frontend (Next.js :3000)
    ↓ HTTP
Backend (FastAPI :8000)
    ↓ HTTP (Ollama API)
Ollama (:11434)  ← USER MUST INSTALL SEPARATELY ❌
    ↓
llama.cpp → GPU/CPU inference
```

### Current Ollama Touchpoints
All Ollama communication goes through **one file**: `backend/core/inference.py` → `OllamaClient` class.

| Method | Ollama Endpoint | Purpose |
|--------|----------------|---------|
| `is_available()` | `GET /api/tags` | Health check |
| `list_models()` | `GET /api/tags` | List installed models |
| `pull_model()` | `POST /api/pull` | Download model from registry |
| `delete_model()` | `DELETE /api/delete` | Remove a model |
| `chat_stream()` | `POST /api/chat` | Streaming inference |
| `chat()` | `POST /api/chat` | Non-streaming inference |

Config in `backend/core/config.py`:
- `ollama_base_url = "http://localhost:11434"`
- `default_model = "llama3.2:latest"`

API routes in `backend/api/models.py` call `ollama_client.*` methods.

---

## Recommended Approach: Embedded Ollama Binary

### Why This Approach
- **Minimal code changes** — backend already speaks Ollama HTTP API, just need to auto-start it
- **Ollama handles everything** — GGUF downloading, GPU detection, memory management, quantization
- **Battle-tested** — Ollama is stable, well-maintained, and handles edge cases (partial downloads, resume, etc.)
- **Cross-platform** — Ollama has binaries for Windows, macOS, Linux

### Alternative Rejected: llama-cpp-python
Using llama-cpp-python directly would eliminate Ollama entirely but requires:
- Rewriting all inference code
- Building your own model download/management system
- Handling GPU detection, memory mapping, GGUF parsing manually
- Much more complex and fragile — not worth it when Ollama already does this perfectly

---

## Implementation Plan

### Phase 1: Ollama Lifecycle Manager
**New file:** `backend/core/ollama_manager.py`

```
Purpose: Download, install, and manage the Ollama binary as a subprocess
```

#### 1.1 — OllamaManager Class
```python
class OllamaManager:
    """Manages the embedded Ollama binary lifecycle."""
    
    # Responsibilities:
    # 1. Check if Ollama is already running (system-wide or our instance)
    # 2. If not, check if binary exists in data/ollama/
    # 3. If not, download it (with progress callback)
    # 4. Start Ollama as subprocess on a configured port
    # 5. Wait for health check to pass
    # 6. Stop Ollama on app shutdown
```

#### 1.2 — Key Methods to Implement
```python
async def ensure_ollama_running(self) -> bool:
    """Main entry point. Ensures Ollama is available. Downloads + starts if needed."""

async def _check_existing_ollama(self) -> bool:
    """Check if Ollama is already running on the configured port."""

async def _download_ollama_binary(self, progress_callback=None) -> Path:
    """Download Ollama binary for current OS to data/ollama/."""
    # Windows: download from https://github.com/ollama/ollama/releases
    # Store in: data/ollama/ollama.exe (Windows) or data/ollama/ollama (Linux/Mac)

async def _start_ollama(self) -> bool:
    """Start Ollama as a subprocess."""
    # Run: ollama serve
    # Set OLLAMA_HOST=127.0.0.1:{port}
    # Set OLLAMA_MODELS={data_dir}/models (keep models inside project)
    # Capture stdout/stderr for logging
    # Wait for health check on /api/tags

async def stop(self) -> None:
    """Stop the managed Ollama subprocess."""

def is_managed(self) -> bool:
    """Returns True if we started Ollama (vs using system Ollama)."""
```

#### 1.3 — Config Additions to `backend/core/config.py`
```python
# Embedded Ollama
ollama_auto_start: bool = True            # Auto-download and start Ollama
ollama_binary_dir: Path = data_dir / "ollama"
ollama_models_dir: Path = data_dir / "models"  # Store models inside project
ollama_port: int = 11434                  # Port for embedded Ollama
```

#### 1.4 — Download URLs (Ollama Releases)
```
Windows: https://github.com/ollama/ollama/releases/download/v{version}/ollama-windows-amd64.zip
Linux:   https://github.com/ollama/ollama/releases/download/v{version}/ollama-linux-amd64.tgz
macOS:   https://github.com/ollama/ollama/releases/download/v{version}/ollama-darwin
```

Store pinned version in config (e.g., `ollama_version: str = "0.6.2"`).

---

### Phase 2: Backend Integration

#### 2.1 — Startup Hook in `backend/main.py`
```python
@app.on_event("startup")
async def startup():
    # ... existing DB init ...
    
    # NEW: Ensure Ollama is running
    if settings.ollama_auto_start:
        from core.ollama_manager import ollama_manager
        success = await ollama_manager.ensure_ollama_running()
        if success:
            logger.info("Ollama is ready")
        else:
            logger.warning("Ollama could not be started — models won't be available")

@app.on_event("shutdown")
async def shutdown():
    # NEW: Stop managed Ollama
    from core.ollama_manager import ollama_manager
    await ollama_manager.stop()
```

#### 2.2 — New API Endpoint: Ollama Status
**Add to** `backend/api/system.py` or new `backend/api/runtime.py`:
```
GET /api/runtime/status → { ollama_running, ollama_managed, ollama_version, models_dir }
POST /api/runtime/setup  → trigger Ollama download + start (for first-time setup flow)
```

#### 2.3 — No Changes to `inference.py`
`OllamaClient` stays exactly the same — it already talks HTTP to Ollama. The manager just ensures Ollama is running before the client tries to connect.

---

### Phase 3: Frontend Setup Flow

#### 3.1 — First-Time Setup Screen
When frontend detects no Ollama + no models:

```
┌──────────────────────────────────────┐
│  Welcome to SoloLLM! 🚀             │
│                                      │
│  Setting up your AI environment...   │
│                                      │
│  [████████░░░░] Downloading engine   │
│                                      │
│  Next: Choose your first model       │
└──────────────────────────────────────┘
```

#### 3.2 — Model Picker (already exists, enhance it)
After Ollama is ready, show model picker with recommended models:

```
┌──────────────────────────────────────┐
│  Choose a Model                      │
│                                      │
│  ⭐ llama3.2 (3B) — 2.0 GB          │
│     Fast, great for chat             │
│                                      │
│  🧠 mistral (7B) — 4.1 GB           │
│     Balanced quality & speed         │
│                                      │
│  🔬 gemma2 (9B) — 5.4 GB            │
│     Best quality, needs more RAM     │
│                                      │
│  [Download Selected]                 │
└──────────────────────────────────────┘
```

Model pull already works via `POST /api/models/pull` with SSE progress.

---

### Phase 4: Error Handling & Edge Cases

| Scenario | Handling |
|----------|----------|
| System Ollama already running | Detect & use it, don't start another |
| Port 11434 occupied by non-Ollama | Try alternate port (11435, 11436...) |
| Download interrupted | Resume/retry (HTTP range requests) |
| No GPU detected | Ollama auto-falls back to CPU (built-in) |
| Insufficient disk space | Check before download, show clear error |
| Binary permissions (Linux/Mac) | chmod +x after download |
| Antivirus blocks binary (Windows) | Show instructions to whitelist |
| Ollama subprocess crashes | Detect via health check, auto-restart (max 3 times) |
| App closed while model downloading | Resume on next start (Ollama handles this) |

---

## File Changes Summary

| File | Change Type | Description |
|------|------------|-------------|
| `backend/core/ollama_manager.py` | **NEW** | Ollama lifecycle manager |
| `backend/core/config.py` | EDIT | Add ollama_auto_start, ollama_binary_dir, ollama_models_dir settings |
| `backend/main.py` | EDIT | Add startup/shutdown hooks for OllamaManager |
| `backend/api/system.py` | EDIT | Add `/api/runtime/status` and `/api/runtime/setup` endpoints |
| `frontend/src/components/setup/SetupWizard.tsx` | **NEW** | First-time setup UI |
| `frontend/src/components/setup/ModelPicker.tsx` | **NEW** | Model selection with size/description |
| `frontend/src/app/page.tsx` | EDIT | Show SetupWizard on first launch |
| `frontend/src/lib/api.ts` | EDIT | Add runtime status/setup API calls |
| `backend/core/inference.py` | NO CHANGE | Already works via HTTP |
| `backend/api/models.py` | NO CHANGE | Pull/delete already implemented |

---

## Directory Structure After Implementation

```
data/
  ollama/
    ollama.exe          ← Downloaded binary (Windows)
  models/
    manifests/          ← Ollama model manifests
    blobs/              ← Model weight files (GGUF)
  db/
    solollm.db
  documents/
  uploads/
```

---

## User Experience After Implementation

```
1. User downloads SoloLLM
2. Runs: npm run dev + uvicorn (or a single start script)
3. Opens browser → sees "Welcome! Setting up..."
4. Engine auto-downloads (~50MB one-time)
5. User picks a model (e.g., llama3.2 3B) → downloads ~2GB
6. Done. User starts chatting.
7. No Ollama installation needed. No terminal commands. No technical knowledge.
```

---

## Implementation Order

1. `ollama_manager.py` — core binary management (download, start, stop, health check)
2. Config additions — new settings
3. `main.py` hooks — auto-start on backend startup
4. API endpoints — runtime status for frontend
5. Frontend SetupWizard — first-time setup flow
6. Frontend ModelPicker — guided model selection
7. Testing & edge cases
