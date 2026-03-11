# Known Issues & Fixes

## Issue #1 — 503 on `/api/models` + ECONNREFUSED on Frontend

**Status:** Ollama not running

### Root Cause
Backend logs show:
```
connect_tcp.failed exception=ConnectError(OSError('All connection attempts failed'))
GET /api/models HTTP/1.1" 503 Service Unavailable
```
Backend tries to connect to **Ollama at `localhost:11434`** but Ollama is not started.  
Without Ollama, the models list returns 503, and the frontend shows `ECONNREFUSED`.

### Fix — Start Ollama

```bash
ollama serve
```

Then verify Ollama is up:
```bash
curl http://localhost:11434
```

### Fix — Pull a Default Model (if no model installed)

```bash
ollama pull llama3.2
```
or any other model:
```bash
ollama pull mistral
ollama pull gemma2
```

List installed models:
```bash
ollama list
```

### Correct Startup Order

1. `ollama serve` — start Ollama (port 11434)
2. `uvicorn main:app --reload --port 8000` — start backend (port 8000)
3. `npm run dev` — start frontend (port 3000)

All three must be running simultaneously for the app to work.
