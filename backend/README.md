# SoloLLM Backend

This is the Ubuntu deployment backend for SoloLLM.

## What is here

- FastAPI backend entrypoint in `main.py`
- application modules under `api/`, `core/`, `rag/`, `storage/`, `academic/`, and `memory/`
- Ubuntu deployment assets under `deploy/`
- production env template in `.env.example`

## Production target

- Ubuntu server hosts the FastAPI backend
- Ollama runs locally on the same server at `127.0.0.1:11434`
- Nginx terminates HTTPS and proxies to the backend on `127.0.0.1:8000`
- Vercel frontend talks to `https://api.your-domain.com`
- one owner login protects the private deployment

## Quick start

1. Create a virtual environment.
2. Install `requirements.txt`.
3. Copy `.env.example` to `.env` and set your production values.
4. Run `uvicorn main:app --host 127.0.0.1 --port 8000 --proxy-headers`.

## Ubuntu assets

- `deploy/ubuntu/bootstrap_ubuntu.sh`: installs base system packages and Python dependencies
- `deploy/ubuntu/healthcheck.sh`: smoke-test for `/api/health`
- `deploy/systemd/solollm-backend.service`: example systemd unit
- `deploy/systemd/ollama.service.notes.md`: Ollama service notes
- `deploy/nginx/api.your-domain.com.conf`: example Nginx site config

## Notes

- Set `SOLOLLM_PRIVATE_ACCESS_ENABLED=true` with `SOLOLLM_OWNER_USERNAME`, `SOLOLLM_OWNER_PASSWORD`, and `SOLOLLM_SESSION_SECRET` for the intended owner-only deployment mode.
- `SOLOLLM_ADMIN_API_TOKEN` stays available as an optional direct API key for OpenAI-compatible clients or scripted admin calls.
- Data defaults to `data/` when no explicit `SOLOLLM_DATA_DIR` is set. On Ubuntu, use a persistent path such as `/var/lib/solollm`.