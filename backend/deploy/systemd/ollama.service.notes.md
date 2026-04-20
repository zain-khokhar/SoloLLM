# Ollama Service Notes

Recommended deployment model:

- install Ollama separately on Ubuntu
- keep Ollama bound to `127.0.0.1:11434`
- point the backend at it through `SOLOLLM_OLLAMA_BASE_URL=http://127.0.0.1:11434`

Basic setup:

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable ollama
sudo systemctl start ollama
```

Verify:

```bash
curl http://127.0.0.1:11434/api/tags
```

For the first Ubuntu deployment, prefer a separately managed Ollama service over the backend's embedded auto-start path.