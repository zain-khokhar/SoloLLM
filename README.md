<div align="center">

<img src="https://img.shields.io/badge/Solo-LLM-blueviolet?style=for-the-badge&logoColor=white" alt="SoloLLM" height="40"/>

# SoloLLM

### Your Personal AI — Running Locally, Privately, Entirely Yours.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=flat-square&logo=next.js)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-Embedded-white?style=flat-square)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Chat with AI models on your own machine. Upload documents and ask questions about them.
Build knowledge graphs. Run autonomous agents. No cloud needed. No data leaves your computer.**

[Get Started](#-quick-start) · [Features](#-what-can-it-do) · [Screenshots](#-what-it-looks-like) · [API Docs](#-api) · [Roadmap](#-roadmap)

---

</div>

## 💡 What is SoloLLM?

SoloLLM is a **self-hosted AI platform** that runs entirely on your computer.

Think of it like ChatGPT — but **private**, **offline**, and **free**.

- No API keys needed
- No monthly subscription
- Your conversations never leave your machine
- Works on Windows, macOS, and Linux

> **The idea is simple:** A small model (like 7B parameters) with the right tools can beat a much larger model on real-world tasks. SoloLLM gives your small models superpowers.

---

## ✨ What Can It Do?

<table>
<tr>
<td width="50%">

### 💬 Chat
Talk to any locally-installed LLM with a clean chat interface. Switch models mid-conversation. Your chats are saved and searchable.

### 📄 Documents & RAG
Upload PDFs, Word docs, text files, code, CSVs, HTML — then **ask questions about them**. SoloLLM finds the right paragraphs, ranks them, and gives you answers with citations.

### 🧠 Knowledge Graph
SoloLLM reads your documents and builds a visual map of people, places, concepts, and how they're connected. Explore it with an interactive graph.

</td>
<td width="50%">

### 🤖 Agents
Give an agent a task and watch it think step-by-step. It can use tools — run calculations, execute code, search your docs, browse the web, and more.

### 🗜️ Context Distillation
The secret sauce. SoloLLM compresses and cleans up document context before sending it to the model — so small models get maximum signal, not noise.

### 📊 Dashboard
See how your system is performing — request counts, token usage, latency, error rates, and which models you use most.

</td>
</tr>
</table>

---

## 🧩 Feature Highlights

| Feature | What It Does |
|:--------|:-------------|
| **Embedded Ollama** | Ollama downloads and starts automatically — zero setup |
| **26+ Model Catalog** | Pick from TinyLlama, Llama 3, Mistral, Gemma, DeepSeek, CodeLlama, and more |
| **Hardware Profiler** | Detects your GPU/CPU/RAM and recommends the best model for your machine |
| **Auto-Continuation** | If a response gets cut off, SoloLLM detects it and continues automatically |
| **Hybrid Search** | Combines semantic (vector) + keyword (BM25) search with reranking |
| **Multi-Hop Retrieval** | For complex questions, it retrieves info in multiple passes |
| **OCR Support** | Can read scanned PDFs and images (EasyOCR / Tesseract) |
| **Web Scraping** | Fetch content from URLs and add it to your knowledge base |
| **OpenAI-Compatible API** | Drop-in replacement for OpenAI's `/v1/chat/completions` endpoint |
| **Export / Import** | Back up and restore your conversations and documents as JSON |
| **Dark Mode** | Easy on the eyes 🌙 |

---

## 🛠️ Tech Stack

```
┌─────────────────────────────────────────────────┐
│                   Frontend                       │
│   Next.js 16 · React 19 · TailwindCSS 4         │
├─────────────────────────────────────────────────┤
│                   Backend                        │
│   FastAPI · Python · SQLite · SSE Streaming      │
├─────────────────────────────────────────────────┤
│                AI / ML Layer                     │
│   Ollama (embedded) · sentence-transformers      │
│   cross-encoders · EasyOCR · NetworkX            │
└─────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **Node.js 18+**
- A computer (GPU recommended but not required)

### 1. Clone the repo

```bash
git clone https://github.com/your-username/solollm.git
cd solollm
```

### 2. Start the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Open your browser

Go to **http://localhost:3000** — that's it!

> **First time?** SoloLLM will show a Setup Wizard. It downloads Ollama for you, lets you pick a model, and you're chatting in minutes.

---

## 📁 Project Structure

```
solollm/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── api/                  # REST endpoints
│   │   ├── chat.py           #   Chat & conversations
│   │   ├── models.py         #   Model management
│   │   ├── documents.py      #   Document upload & RAG queries
│   │   ├── distillation.py   #   Context distillation
│   │   ├── graph.py          #   Knowledge graph
│   │   ├── agent.py          #   Agent framework
│   │   ├── dashboard.py      #   Metrics & analytics
│   │   ├── openai_compat.py  #   OpenAI-compatible API
│   │   └── export_import.py  #   Backup & restore
│   ├── core/                 # Business logic
│   │   ├── config.py         #   All settings
│   │   ├── inference.py      #   Ollama client
│   │   ├── ollama_manager.py #   Embedded Ollama lifecycle
│   │   ├── agents.py         #   ReAct agent & tools
│   │   ├── distillation.py   #   Context compression engine
│   │   ├── continuation.py   #   Auto-continuation
│   │   ├── kv_cache.py       #   KV-cache tracking
│   │   ├── profiler.py       #   Hardware detection
│   │   └── token_budget.py   #   Token counting
│   ├── rag/                  # Document processing pipeline
│   ├── memory/               # Knowledge graph engine
│   └── storage/              # Database & schemas
├── frontend/
│   └── src/
│       ├── app/              # Next.js pages
│       ├── components/       # UI components
│       ├── lib/              # API client
│       └── types/            # TypeScript types
└── data/                     # Local data (git-ignored)
    ├── db/                   #   SQLite databases
    ├── models/               #   Ollama model files
    ├── documents/            #   Uploaded documents
    └── cache/                #   Embeddings cache
```

---

## 🔌 API

SoloLLM exposes a full REST API. Here are the main groups:

| Endpoint Group | Base Path | What It Does |
|:---------------|:----------|:-------------|
| **Chat** | `/api` | Send messages, manage conversations |
| **Models** | `/api` | List, pull, delete models |
| **Documents** | `/api/documents` | Upload files, search with RAG |
| **Distillation** | `/api/distillation` | Configure context compression |
| **Knowledge Graph** | `/api/graph` | Search entities, visualize relationships |
| **Agents** | `/api/agent` | Run agents, manage tools & memory |
| **Dashboard** | `/api/dashboard` | Usage metrics & analytics |
| **System** | `/api` | Health checks, settings, hardware profile |
| **OpenAI-Compatible** | `/v1/chat/completions` | Drop-in replacement for OpenAI API |
| **Export/Import** | `/api/export` | Backup & restore data |

### OpenAI-Compatible Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

Works with any tool that supports the OpenAI API format.

---

## ⚙️ Configuration

All settings live in `backend/core/config.py` and can be overridden with environment variables:

| Setting | Default | What It Does |
|:--------|:--------|:-------------|
| `SOLOLLM_OLLAMA_AUTO_START` | `true` | Auto-download and start Ollama |
| `SOLOLLM_OLLAMA_PORT` | `11434` | Ollama API port |
| `SOLOLLM_DEFAULT_MODEL` | — | Default model for new chats |
| `SOLOLLM_MAX_TOKENS` | `2048` | Max tokens per response |
| `SOLOLLM_TEMPERATURE` | `0.7` | Response creativity (0–1) |
| `SOLOLLM_CONTEXT_WINDOW` | `4096` | Context window size |
| `SOLOLLM_DISTILLATION_ENABLED` | `true` | Enable context compression |
| `SOLOLLM_KNOWLEDGE_GRAPH_ENABLED` | `true` | Enable knowledge graph |
| `SOLOLLM_AGENT_ENABLED` | `true` | Enable agent framework |

---

## 🗺️ Roadmap

SoloLLM was built in phases:

- [x] **Phase 1** — Core chat + Ollama integration
- [x] **Phase 2** — RAG pipeline (upload docs, retrieve, cite)
- [x] **Phase 3** — Context distillation engine
- [x] **Phase 4** — Knowledge graph + web scraping
- [x] **Phase 5** — Agent framework with tools
- [x] **Phase 6** — OpenAI-compatible API, dashboard, export/import

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repo
2. Create your branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built for people who want AI without giving up their privacy.**

⭐ Star this repo if you find it useful!

</div>