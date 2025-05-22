# Agent Cortex v2

**Agent Cortex** is a local, multi-tool AI assistant built with LangChain, Fast retrieval, and a reasoning-capable language model (Mistral 7B). It can answer questions using a mix of:

- 🔍 Document retrieval (RAG)
- 💬 Short-term memory (chat history)
- 🧠 Long-term memory (fact retention across sessions)
- 🌐 Web search (DuckDuckGo)
- 🐍 Python interpreter tool
- 🧮 Math calculations

All running **locally** with no paid APIs.

---

## Working Cortex CLI

![Agent Cortex CLI Demo](./assets/cli-screenshot-v1.png)

In this terminal session, Agent Cortex handles three different types of queries using different tools. The LLM decides which tool will be optimal to get the best answer:

1. **Retrieval (RAG)** — `Where are the fireworks on July 3rd?`  
   → Retrieved from local documents indexed about Bristol, RI events.

2. **Web Search** — `What is the weather in Bristol, RI tomorrow?`  
   → Uses DuckDuckGo to search live internet results and summarizes forecast.

3. **Calculator Tool** — `What is 5 * 7 + 15?`  
   → Routes through a custom calculator tool to evaluate the expression.

4. **Short-Term-Memory** — `My name is Jacob.` then later: `What is my name?`  
   → Agent remembers and recalls personal facts.

5. **Long-Term-Memory** — The agent determins if input is long term storage desirable and stores in a local vector database. upon shutting the agent down and startup up later it is able to recall information about the user that it had saved.
   → Agent remembers and recalls personal facts.

6. **Python REPL** — `sum([1, 2, 3])`  
   → Executes Python code securely using a local interpreter.

Each query is interpreted by the ReAct-based agent and routed to the appropriate tool — all executed **locally** with no API calls or internet billing of an LLM. The websearch is real though but not using outside LLMs for reasoning.

---

## Features

- Retrieval-Augmented Generation from `.txt` documents
- Short-term memory using chat history context
- Long-term memory — agent remembers facts across sessions
- Tool-based reasoning using LangChain’s ReAct agent
- DuckDuckGo web search integration
- 🐍 Python REPL tool for executing code
- Calculator for numeric inputs
- Mistral 7B via Ollama (runs locally as CLI)
- Fallback + context injection for vague queries
- CLI-based agent chat interface loop

---

## Tech Stack

- [LangChain](https://www.langchain.com/)
- [Ollama](https://ollama.com) (local LLM hosting)
- [Mistral 7B](https://ollama.com/library/mistral)
- [ChromaDB](https://www.trychroma.com/) (vector store)
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- Python 3.10
- Poetry for dependency management

---

## Ollama + Mistral Setup

This project uses [Ollama](https://ollama.com) to run the `mistral` model locally.

### 1. Install Ollama (macOS)

```bash
brew install ollama
```

Or download from [ollama.com/download](https://ollama.com/download) and install the desktop app.

### 2. Download the Mistral Model

```bash
ollama run mistral
```

This will download and launch the Mistral model. Leave it running.

> Make sure `curl http://localhost:11434` returns `{"status":"ok"}`

---

## Getting Started

### Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/agent_cortex_v1.git
cd agent_cortex_v1
```

### Install Poetry & Dependencies

```bash
poetry install
```

### Index Your Documents

Place `.txt` files under `data/documents`, then run:

```bash
PYTHONPATH=. poetry run python scripts/index_documents.py
```

### Start the Agent

```bash
poetry run python main.py
```

You'll be prompted with:

```text
You:
```

Try asking:

- `"What time is the Fourth of July parade?"`
- `"My name is Jacob" followed by "What is my name?"`
- `"sum([2, 4, 6])"`
- `"Who is todays date and the weather look like in Boston?"`
- `"What is 25 * 4 + 3?"`

---

## Limitations

While Agent Cortex v1 is functional, it's an early prototype with several known limitations:

#### Agent Behavior

- Long-term memory is fact-based only: It stores facts like names and locations, not full conversations.
- Short-term memory is session-only: Once you close the CLI, short-term context is reset.
- No agent reflection or self-correction: It does not retry intelligently or summarize thoughts beyond what the base model provides.
- Inconsistent ReAct formatting: The LLM may sometimes fail to produce valid Thought / Action / Action Input format, causing parsing errors or retries.
- Fallbacks are basic and do not yet include streaming or error correction

#### Retrieval System

- Only supports .txt files: No PDF, HTML, or Markdown parsing.
- No document metadata or filtering: The retriever does not rank sources by type, date, or confidence.
- No chunking or advanced preprocessing: Raw text is split into single documents without semantic boundaries.
- No multi-vector fusion: Only single-query similarity search; no query rewriting or reranking logic.
- Static index: You must manually re-index documents after any updates.

#### Performance & Deployment

- No streaming output: The full response is printed only after the agent completes.
- Latency: Mistral via Ollama is slower than hosted APIs, especially on lower-spec machines.
- Ollama dependency: Requires installing and running the Ollama server separately, which some users may find nontrivial.

#### Model Limitations

- No fine-tuning: The Mistral model is used out-of-the-box with no task-specific customization.
- No prompt injection prevention: User input is not sanitized or structured securely for prompt-based attacks.
- No multi-turn tool use: Tools are single-action only — no recursive or multi-step reasoning chains.

---

## License

MIT
