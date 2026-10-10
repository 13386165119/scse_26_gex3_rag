# SCSE 26 Graded Exercise 3 — RAG

`policy_loader.py` is the instructor file and is unchanged.

Complete these three files:

- `rag_tool.py` — Lecture 11 embeddings (`nomic-embed-text`) and Chroma similarity search. A small offline BM25 index is included for checks that do not need Ollama.
- `agent.py` — Qwen3-0.6B draft, review, and revision. Phone numbers that are not present in the retrieved policies are rejected.
- `rag.py` — runs the assistant. The default question is the IT Service Desk phone question.

## Setup

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-rag.txt
ollama pull nomic-embed-text
ollama pull qwen3:0.6b
```

Ollama must be running. The chat model is `qwen3:0.6b`, not the 8B model from Lecture 11.

## Run

```text
python rag.py
python rag.py "How can I reset my university account password?"
python check_data.py
python -m unittest test_agent.py
```

`python rag.py` asks: "What exact phone number should I call for the University IT Service Desk?"

The policy file does not contain that phone number. The assistant must say it does not know, and it must not invent one.
