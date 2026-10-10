"""Run the assignment's retrieval-augmented Qwen3-0.6B assistant.

Example:
  python rag.py
  python rag.py "How can I reset my university account password?"
  python rag.py --show-sources
  python rag.py --retrieve-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from agent import PolicyAgent, default_model_path
from policy_loader import load_policy_documents
from rag_tool import PolicyRetriever, VectorPolicyRetriever, format_context

# Exactly the instructor-required loader, without editing policy_loader.py.
documents = load_policy_documents()

DEFAULT_QUESTION = "What exact phone number should I call for the University IT Service Desk?"


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG over the university IT policies")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--model-path", help="Local Qwen3-0.6B directory; never automatically downloads weights")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval", choices=['vector', 'bm25'], default='vector')
    parser.add_argument("--backend", choices=['ollama', 'transformers'], default='ollama')
    parser.add_argument("--trace", help="Save actual generation/reflection/guard results as JSON")
    parser.add_argument("--no-reflection", action='store_true')
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument("--retrieve-only", action="store_true",
                        help="Debug retrieval; this is NOT a model inference test")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error('--top-k must be positive')
    if args.retrieval == 'vector':
        retriever = VectorPolicyRetriever(documents, str(Path(__file__).resolve().parent / '.chroma'))
    else:
        retriever = PolicyRetriever(documents)
    selected = retriever.search(args.question, top_k=args.top_k)
    if args.show_sources or args.retrieve_only:
        print("=== RETRIEVED POLICIES ===")
        print(format_context(selected) if selected else "(none)")
        print()
    if args.retrieve_only:
        return

    agent = PolicyAgent(model_path=args.model_path, backend=args.backend, reflection=not args.no_reflection)
    answer = agent.answer(args.question, selected)
    print("QUESTION:", args.question)
    print("ANSWER:", answer)
    print("RETRIEVED POLICY IDS:", [doc.metadata["policy_id"] for doc in selected])
    if args.trace:
        Path(args.trace).write_text(json.dumps(agent.last_trace, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == "__main__":
    main()
