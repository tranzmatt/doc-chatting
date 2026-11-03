#!/usr/bin/env python3
"""Run an end-to-end local RAG test with Ollama, Hugging Face or GGUF.

Example:
    python tests/run_integration.py --backend ollama --model llama3.2 --download-samples
    python tests/run_integration.py --backend huggingface --model Qwen/Qwen2.5-1.5B-Instruct --download-samples
    python tests/run_integration.py --model-path /models/model.gguf --download-samples
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "chatting_with_your_docs.py"


def run_case(name, command, timeout):
    print(f"\n=== {name} ===", flush=True)
    print("+ " + " ".join(map(str, command)), flush=True)
    start = time.monotonic()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{name} timed out after {timeout} seconds") from exc
    print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    elapsed = time.monotonic() - start
    print(f"{name}: exit={result.returncode}, elapsed={elapsed:.1f}s", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed (exit {result.returncode})")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["llama-cpp", "ollama", "huggingface"], default="llama-cpp")
    parser.add_argument("--model", help="Ollama model tag or Hugging Face model ID")
    parser.add_argument("--model-path", type=Path, help="Existing local GGUF model for llama-cpp")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--documents", type=Path, default=ROOT / "sample_documents")
    parser.add_argument("--download-samples", action="store_true", help="Download the public sample PDFs")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--context-window", type=int, default=3900)
    parser.add_argument("--question", default="What should I do in the event of a military laser incident?")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout per subprocess in seconds")
    parser.add_argument("--skip-reuse", action="store_true", help="Skip the persisted-index reload test")
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve() if args.model_path else None
    documents = args.documents.expanduser().resolve()
    if args.backend == "llama-cpp" and (model_path is None or not model_path.is_file()):
        parser.error("--model-path must point to an existing GGUF file for llama-cpp")
    if args.backend != "llama-cpp" and not args.model:
        parser.error("--model is required for Ollama and Hugging Face")
    if args.backend == "ollama":
        import requests
        try:
            response = requests.get(args.ollama_url.rstrip("/") + "/api/tags", timeout=10)
            response.raise_for_status()
            available = [entry.get("name") for entry in response.json().get("models", [])]
        except (requests.RequestException, ValueError) as exc:
            parser.error(f"Cannot contact Ollama at {args.ollama_url}: {exc}")
        if args.model not in available and (args.model + ":latest") not in available:
            parser.error(f"Ollama model {args.model!r} not installed. Available models: {available}. Run: ollama pull {args.model}")
    if not args.download_samples and not list(documents.rglob("*.pdf")):
        parser.error(f"No PDFs found under {documents}; supply --documents or --download-samples")

    print(f"Python: {sys.version.split()[0]}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model if args.model else model_path}")
    print(f"PDFs: {documents}")
    print(f"GPU layers: {args.gpu_layers}")
    with tempfile.TemporaryDirectory(prefix="doc-chatting-test-") as tmp:
        index_dir = Path(tmp) / "index"
        command = [
            sys.executable, str(APP),
            "--documents", str(documents),
            "--index-dir", str(index_dir),
            "--backend", args.backend,
            "--gpu-layers", str(args.gpu_layers),
            "--context-window", str(args.context_window),
            "--question", args.question,
        ]
        if args.backend == "llama-cpp":
            command.extend(["--model-path", str(model_path)])
        else:
            command.extend(["--model", args.model])
        if args.backend == "ollama":
            command.extend(["--ollama-url", args.ollama_url,
                            "--request-timeout", str(args.request_timeout)])
        if args.download_samples:
            command.append("--download-samples")
        first_output = run_case("Build index and answer", command, args.timeout)
        if "Index saved to " not in first_output or not index_dir.is_dir():
            raise RuntimeError("No persisted index created")
        if not first_output.strip().splitlines()[-1].strip():
            raise RuntimeError("Query produced no visible output")
        if not args.skip_reuse:
            reused_output = run_case(
                "Reload index and answer", command + ["--reuse-index"], args.timeout
            )
            if "Loading existing index from " not in reused_output:
                raise RuntimeError("Persisted index was not reloaded")
            if not reused_output.strip().splitlines()[-1].strip():
                raise RuntimeError("Reload query produced no visible output")

    print("\nPASS: end-to-end RAG workflow completed.", flush=True)
    print("Note: This is a functional test, not an evaluation of answer accuracy.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
