#!/usr/bin/env python3
"""Run an end-to-end local RAG test (requires installed dependencies and GGUF weights).

Example:
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
    parser.add_argument("--model-path", required=True, type=Path, help="Existing local GGUF model")
    parser.add_argument("--documents", type=Path, default=ROOT / "sample_documents")
    parser.add_argument("--download-samples", action="store_true", help="Download the public sample PDFs")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--context-window", type=int, default=3900)
    parser.add_argument("--question", default="What should I do in the event of a military laser incident?")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout per subprocess in seconds")
    parser.add_argument("--skip-reuse", action="store_true", help="Skip the persisted-index reload test")
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    documents = args.documents.expanduser().resolve()
    if not model_path.is_file():
        parser.error(f"GGUF file not found: {model_path}")
    if not args.download_samples and not list(documents.rglob("*.pdf")):
        parser.error(f"No PDFs found under {documents}; supply --documents or --download-samples")

    print(f"Python: {sys.version.split()[0]}")
    print(f"Model: {model_path}")
    print(f"PDFs: {documents}")
    print(f"GPU layers: {args.gpu_layers}")
    with tempfile.TemporaryDirectory(prefix="doc-chatting-test-") as tmp:
        index_dir = Path(tmp) / "index"
        command = [
            sys.executable, str(APP),
            "--documents", str(documents),
            "--model-path", str(model_path),
            "--index-dir", str(index_dir),
            "--gpu-layers", str(args.gpu_layers),
            "--context-window", str(args.context_window),
            "--question", args.question,
        ]
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
