#!/usr/bin/env python3
"""Standalone adaptation of DoD's 'Chatting with your Docs' notebook.

Uses LlamaIndex with Ollama, Hugging Face Transformers, or llama.cpp; HuggingFace embeddings,
and PDF files to answer questions using retrieval-augmented generation.

Install (CPU):
    python -m pip install llama-index-core llama-index-llms-llama-cpp \
        llama-index-embeddings-huggingface llama-index-readers-file \
        llama-cpp-python sentence-transformers pypdf requests gradio

For CUDA acceleration, install a CUDA-enabled build of llama-cpp-python
appropriate for your CUDA toolkit and system, rather than the CPU wheel.

Examples:
    python chatting_with_your_docs.py --download-samples
    python chatting_with_your_docs.py --download-samples --backend ollama --model llama3.2
    python chatting_with_your_docs.py --documents ./my_pdfs --backend huggingface --model Qwen/Qwen2.5-1.5B-Instruct
    python chatting_with_your_docs.py --documents ./my_pdfs --model-path ./model.gguf
    python chatting_with_your_docs.py --documents ./my_pdfs --gradio
    python chatting_with_your_docs.py --documents ./my_pdfs --question 'Summarize the policy'

By default no public Gradio sharing is enabled.
"""

import argparse
from pathlib import Path
import sys

import requests

SAMPLE_PDFS = {
    "dod_doj_policy.pdf": "https://www.esd.whs.mil/Portals/54/Documents/DD/issuances/dodi/503007p.pdf?ver=FdbnkRjs8wfSzwTV7XNPGw%3d%3d",
    "dod_lasers_policy.pdf": "https://www.esd.whs.mil/Portals/54/Documents/DD/issuances/dodi/605515p.pdf?ver=NL-WXDYnI9H5TOwUUi82lw%3d%3d",
}
DEFAULT_MODEL_URL = (
    "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/"
    "mistral-7b-openorca.Q5_K_M.gguf"
)


def download_samples(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, url in SAMPLE_PDFS.items():
        dest = directory / filename
        if dest.exists():
            print(f"Already exists: {dest}")
            continue
        print(f"Downloading {filename} ...")
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with dest.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        print(f"Saved: {dest}")


def create_llm(args):
    if args.backend == "ollama":
        from llama_index.llms.ollama import Ollama
        return Ollama(
            model=args.model,
            base_url=args.ollama_url,
            request_timeout=args.request_timeout,
            context_window=args.context_window,
            temperature=0.1,
        )

    if args.backend == "huggingface":
        from llama_index.llms.huggingface import HuggingFaceLLM
        return HuggingFaceLLM(
            model_name=args.model,
            tokenizer_name=args.model,
            context_window=args.context_window,
            max_new_tokens=256,
            device_map="auto",
            generate_kwargs={"do_sample": False},
        )

    from llama_index.llms.llama_cpp import LlamaCPP
    from llama_index.llms.llama_cpp.base import messages_to_prompt, completion_to_prompt
    kwargs = dict(
        temperature=0.1,
        max_new_tokens=256,
        context_window=args.context_window,
        model_kwargs={"n_gpu_layers": args.gpu_layers},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=args.verbose,
    )
    if args.model_path:
        kwargs["model_path"] = str(args.model_path)
    else:
        kwargs["model_url"] = args.model_url
    return LlamaCPP(**kwargs)


def build_query_engine(args):
    from llama_index.core import (
        Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex,
        load_index_from_storage,
    )
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.llm = create_llm(args)
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding_model)

    if args.reuse_index and args.index_dir.exists():
        print(f"Loading existing index from {args.index_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=str(args.index_dir))
        index = load_index_from_storage(storage_context)
    else:
        pdfs = sorted(args.documents.rglob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(
                f"No PDF files found in {args.documents}. "
                "Use --download-samples or --documents PATH."
            )
        print(f"Indexing {len(pdfs)} PDF(s) from {args.documents} ...")
        documents = SimpleDirectoryReader(
            input_files=[str(path) for path in pdfs],
        ).load_data()
        index = VectorStoreIndex.from_documents(documents)
        args.index_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(args.index_dir))
        print(f"Index saved to {args.index_dir}")
    return index.as_query_engine()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--documents", type=Path, default=Path("sample_documents"), help="Directory containing PDF documents")
    parser.add_argument("--download-samples", action="store_true", help="Download two example DoD policy PDFs")
    parser.add_argument("--backend", choices=["llama-cpp", "ollama", "huggingface"], default="llama-cpp",
                        help="LLM backend (default: llama-cpp for backwards compatibility)")
    parser.add_argument("--model", help="Ollama model tag or Hugging Face Transformers model ID")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama server URL")
    parser.add_argument("--request-timeout", type=float, default=300.0, help="Ollama request timeout (seconds)")
    parser.add_argument("--model-path", type=Path, help="Path to an existing GGUF file (llama-cpp only)")
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL, help="URL of a GGUF model to download")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--gpu-layers", type=int, default=0, help="GPU-offloaded layers, e.g. 30; 0 is CPU only")
    parser.add_argument("--context-window", type=int, default=3900)
    parser.add_argument("--index-dir", type=Path, default=Path("test_index"))
    parser.add_argument("--reuse-index", action="store_true", help="Load a previously saved index, if present")
    parser.add_argument("--question", help="Ask one question and exit")
    parser.add_argument("--gradio", action="store_true", help="Start a local Gradio web interface")
    parser.add_argument("--host", default="127.0.0.1", help="Gradio bind address")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.backend in {"ollama", "huggingface"} and not args.model:
        parser.error("--model is required with --backend ollama or --backend huggingface")
    if args.backend == "llama-cpp" and args.model:
        parser.error("--model is for Ollama/Hugging Face; use --model-path or --model-url for llama-cpp")
    if args.backend != "llama-cpp" and args.model_path:
        parser.error("--model-path is only supported with --backend llama-cpp")

    try:
        if args.download_samples:
            download_samples(args.documents)
        engine = build_query_engine(args)

        def ask(question: str) -> str:
            if not question.strip():
                return "Please enter a question."
            return str(engine.query(question))

        if args.question:
            print(ask(args.question))
        elif args.gradio:
            import gradio as gr
            gr.Interface(
                fn=ask,
                inputs=gr.Textbox(label="Question", lines=2),
                outputs=gr.Textbox(label="Answer", lines=12),
                title="Chatting with your Docs",
                description="Ask questions about the PDFs in your local document index.",
            ).launch(server_name=args.host, server_port=args.port, share=False)
        else:
            print("Ready. Enter a question, or type 'quit' to exit.")
            while True:
                try:
                    question = input("\nQuestion> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if question.lower() in {"quit", "exit", "q"}:
                    break
                if question:
                    print("\n" + ask(question))
    except (FileNotFoundError, requests.RequestException) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
