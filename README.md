# doc-chatting

A standalone Python adaptation of the [DoD **Chatting with your Docs** notebook](https://github.com/deptofdefense/LLMs-at-DoD/blob/main/tutorials/Chatting%20with%20your%20Docs.ipynb) by Glenn Parham (Defense Digital Service). Ask questions about local PDFs using retrieval-augmented generation (RAG), a local GGUF language model with llama.cpp, LlamaIndex, and Hugging Face embeddings.

## Features

- Download two public DoD policy PDFs for a working example, or index your own PDFs.
- Run inference locally with a GGUF model and optional GPU layer offloading.
- Persist the vector index and reload it on subsequent runs.
- Ask a single question, use a terminal prompt, or launch a **local-only** Gradio UI.

## Prerequisites

Python 3.10+ recommended. A GGUF model needs significant disk space and memory; CPU-only inference may be slow. The default model is `TheBloke/Mistral-7B-OpenOrca-GGUF` (`Q5_K_M`) and is downloaded on first use. You can specify a GGUF model already on disk instead.

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

> **GPU acceleration:** The ordinary `llama-cpp-python` wheel may be CPU-only. To use CUDA, install a CUDA-enabled build matching your toolchain, for example with `CMAKE_ARGS="-DGGML_CUDA=on"` and `FORCE_CMAKE=1` when building from source. See the [llama-cpp-python install instructions](https://github.com/abetlen/llama-cpp-python#installation) for your platform.

## Quick start

Download the example PDFs, build an index, and open an interactive command-line session:

```bash
python chatting_with_your_docs.py --download-samples
```

At the `Question>` prompt, ask about the policies. Type `quit` to exit.

Run against your own PDF directory and local model:

```bash
python chatting_with_your_docs.py --documents ./my_pdfs --model-path ./model.gguf --gpu-layers 30
```

Ask a single question:

```bash
python chatting_with_your_docs.py --documents ./my_pdfs --model-path ./model.gguf \
  --question "Summarize the key requirements"
```

Reuse an existing index (rather than re-embedding documents):

```bash
python chatting_with_your_docs.py --documents ./my_pdfs --model-path ./model.gguf --reuse-index
```

Start a Gradio web UI:

```bash
python chatting_with_your_docs.py --documents ./my_pdfs --model-path ./model.gguf --reuse-index --gradio
```

Open <http://127.0.0.1:7860>. By default Gradio listens on localhost and `share=False`; it does **not** publish a public sharing link. Only bind a public/network host if you understand the security implications. **Do not use this example with classified or sensitive documents without an appropriately authorized environment and review.**

## CLI options

```bash
python chatting_with_your_docs.py --help
```

Important flags: `--documents`, `--download-samples`, `--model-path`, `--model-url`, `--embedding-model`, `--gpu-layers`, `--context-window`, `--index-dir`, `--reuse-index`, `--question`, `--gradio`, `--host`, `--port`, `--verbose`.

## How it works

1. Read PDFs from a folder with LlamaIndex's `SimpleDirectoryReader` and PDF reader integration.
2. Embed document passages using `BAAI/bge-small-en-v1.5` by default.
3. Create and persist a `VectorStoreIndex` in `test_index/`.
4. Retrieve matching passages for each question and generate an answer using llama.cpp.

**Index freshness:** `--reuse-index` loads the saved index as-is, even when the files in `--documents` have changed. Omit the flag to rebuild it. If changing embedding models, rebuild the index.

## Notes

- First-time use may download model weights and embedding weights.
- The sample policy PDF URLs are external government resources and may change over time.
- This script uses a **local llama.cpp GGUF model**, not an OpenAI-compatible hosted/vLLM endpoint. That can be a future enhancement.
- No model weights, downloaded PDFs, or generated indices are tracked in Git.
- Generated answers may contain errors. Check citations and source PDFs before relying on answers.

## Attribution

Adapted from [deptofdefense/LLMs-at-DoD — *Chatting with your Docs*](https://github.com/deptofdefense/LLMs-at-DoD/blob/main/tutorials/Chatting%20with%20your%20Docs.ipynb). See the upstream project for original documentation and applicable licensing. This repository is an independent adaptation and is not an official Department of Defense product.
