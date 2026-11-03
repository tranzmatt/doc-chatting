# doc-chatting

A standalone Python adaptation of the [DoD **Chatting with your Docs** notebook](https://github.com/deptofdefense/LLMs-at-DoD/blob/main/tutorials/Chatting%20with%20your%20Docs.ipynb) by Glenn Parham (Defense Digital Service). Ask questions about local PDFs using retrieval-augmented generation (RAG), Ollama, Hugging Face Transformers, or a local GGUF model, plus LlamaIndex and Hugging Face embeddings.

## Features

- Download two public DoD policy PDFs for a working example, or index your own PDFs.
- Choose an installed Ollama model by tag, a Hugging Face Transformers model by ID, or a local GGUF model with optional GPU layer offloading.
- Persist the vector index and reload it on subsequent runs.
- Ask a single question, use a terminal prompt, or launch a **local-only** Gradio UI.

## Prerequisites

Python 3.11 recommended. Model memory requirements vary widely. **Ollama is the easiest choice if you already have models installed**; the script does not download an additional LLM for the Ollama backend. Hugging Face Transformers loads a model from its normal cache or downloads it by model ID if missing. The legacy llama.cpp default downloads `TheBloke/Mistral-7B-OpenOrca-GGUF` on first use.

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-ollama.txt  # or requirements-huggingface.txt / requirements-llama-cpp.txt
```

> **GPU acceleration:** The ordinary `llama-cpp-python` wheel may be CPU-only. To use CUDA, install a CUDA-enabled build matching your toolchain, for example with `CMAKE_ARGS="-DGGML_CUDA=on"` and `FORCE_CMAKE=1` when building from source. See the [llama-cpp-python install instructions](https://github.com/abetlen/llama-cpp-python#installation) for your platform.

## Quick start

Download the example PDFs, build an index, and open an interactive command-line session using a model already available in Ollama:

```bash
ollama list
python chatting_with_your_docs.py --backend ollama --model llama3.2 --download-samples
```

Replace `llama3.2` with a tag shown by `ollama list`. Ollama must be running (normally at `http://127.0.0.1:11434`). For a different server, specify `--ollama-url http://HOST:11434`.

At the `Question>` prompt, ask about the policies. Type `quit` to exit.

Run against your own PDF directory and local GGUF model:

```bash
python chatting_with_your_docs.py --documents ./my_pdfs --model-path ./model.gguf --gpu-layers 30
```

Ask a single question with a Hugging Face Transformers model (downloaded/cached automatically):

```bash
python chatting_with_your_docs.py --backend huggingface --model Qwen/Qwen2.5-1.5B-Instruct --documents ./my_pdfs \
  --question "Summarize the key requirements"
```

Or ask with a GGUF model:

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

## Testing

Run the fast offline smoke tests after installing dependencies. These check CLI help, default model configuration, streamed sample-PDF downloads, and HTTP error handling without loading a language model:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Run an **end-to-end integration test** using a GGUF model already on your machine. With `--download-samples`, the test now **generates a small synthetic PDF locally**, avoiding dependence on external document servers (the original DoD PDF links may return HTTP 403). It builds a fresh index in a temporary directory, asks a question, then loads the persisted index and asks again:

```bash
python tests/run_integration.py --backend ollama --model llama3.2 --download-samples
```

To test a Hugging Face model ID instead:

```bash
python tests/run_integration.py --backend huggingface \
  --model Qwen/Qwen2.5-1.5B-Instruct --download-samples
```

To use a local GGUF model and existing documents with GPU offload instead:

```bash
python tests/run_integration.py --model-path /absolute/path/to/model.gguf \
  --documents ./my_pdfs --gpu-layers 30
```

The integration runner accepts `--context-window`, `--question`, `--timeout` (seconds per run; default 900), and `--skip-reuse`. It prints the underlying command, standard output, errors and elapsed time. A successful run ends with `PASS: end-to-end RAG workflow completed.` A failed test exits nonzero with `FAIL:` and an error. It checks execution and persistence, **not** whether the model's answer is factually correct.

To specifically test the original DoD PDF downloads, use `--official-samples` instead of `--download-samples`; these external URLs may fail with HTTP 403 and are not required to test the RAG pipeline. The synthetic document is a test fixture, **not official DoD safety guidance**. The full integration test can download embedding weights on first use and consumes model RAM/VRAM. It leaves your existing indices untouched; its temporary test index is removed afterward. You do not need Gradio running. Use only public or appropriately authorized documents.

## CLI options

```bash
python chatting_with_your_docs.py --help
```

Important flags: `--backend`, `--model`, `--ollama-url`, `--request-timeout`, `--documents`, `--download-samples`, `--model-path`, `--model-url`, `--embedding-model`, `--gpu-layers`, `--context-window`, `--index-dir`, `--reuse-index`, `--question`, `--gradio`, `--host`, `--port`, `--verbose`.

## How it works

1. Read PDFs from a folder with LlamaIndex's `SimpleDirectoryReader` and PDF reader integration.
2. Embed document passages using `BAAI/bge-small-en-v1.5` by default.
3. Create and persist a `VectorStoreIndex` in `test_index/`.
4. Retrieve matching passages for each question and generate an answer using llama.cpp.

**Index freshness:** `--reuse-index` loads the saved index as-is, even when the files in `--documents` have changed. Omit the flag to rebuild it. If changing embedding models, rebuild the index.

## Notes

- First-time use may download model weights and embedding weights.
- The sample policy PDF URLs are external government resources and may change over time.
- The inference backend can be Ollama, Hugging Face Transformers, or llama.cpp. It does not yet directly connect to an OpenAI-compatible vLLM endpoint.
- The **embedding model** (`BAAI/bge-small-en-v1.5`) is separate from the generation model and is cached through Hugging Face.
- No model weights, downloaded PDFs, or generated indices are tracked in Git.
- Generated answers may contain errors. Check citations and source PDFs before relying on answers.

## Attribution

Adapted from [deptofdefense/LLMs-at-DoD — *Chatting with your Docs*](https://github.com/deptofdefense/LLMs-at-DoD/blob/main/tutorials/Chatting%20with%20your%20Docs.ipynb). See the upstream project for original documentation and applicable licensing. This repository is an independent adaptation and is not an official Department of Defense product.
