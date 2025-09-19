# autot
POC for automated code translation with LLMs and multiple context vector DBs

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
You may not use this work for commercial purposes without permission.
See the [LICENSE](LICENSE) file for details.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

# autot.py — RAG-powered Code Translation (POC)

`autot.py` is a proof-of-concept translator that uses Retrieval-Augmented Generation (RAG) to convert code from a **source language** to a **target language** while staying consistent with what’s already been translated. It maintains three separate context databases:

1) **Source DB** — facts/examples about the source language (idioms, APIs, patterns)
2) **Target DB** — facts/examples about the target language (idioms, APIs, patterns)
3) **Project Consistency DB** — incrementally built memory of what has already been translated (naming, helper utilities, architectural decisions) to keep future translations consistent.

---

## Why this exists (short intro)

Straight LLM translation is easy to start and hard to scale: early files look fine, later ones drift in naming, patterns, and helper abstractions. This POC shows how to use **multiple RAG contexts** to (a) give the model stable grounding in both languages and (b) enforce project-wide consistency that **improves with each translated file**.

---

## Features

- 💡 **Dual-sided grounding:** retrieves source-language and target-language exemplars separately.
- 🔁 **Incremental memory:** updates a project DB with new symbols, helpers, and conventions as you translate files.
- 🧩 **Chunking & metadata:** stores code chunks with language, scope, path, and symbol metadata for precise retrieval.
- 🧪 **POC-level ergonomics:** simple CLI, local vector stores, and minimal dependencies.
- 📝 **Deterministic scaffolding:** optional “header + body + tests” prompt sections to reduce drift.

---

## Requirements

- Python 3.10+
- A local vector store (e.g., Chroma/FAISS) and embeddings model
- Access to an LLM provider (e.g., OpenAI, Azure OpenAI, Anthropic, etc.)

> ⚠️ This README assumes common libs like `chromadb`/`faiss-cpu`, `tiktoken`, and an LLM client. Adjust to your environment if `autot.py` uses different tooling.

---

## Installation

```bash
# 1) Create a virtual env
python -m venv .venv && source .venv/bin/activate

# 2) Install dependencies (adapt if your project uses others)
pip install chromadb faiss-cpu openai tiktoken pydantic rich

# 3) Set your LLM credentials (example for OpenAI)
export OPENAI_API_KEY=sk-...
```

---

## Quick Start

```bash
# Build the language knowledge bases once
python autot.py build-db   --db-src .rag/source_db   --db-tgt .rag/target_db   --src-lang py   --tgt-lang ts   --seed-src ./seeds/python_examples   --seed-tgt ./seeds/typescript_examples

# Translate a single file, updating the project consistency DB as we go
python autot.py translate   --db-src .rag/source_db   --db-tgt .rag/target_db   --db-proj .rag/project_db   --src-lang py   --tgt-lang ts   --in ./examples/src/sample.py   --out ./examples/out/sample.ts
```

> If your script exposes different subcommands/flags, run:
>
> ```bash
> python autot.py -h
> ```
> and map the concepts above to your actual options.

---

## Typical Workflow

1) **Seed language DBs**
   - Point `--seed-src` at high-quality source-language examples (idiomatic snippets, standard library usages).
   - Point `--seed-tgt` at high-quality target-language examples (idiomatic equivalents, standard patterns).

2) **Translate files**
   - For each `--in` file, `autot.py` retrieves:
     - top-K source chunks → “what the original intends”
     - top-K target chunks → “how to do that idiomatically”
     - top-K project chunks → “what we already decided earlier”
   - The LLM generates the target file.
   - The resulting code is parsed and **added to the project DB** to guide subsequent translations.

3) **Iterate**
   - As more files are translated, the **project DB grows**, and naming/conventions stabilize.

---

## CLI (proposed)

> The exact flags may differ in your copy of `autot.py`. This section provides a practical baseline.

### Build/refresh language DBs

```bash
python autot.py build-db   --db-src <dir>   --db-tgt <dir>   --src-lang <id>   --tgt-lang <id>   --seed-src <path or glob>   --seed-tgt <path or glob>   [--embed-model <name>]   [--chunk-size 800] [--chunk-overlap 120]   [--force-rebuild]
```

### Translate files/directories

```bash
python autot.py translate   --db-src <dir>   --db-tgt <dir>   --db-proj <dir>   --src-lang <id>   --tgt-lang <id>   --in <file-or-dir>   --out <file-or-dir>   [--llm <provider:model>]   [--topk-src 6] [--topk-tgt 6] [--topk-proj 8]   [--temperature 0.2]   [--max-tokens 4096]   [--prompt-style strict|balanced|creative]   [--dry-run] [--no-update-proj]
```

### Inspect what was retrieved (debug)

```bash
python autot.py explain   --for <input-file>   --show src,tgt,proj   --n 5
```

---

## How the three DBs are used

```
           ┌───────────┐      ┌───────────┐      ┌─────────────────┐
           │ Source DB │      │ Target DB │      │ Project Consistency│
           └─────┬─────┘      └─────┬─────┘      └──────────┬────────┘
                 │                 │                           │
       retrieve top-K      retrieve top-K             retrieve top-K
                 └───────────────┬─────────────────────────────┘
                                 ▼
                           Prompt Builder
                                 ▼
                           LLM Generation
                                 ▼
                         Target Code Artifact
                                 ▼
                     (ingest into Project DB)
```

- **Source DB** reminds the model *what the original code means* in its native idioms.
- **Target DB** shows *how to express those ideas idiomatically* in the target language.
- **Project DB** enforces *consistency* across files (names, helpers, error handling, logging, test style).

---

## Prompts (recommended structure)

- **System:** translator role, constraints (style, lint rules, error handling).
- **Context blocks:**
  1) Source intent: representative snippets from Source DB.
  2) Target idioms: representative snippets from Target DB.
  3) Project decisions: previously translated helpers/names/tests from Project DB.
- **User:** the actual source file, path, and any rules (e.g., “no dynamic imports”, “prefer async/await”).
- **Output contract:** request compilable code only, followed by a rationale in comments if desired.

---

## Data layout (example)

```
.rag/
  source_db/         # embeddings + metadata for source language
  target_db/         # embeddings + metadata for target language
  project_db/        # grows as you translate
seeds/
  python_examples/   # idiomatic source examples
  typescript_examples/
examples/
  src/               # inputs to translate
  out/               # model outputs
```

---

## Tips for better results

- Seed with **idiomatic** code, not generic snippets.
- Keep chunk size near a **logical scope** (function/class) and include symbol metadata.
- Start with **low temperature**; raise only if you need creative mappings.
- After the first few files, skim the generated helpers and **refactor once**, then re-ingest to stabilize the style.
- Use `--no-update-proj` if you’re experimenting and don’t want to “teach” the project DB yet.

---

## Limitations

- This is a **POC**: no guarantees on performance or perfect determinism.
- Retrieval quality hinges on the **seed corpora** and chunk metadata.
- Large files may require **streaming** or **chunkwise** generation.

---

## Troubleshooting

- **Incoherent outputs** → check seeds; reduce temperature; increase `topk-proj`.
- **Drift across files** → make sure the project DB is being updated and retrieved; inspect with `explain`.
- **Slow retrieval** → consider FAISS with HNSW, or reduce DB size by curating seeds.
- **Token limits** → lower `topk-*`, tighten chunk sizes, or switch to a model with a larger context window.

---

## License

Choose a license for your repo (e.g., MIT/Apache-2.0) and add it here.

---

## A final note

This README maps to the **concepts** implemented in `autot.py`. If your CLI or flags differ, keep the three-DB workflow intact and adapt the command examples to match `python autot.py -h`.

---

*This README was created with the help of generative AI.*
