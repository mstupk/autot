# autot
POC for automated RAG code translation with LLMs and multiple context vector DBs

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
You may not use this work for commercial purposes without permission.
See the [LICENSE](LICENSE) file for details.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

# autot.py — RAG‑assisted Lisp → modern Common Lisp translator (POC)

`autot.py` is a proof‑of‑concept that translates legacy **Lisp** code into **modern Common Lisp** using Retrieval‑Augmented Generation (RAG) with **three context databases**:

1) **Source DB** — examples & context from *source‑language* documentation
2) **Target DB** — examples & context from *target‑language* documentation (modern Common Lisp idioms)
3) **Done/Project DB** — an incrementally built memory of already processed files to keep later translations consistent

The script prepares/loads the first two DBs from documentation files, uses all three to build a contextual prompt, calls a local **Ollama** model for generation, and writes per‑file outputs alongside your sources.

---

## Requirements

- Python 3.9+ (recommended)
- Python packages:
  - `sentence-transformers` (embeddings, model: `all-mpnet-base-v2`)
  - `numpy`
  - `scikit-learn` (imported for pairwise utilities)
  - `ollama` (Python client)
- A local **Ollama** daemon running and a chat model available (default: `deepseek-r1:70b`).

> **Note:** `deepseek-r1:70b` is a very large model; you can override it with `-m <model>` (e.g., a smaller model you’ve already pulled).

```bash
# Setup (example)
python -m venv .venv && source .venv/bin/activate
pip install sentence-transformers numpy scikit-learn ollama

# Make sure Ollama is installed and running locally, then pull a model:
ollama pull deepseek-r1:70b
# …or choose a smaller alternative you have available, then use -m to select it.
```

---

## Inputs & Data Files

- **Documentation seeds** (plain text files)
  - `--src-docs` (default: `./src_docs.txt`) — describes legacy/source idioms
  - `--trg-docs` (default: `./trg_docs_2.txt`) — describes modern/target idioms

  These are parsed into **(code, context)** pairs and **text chunks**. Code is detected heuristically (Lisp forms), and the surrounding prose is used as context.

- **Context DBs (JSON, auto‑built/loaded)**
  - `--src` (default: `src_db.json`) — saved embeddings + samples derived from `--src-docs`
  - `--trg` (default: `trg_db.json`) — saved embeddings + samples derived from `--trg-docs`

  On first run (if the JSON files don’t exist), the script builds them from the docs and saves them. Subsequent runs load them.

- **Project “done” DB (JSON)**
  - `done_db.json` — updated after successful translations; stores normalized source snippets and embeddings to help keep future translations consistent.

---

## What it processes and what it writes

- **Scans** the input directory recursively for files matching `**/*.lisp*`.
- **Per input file**, produces up to three sibling outputs:
  - `<file>.autot` — the translated Common Lisp (from the model’s ```lisp block)
  - `<file>.comment` — human‑readable explanations (from the ```comments block)
  - `<file>.think` — optional internal reasoning if the model emitted it (from `<think>…</think>` or ```think)
- Also writes helper logs:
  - `pathlist.txt` — all matched input files (for visibility)
  - `processed_files.txt` — an append‑only ledger to avoid re‑processing the same paths in later runs

---

## CLI usage

```text
usage: autot.py [-h] [-s SRC_DOCS] [-t TRG_DOCS] [-m MODEL] [-i INPUT_DIR] [-v] [--src SRC] [--trg TRG]

Translate Lisp code to modern Common Lisp using RAG + Ollama.

options:
  -h, --help            show this help message and exit
  -s, --src-docs SRC_DOCS
                        Path to source language docs (default: ./src_docs.txt)
  -t, --trg-docs TRG_DOCS
                        Path to target language docs (default: ./trg_docs_2.txt)
  -m, --model MODEL     Ollama model to use (default: deepseek-r1:70b)
  -i, --input-dir INPUT_DIR
                        Directory of .lisp* files to translate (default: ./symbolics/sys.sct)
  -v, --verbose         Print LLM output to STDOUT in realtime as it is generated
      --src SRC         Path to save/load the SOURCE context DB (JSON) (default: src_db.json)
      --trg TRG         Path to save/load the TARGET context DB (JSON) (default: trg_db.json)
```

> **Tip:** The default `--input-dir` must be a directory; point it to the root of your Lisp sources.

---

## Quick start

```bash
# 1) Ensure Ollama is running and you’ve pulled a suitable model
ollama run deepseek-r1:70b  # (or another model)

# 2) Prepare (or provide) your docs; defaults shown here
ls ./src_docs.txt ./trg_docs_2.txt

# 3) Run translation on your project directory
python autot.py   -s ./src_docs.txt   -t ./trg_docs_2.txt   -i ./my_legacy_lisp_sources   -m deepseek-r1:70b   --src ./src_db.json   --trg ./trg_db.json   -v
```

On first run, the script will build `src_db.json` and `trg_db.json` from your docs, then translate all `**/*.lisp*` under `-i`. For each file it will create `.autot`, `.comment`, and (if present) `.think` outputs next to the source file.

---

## How the three DBs shape the prompt

- **Source DB**: Helps interpret what the legacy Lisp code intends.
- **Target DB**: Shows how to express those ideas in modern Common Lisp.
- **Done DB**: Adds the most recent translated samples to the prompt to keep naming and helper patterns consistent as the project progresses.

> Internally, the script embeds code and prose with `all-mpnet-base-v2`. The prompt includes a small number of examples from each DB and instructs the model to output code in a ```lisp block plus explanations in a ```comments block.

---

## Troubleshooting

- **Ollama connection errors**: Ensure the daemon is running (`ollama serve`) and the `-m` model is pulled.
- **Model too large / OOM**: Choose and pull a smaller model and pass it via `-m`.
- **No files found**: Verify `-i` points to a directory that actually contains `*.lisp` files.
- **DB rebuild**: Delete `src_db.json`/`trg_db.json` to force a rebuild from docs.
- **Cold start on embeddings**: The first run may download `all-mpnet-base-v2` for `sentence-transformers`.

---

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
You may not use this work for commercial purposes without permission.
See the [LICENSE](LICENSE) file for details.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

---

*This README was created with the help of generative AI.*
