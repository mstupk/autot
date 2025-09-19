# Experiment 1 — Demo: Translating **C++ → Rust** with RAG

This mini‑experiment shows how to use `autot.py` to translate C++ code into Rust with **three RAG context DBs**:
- **Source DB**: C++ reference material
- **Target DB**: Rust reference material
- **Project DB**: incrementally built memory of already translated code

You’ll:
1) **Mirror** a small slice of the C++ and Rust documentation with `wget -r`
2) **Flatten** all the content into two plain‑text files (`src_docs.txt`, `trg_docs_2.txt`)
3) **Adapt** `autot.py` for C++ → Rust
4) **Run** the translation on a C++ folder

> ⚠️ **Licensing / robots.txt**: Only mirror the minimum needed for your own local experimentation, respect each website’s terms, and avoid heavy crawling.

---

## 0) Prerequisites

- Linux/macOS shell with `bash`, `wget`, `find`, `xargs`
- **Text extractors**: `lynx` (recommended) and optionally `pandoc`
- Python 3.9+ with packages:
  - `sentence-transformers`, `numpy`, `scikit-learn`, `ollama`
- An **Ollama** model available locally (e.g., `deepseek-r1:70b`, or a smaller one)
- Your C++ sources in a directory (e.g., `experiment_1/input_src/`)

Install tools (Ubuntu/Debian example):
```bash
sudo apt-get update
sudo apt-get install -y wget lynx pandoc
```

Python env:
```bash
python -m venv .venv && source .venv/bin/activate
pip install sentence-transformers numpy scikit-learn ollama
```

---

## 1) Download a *small* slice of docs with `wget -r`

Below are **starter** URLs. Tweak or add more as needed (keep it small).

- **C++ (source docs)** — e.g. a few pages from cppreference and GCC docs:
  - https://en.cppreference.com/w/
  - https://gcc.gnu.org/onlinedocs/gcc-14.1.0/gcc/

- **Rust (target docs)** — e.g. the Rust book and std reference:
  - https://doc.rust-lang.org/book/
  - https://doc.rust-lang.org/std/

**Example command pattern** (limits depth and avoids non‑content pages):
```bash
# C++
mkdir -p experiment_1/docs_cpp
wget -r -l 2 -k -E -p -np -nH --no-verbose   --reject-regex '.*(search|login|\.svg|\.png|\.jpg|\.jpeg|\.gif|\.css|\.js).*'   -P experiment_1/docs_cpp   https://en.cppreference.com/w/   https://gcc.gnu.org/onlinedocs/gcc-14.1.0/gcc/

# Rust
mkdir -p experiment_1/docs_rust
wget -r -l 2 -k -E -p -np -nH --no-verbose   --reject-regex '.*(search|login|\.svg|\.png|\.jpg|\.jpeg|\.gif|\.css|\.js).*'   -P experiment_1/docs_rust   https://doc.rust-lang.org/book/   https://doc.rust-lang.org/std/
```

**Flags explained:**
- `-r` recursive, `-l 2` only two link levels deep (keep it small)
- `-k -E -p` make local pages viewable (`-k` convert links, `-E` add extensions, `-p` get page requisites)
- `-np -nH` don’t go up to parent, don’t create host directories
- `--reject-regex` skips images, CSS/JS, and likely non‑content endpoints

---

## 2) Flatten everything into two `.txt` files

Convert each HTML page to text and concatenate into `src_docs.txt` (C++) and `trg_docs_2.txt` (Rust). Prefer `lynx -dump`, fall back to `pandoc` if needed.

```bash
# C++ → src_docs.txt
find experiment_1/docs_cpp -type f -iregex '.*\.(html|htm|md|txt)$' -print0 | while IFS= read -r -d '' f; do
  case "$f" in
    *.md|*.txt) cat "$f" ;;
    *) lynx -dump -nolist "$f" 2>/dev/null || pandoc -f html -t plain "$f" 2>/dev/null ;;
  esac
  echo -e "\n\n===== FILE: $f =====\n\n"
done > src_docs.txt

# Rust → trg_docs_2.txt
find experiment_1/docs_rust -type f -iregex '.*\.(html|htm|md|txt)$' -print0 | while IFS= read -r -d '' f; do
  case "$f" in
    *.md|*.txt) cat "$f" ;;
    *) lynx -dump -nolist "$f" 2>/dev/null || pandoc -f html -t plain "$f" 2>/dev/null ;;
  esac
  echo -e "\n\n===== FILE: $f =====\n\n"
done > trg_docs_2.txt
```

These two files are the **seed corpora** `autot.py` will embed for the **Source DB** (C++) and **Target DB** (Rust).

---

## 3) Adapt `autot.py` for C++ → Rust

Update three things in your script so the prompt and file scanning match this experiment:

1. **Scanning pattern** (was for Lisp). Make it pick C++ files:
   - Change the recursive matcher from something like `**/*.lisp*` to a C/C++ set, e.g.:
     - `**/*.{c,cc,cxx,cpp,h,hpp}` (pick what you actually use)
2. **Prompt code fences & labels**:
   - Replace output fence from <code>```lisp</code> to <code>```rust</code>.
   - Replace any “Lisp/Common Lisp” wording with “C++/Rust” where it describes source/target roles.
3. **Language hints**:
   - Where the script builds retrieval prompts, ensure the **Source DB** is framed as “C++ idioms/semantics” and the **Target DB** as “Rust idioms/patterns”.

> You don’t need to change the DB filenames: keep `src_docs.txt` and `trg_docs_2.txt` to avoid editing defaults, or pass `-s` / `-t` explicitly.

---

## 4) Run the demo

Put your C++ sources under `experiment_1/input_src/`, then:

```bash
# (Ensure Ollama is running and you pulled a model)
ollama pull deepseek-r1:70b   # or a smaller model you prefer

# Use defaults or pass explicit paths
python autot.py   -s ./src_docs.txt   -t ./trg_docs_2.txt   -i ./experiment_1/input_src   -m deepseek-r1:70b   --src ./src_db.json   --trg ./trg_db.json   -v
```

For each input file, you should see sibling outputs like:
- `foo.cpp.autot` — Rust translation (in ```rust block)
- `foo.cpp.comment` — rationale and notes
- `foo.cpp.think` — (optional) if your model emits it

---

## One‑shot automation

Drop this script as `experiment_1/setup_demo.sh` and run it. It will:
- Verify tools
- Mirror small doc slices
- Flatten them to text
- (Optionally) run `autot.py` if an input folder exists

```bash
./experiment_1/setup_demo.sh
```

---

## Appendix — Tuning `wget`

- Adjust `-l` depth for a bigger/smaller mirror; keep it low to stay polite.
- Add `--quiet` or `--no-verbose` to reduce logs.
- Curate which top‑level paths you mirror to avoid massive downloads.

---

*This README was created with the help of generative AI.*
