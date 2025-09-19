#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
CPP_URLS=(
  "https://en.cppreference.com/w/"
  "https://gcc.gnu.org/onlinedocs/gcc-14.1.0/gcc/"
)
RUST_URLS=(
  "https://doc.rust-lang.org/book/"
  "https://doc.rust-lang.org/std/"
)

CPP_OUT="experiment_1/docs_cpp"
RUST_OUT="experiment_1/docs_rust"
SRC_TXT="src_docs.txt"
TRG_TXT="trg_docs_2.txt"
INPUT_DIR="experiment_1/input_src"   # put your C++ here
MODEL="${MODEL:-deepseek-r1:70b}"    # override with: MODEL=... ./setup_demo.sh

# --- Checks ---
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing $1. Please install it."; exit 1; }; }
need wget
need find
need bash
command -v python >/dev/null 2>&1 || echo "Note: python not in PATH."
command -v lynx >/dev/null 2>&1 || echo "Note: lynx not found; will try pandoc only."
command -v pandoc >/dev/null 2>&1 || echo "Note: pandoc not found; relying on lynx only."
command -v ollama >/dev/null 2>&1 || echo "Note: ollama not found in PATH; translation step may fail."

mkdir -p "$CPP_OUT" "$RUST_OUT" "$INPUT_DIR"

# --- Mirror docs ---
echo "[1/4] Mirroring small doc slices with wget..."
wget_common=( -r -l 2 -k -E -p -np -nH --no-verbose --reject-regex '.*(search|login|\.svg|\.png|\.jpg|\.jpeg|\.gif|\.css|\.js).*' )

echo "  C++ → $CPP_OUT"
for u in "${CPP_URLS[@]}"; do
  wget "${wget_common[@]}" -P "$CPP_OUT" "$u"
done

echo "  Rust → $RUST_OUT"
for u in "${RUST_URLS[@]}"; do
  wget "${wget_common[@]}" -P "$RUST_OUT" "$u"
done

# --- Flatten to text ---
echo "[2/4] Flattening C++ docs → $SRC_TXT"
> "$SRC_TXT"
while IFS= read -r -d '' f; do
  case "$f" in
    *.md|*.txt) cat "$f" ;;
    *) lynx -dump -nolist "$f" 2>/dev/null || pandoc -f html -t plain "$f" 2>/dev/null || true ;;
  esac
  printf "\n\n===== FILE: %s =====\n\n" "$f"
done < <(find "$CPP_OUT" -type f -iregex '.*\.\(html\|htm\|md\|txt\)$' -print0) >> "$SRC_TXT"

echo "[3/4] Flattening Rust docs → $TRG_TXT"
> "$TRG_TXT"
while IFS= read -r -d '' f; do
  case "$f" in
    *.md|*.txt) cat "$f" ;;
    *) lynx -dump -nolist "$f" 2>/dev/null || pandoc -f html -t plain "$f" 2>/dev/null || true ;;
  esac
  printf "\n\n===== FILE: %s =====\n\n" "$f"
done < <(find "$RUST_OUT" -type f -iregex '.*\.\(html\|htm\|md\|txt\)$' -print0) >> "$TRG_TXT"

# --- Optional: run translation if autot.py and input are present ---
echo "[4/4] Running autot.py (if present)"
if [[ -f "autot.py" ]]; then
  if [[ -d "$INPUT_DIR" ]] && [[ -n "$(find "$INPUT_DIR" -type f -name '*.*' -print -quit)" ]]; then
    echo "  Found C++ input under $INPUT_DIR. Launching translation..."
    python autot.py -s "$SRC_TXT" -t "$TRG_TXT" -i "$INPUT_DIR" -m "$MODEL" --src src_db.json --trg trg_db.json -v || true
  else
    echo "  No input files found in $INPUT_DIR; skipping translate step."
  fi
else
  echo "  autot.py not found in current directory; skipping translate step."
fi

echo "Done. Seed corpora: $SRC_TXT / $TRG_TXT"
