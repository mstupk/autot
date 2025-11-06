import os
import re
import glob
import ollama
import numpy as np
import json
import argparse
import sys
import hashlib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity  # kept to preserve functionality, even if unused

# --- New: LangChain prompt imports (robust to different installs) ---
try:
    from langchain_core.prompts import PromptTemplate
except Exception:
    from langchain.prompts import PromptTemplate

# --- New: PDF and HTML processing imports ---
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyPDF2 not installed. PDF files will be skipped.")

try:
    from bs4 import BeautifulSoup
    HTML_SUPPORT = True
except ImportError:
    HTML_SUPPORT = False
    print("Warning: BeautifulSoup not installed. HTML files will be skipped.")


class LispTranslationRAG:
    def __init__(self, src_docs_path, trg_docs_path, ollama_model='deepseek-r1:70b'):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.ollama = ollama_model
        self.src_docs_path = src_docs_path
        self.trg_docs_path = trg_docs_path

        # Initialize databases with context-aware storage
        self.src_db = {
            'embeddings': np.zeros((0, 768)),
            'samples': [],  # (code, context) tuples
            'text_embeddings': np.zeros((0, 768)),
            'text_chunks': []
        }
        self.trg_db = {
            'embeddings': np.zeros((0, 768)),
            'samples': [],  # (code, context) tuples
            'text_embeddings': np.zeros((0, 768)),
            'text_chunks': []
        }
        self.done_db = {
            'embeddings': np.zeros((0, 768)),
            'samples': [],  # (code, context) tuples (here: normalized source snippets)
            'text_embeddings': np.zeros((0, 768)),
            'text_chunks': [],
            'filepaths': []
        }
        self.translation_cache = {}

    # --- New: File type detection and content extraction ---
    def _extract_text_from_file(self, filepath):
        """Extract text content from various file types (TXT, HTML, PDF)"""
        ext = os.path.splitext(filepath)[1].lower()

        try:
            if ext == '.txt':
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()

            elif ext == '.html' and HTML_SUPPORT:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    return soup.get_text()

            elif ext == '.pdf' and PDF_SUPPORT:
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    return text

            else:
                print(f"Warning: Unsupported file type or missing dependency: {filepath}")
                return ""

        except Exception as e:
            print(f"Error reading file {filepath}: {str(e)}")
            return ""

    def _process_directory(self, dir_path):
        """Process all supported files in a directory and return combined content"""
        if not os.path.exists(dir_path):
            print(f"Error: Directory {dir_path} does not exist")
            return ""

        all_content = []

        # Process all supported file types
        for pattern in ['*.txt', '*.html', '*.pdf']:
            for filepath in glob.glob(os.path.join(dir_path, '**', pattern), recursive=True):
                print(f"Processing: {filepath}")
                content = self._extract_text_from_file(filepath)
                if content.strip():
                    all_content.append(f"\n--- Content from {os.path.basename(filepath)} ---\n")
                    all_content.append(content)

        # Also check if the path itself is a file
        if os.path.isfile(dir_path):
            content = self._extract_text_from_file(dir_path)
            if content.strip():
                all_content.append(content)

        return "\n".join(all_content)

    # --- Persistence for context DBs --------------------------------------

    def _save_db(self, db, path):
        """Save a context DB (src/trg) to JSON."""
        try:
            data = {
                'embeddings': db['embeddings'].tolist() if isinstance(db['embeddings'], np.ndarray) else db['embeddings'],
                'samples': db['samples'],
                'text_embeddings': db['text_embeddings'].tolist() if isinstance(db['text_embeddings'], np.ndarray) else db['text_embeddings'],
                'text_chunks': db['text_chunks'],
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: Could not save DB to {path} - {str(e)}")

    def _load_db(self, path):
        """Load a context DB (src/trg) from JSON. Returns a dict or None on failure."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            db = {
                'embeddings': np.array(data.get('embeddings', [])),
                'samples': data.get('samples', []),
                'text_embeddings': np.array(data.get('text_embeddings', [])),
                'text_chunks': data.get('text_chunks', []),
            }
            # Shape fixes if empty
            if db['embeddings'].size == 0:
                db['embeddings'] = np.zeros((0, 768))
            if db['text_embeddings'].size == 0:
                db['text_embeddings'] = np.zeros((0, 768))
            return db
        except Exception as e:
            print(f"Warning: Could not load DB from {path} - {str(e)}")
            return None

    def prepare_context_dbs(self, src_db_path, trg_db_path):
        """Load existing context DBs if present; otherwise build from docs and save to the given paths."""
        # Source DB
        if os.path.exists(src_db_path):
            loaded = self._load_db(src_db_path)
            if loaded:
                self.src_db = loaded
                print(f"Loaded source context DB from: {src_db_path}")
            else:
                print(f"Rebuilding source DB from docs due to load failure.")
                self._build_enhanced_database(self.src_docs_path, self.src_db)
                self._save_db(self.src_db, src_db_path)
                print(f"Saved new source context DB to: {src_db_path}")
        else:
            print(f"Building source context DB from docs (no existing DB at {src_db_path})...")
            self._build_enhanced_database(self.src_docs_path, self.src_db)
            self._save_db(self.src_db, src_db_path)
            print(f"Saved source context DB to: {src_db_path}")

        # Target DB
        if os.path.exists(trg_db_path):
            loaded = self._load_db(trg_db_path)
            if loaded:
                self.trg_db = loaded
                print(f"Loaded target context DB from: {trg_db_path}")
            else:
                print(f"Rebuilding target DB from docs due to load failure.")
                self._build_enhanced_database(self.trg_docs_path, self.trg_db)
                self._save_db(self.trg_db, trg_db_path)
                print(f"Saved new target context DB to: {trg_db_path}")
        else:
            print(f"Building target context DB from docs (no existing DB at {trg_db_path})...")
            self._build_enhanced_database(self.trg_docs_path, self.trg_db)
            self._save_db(self.trg_db, trg_db_path)
            print(f"Saved target context DB to: {trg_db_path}")

        print("\nContext DB Status:")
        print(f"Source: {len(self.src_db['samples'])} code-context pairs, {len(self.src_db['text_chunks'])} text chunks")
        print(f"Target: {len(self.trg_db['samples'])} code-context pairs, {len(self.trg_db['text_chunks'])} text chunks")

    # --- Doc processing ----------------------------------------------------

    def _extract_code_context_pairs(self, text):
        """Extract (code, context) pairs from documentation"""
        sections = re.split(r'\n\s*\n', text)
        pairs = []
        for section in sections:
            code_blocks = re.findall(r'(?:^|\n)(?:;+\s*Example:?\s*)?(\(.*?\))(?=\n|$)', section, re.DOTALL)
            if code_blocks:
                context = re.sub(r'(\(.*?\))', '', section)
                context = ' '.join(context.split()).strip()
                for code in code_blocks:
                    if code.strip():
                        pairs.append((code.strip(), context))
        return pairs

    def _process_doc_content(self, content, source_name="document"):
        """Process documentation content into (code, context) pairs and text chunks"""
        code_context_pairs = self._extract_code_context_pairs(content)
        text_chunks = [
            chunk.strip() for chunk in
            re.split(r'\n\s*\n', re.sub(r'\(.*?\)', '', content))
            if chunk.strip() and len(chunk.split()) > 5
        ]
        return code_context_pairs, text_chunks

    def _build_enhanced_database(self, doc_path, db):
        """Build database with both code-context pairs and text chunks from files or directories"""
        # Clear the database first
        db['embeddings'] = np.zeros((0, 768))
        db['samples'] = []
        db['text_embeddings'] = np.zeros((0, 768))
        db['text_chunks'] = []

        # Extract content from the path (file or directory)
        if os.path.isdir(doc_path):
            content = self._process_directory(doc_path)
        else:
            content = self._extract_text_from_file(doc_path)

        if not content.strip():
            print(f"Warning: No content extracted from {doc_path}")
            return

        code_context_pairs, text_chunks = self._process_doc_content(content, doc_path)

        # Process code-context pairs
        for code, context in code_context_pairs:
            try:
                code_embedding = self.model.encode(code).reshape(1, -1)
                context_embedding = self.model.encode(context).reshape(1, -1)

                if db['embeddings'].shape[0] == 0:
                    db['embeddings'] = code_embedding
                else:
                    db['embeddings'] = np.vstack([db['embeddings'], code_embedding])
                db['samples'].append((code, context))

                if db['text_embeddings'].shape[0] == 0:
                    db['text_embeddings'] = context_embedding
                else:
                    db['text_embeddings'] = np.vstack([db['text_embeddings'], context_embedding])
                db['text_chunks'].append(context)
            except Exception as e:
                print(f"Error processing sample: {str(e)}")

        # Process additional text chunks
        for chunk in text_chunks:
            try:
                chunk_embedding = self.model.encode(chunk).reshape(1, -1)
                if db['text_embeddings'].shape[0] == 0:
                    db['text_embeddings'] = chunk_embedding
                else:
                    db['text_embeddings'] = np.vstack([db['text_embeddings'], chunk_embedding])
                db['text_chunks'].append(chunk)
            except Exception as e:
                print(f"Error processing text chunk: {str(e)}")

    # --- Done DB persistence ----------------------------------------------

    def _load_done_db(self):
        """Load the done database from file if it exists"""
        if os.path.exists('done_db.json'):
            try:
                with open('done_db.json', 'r') as f:
                    data = json.load(f)
                    self.done_db['embeddings'] = np.array(data.get('embeddings', []))
                    self.done_db['samples'] = data.get('samples', [])
                    self.done_db['filepaths'] = data.get('filepaths', [])
                    if self.done_db['embeddings'].size == 0:
                        self.done_db['embeddings'] = np.zeros((0, 768))
            except Exception as e:
                print(f"Warning: Could not load done_db - {str(e)}")

    def _save_done_db(self):
        """Save the done database to file"""
        try:
            with open('done_db.json', 'w') as f:
                json.dump({
                    'embeddings': self.done_db['embeddings'].tolist(),
                    'samples': self.done_db['samples'],
                    'filepaths': self.done_db['filepaths']
                }, f)
        except Exception as e:
            print(f"Warning: Could not save done_db - {str(e)}")

    def _update_done_db(self, filepath, source_code):
        """Update the done database with a new translation"""
        try:
            processed_code = self._preprocess_code(source_code)
            embedding = self.model.encode(processed_code).reshape(1, -1)

            if self.done_db['embeddings'].shape[0] == 0:
                self.done_db['embeddings'] = embedding
            else:
                self.done_db['embeddings'] = np.vstack([
                    self.done_db['embeddings'],
                    embedding
                ])

            self.done_db['samples'].append(processed_code)
            self.done_db['filepaths'].append(filepath)
            self._save_done_db()
        except Exception as e:
            print(f"Warning: Could not update done_db - {str(e)}")

    # --- Utilities ---------------------------------------------------------

    def _preprocess_code(self, code):
        """Normalize code for consistent processing"""
        code = re.sub(r';.*', '', code)  # Remove Lisp line comments
        code = re.sub(r'\s+', ' ', code).strip()  # Normalize whitespace
        return code

    def _extract_code_block(self, text):
        """Extract code from markdown block labeled as lisp; fallback to whole text if not found."""
        match = re.search(r'```lisp\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    def _extract_comments_block(self, text):
        """Extract comments/explanations from a ```comments block."""
        match = re.search(r'```comments\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else "No explanations provided"

    def _extract_think_block(self, text):
        """Extract a 'think' block if present. Supports <think>...</think> or ```think blocks."""
        m1 = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
        if m1:
            return m1.group(1).strip()
        m2 = re.search(r'```think\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
        if m2:
            return m2.group(1).strip()
        return None

    def _write_output(self, path, content):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    # --- Prompt builder (LangChain) ---------------------------------------

    def _generate_contextual_prompt_with_langchain(self, source_code):
        """
        Generate the final prompt using LangChain PromptTemplate.
        Pulls up to 3 examples from each DB (source, target, done) just like before.
        """
        # Example snippet templates
        example_tmpl = PromptTemplate(
            input_variables=["context", "code"],
            template="Context: {context}\nCode: {code}"
        )
        prev_tmpl = PromptTemplate(
            input_variables=["snippet"],
            template="{snippet}"
        )

        # Render Source Examples (up to 3)
        src_examples = []
        for code, ctx in self.src_db['samples'][:3]:
            src_examples.append(example_tmpl.format(context=ctx, code=code))
        src_block = "\n\n".join(src_examples) if src_examples else "None"

        # Render Target Examples (up to 3)
        trg_examples = []
        for code, ctx in self.trg_db['samples'][:3]:
            trg_examples.append(example_tmpl.format(context=ctx, code=code))
        trg_block = "\n\n".join(trg_examples) if trg_examples else "None"

        # Render Previous Translations (up to 3 most recent)
        prev_examples = []
        if self.done_db['samples']:
            for snippet in self.done_db['samples'][-3:]:
                prev_examples.append(prev_tmpl.format(snippet=snippet))
        prev_block = "\n\n".join(prev_examples) if prev_examples else "None"

        # Overall instruction template
        overall_tmpl = PromptTemplate(
            input_variables=["source_examples", "target_examples", "previous_translations", "code_to_translate"],
            template=(
                "Translate this Lisp code to modern Common Lisp while preserving all functionality.\n"
                "The first knowledge source provided shall help you understand what this lisp code actually does, "
                "the second language source describes the target implementation, so adhere to it in your answers. "
                "The third knowledge source represents what you have done so far: You must always remain consistent to it!\n\n"
                "While translating this Common Lisp code, always proceed step by step: "
                "What is the expected input? What does the source code you shall translate do? "
                "How do you preserve all its functionality using the target implementation?\n\n"
                "Source Examples:\n{source_examples}\n\n"
                "Target Examples:\n{target_examples}\n\n"
                "Previous Translations:\n{previous_translations}\n\n"
                "Code to translate:\n{code_to_translate}\n\n"
                "Provide the translated code in a ```lisp block and explanations in a ```comments block. "
                "If you include chain-of-thought or hidden reasoning, wrap it in <think>...</think> (or a ```think block)."
            )
        )

        return overall_tmpl.format(
            source_examples=src_block,
            target_examples=trg_block,
            previous_translations=prev_block,
            code_to_translate=source_code
        )

    # --- Translation methods ----------------------------------------------

    def translate_file(self, input_path, verbose=False):
        """Enhanced translation that updates done_db.
        If verbose=True, stream tokens to STDOUT in realtime; only write files after full output is generated.
        """
        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()

            file_hash = hashlib.md5(source_code.encode()).hexdigest()
            if file_hash in self.translation_cache:
                translated_code, comments, think = self.translation_cache[file_hash]
                base_path = os.path.splitext(input_path)[0]
                self._write_output(f"{base_path}.autot", translated_code)
                self._write_output(f"{base_path}.comment", comments)
                if think:
                    self._write_output(f"{base_path}.think", think)
                return translated_code, comments

            # --- Build prompt with LangChain ---
            prompt = self._generate_contextual_prompt_with_langchain(source_code)

            # --- Generate with/without streaming ---
            full_output = ""
            if verbose:
                stream = ollama.generate(
                    model=self.ollama,
                    prompt=prompt,
                    options={'temperature': 0.1, 'num_ctx': 4096},
                    stream=True
                )
                for chunk in stream:
                    part = chunk.get('response', '')
                    if part:
                        full_output += part
                        print(part, end='', flush=True)
                print()
            else:
                resp = ollama.generate(
                    model=self.ollama,
                    prompt=prompt,
                    options={'temperature': 0.1, 'num_ctx': 4096}
                )
                full_output = resp.get('response', '')

            # --- After full output is available, split into files ---
            translated_code = self._extract_code_block(full_output)
            comments = self._extract_comments_block(full_output)
            think = self._extract_think_block(full_output)

            base_path = os.path.splitext(input_path)[0]
            self._write_output(f"{base_path}.autot", translated_code)
            self._write_output(f"{base_path}.comment", comments)
            if think:
                self._write_output(f"{base_path}.think", think)

            # Update done_db after successful translation
            self._update_done_db(input_path, source_code)

            self.translation_cache[file_hash] = (translated_code, comments, think)
            return translated_code, comments

        except Exception as e:
            print(f"Failed to translate {input_path}: {str(e)}")
            return None, str(e)

    def translate_directory(self, input_dir, verbose=False):
        """Process directory with all three databases"""
        if not os.path.exists(input_dir):
            print(f"Error: Input directory {input_dir} does not exist")
            return

        # First collect all files
        all_files = []
        for filepath in glob.glob(os.path.join(input_dir, '**/*.lisp*'), recursive=True):
            all_files.append(filepath)

        # Write to pathlist.txt
        with open("pathlist.txt", "w") as f1:
            for filepath in all_files:
                f1.write(f"{filepath}\n")

        # Process files
        processed_files = set()
        if os.path.exists("processed_files.txt"):
            with open("processed_files.txt", "r") as f3:
                processed_files = {line.strip() for line in f3 if line.strip()}

        with open("processed_files.txt", "a") as f3:
            for path in all_files:
                if path not in processed_files:
                    print(f"\nTranslating: {path}")
                    translated, comments = self.translate_file(path, verbose=verbose)
                    if translated:
                        print(f"Successfully translated to: {os.path.splitext(path)[0]}.autot")
                        print(f"Comments saved to: {os.path.splitext(path)[0]}.comment")
                        print(f"Think saved to: {os.path.splitext(path)[0]}.think (if provided)")
                        f3.write(f"{path}\n")
                        f3.flush()
                        processed_files.add(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Translate Lisp code to modern Common Lisp using RAG + Ollama.")
    # Docs for building context on first run
    parser.add_argument('-s', '--src-docs', default='./src_docs.txt', help='Path to source language docs (file or directory containing TXT, HTML, PDF files)')
    parser.add_argument('-t', '--trg-docs', default='./trg_docs_2.txt', help='Path to target language docs (file or directory containing TXT, HTML, PDF files)')
    # Model & input
    parser.add_argument('-m', '--model', default='deepseek-r1:70b', help='Ollama model to use')
    parser.add_argument('-i', '--input-dir', default='./symbolics/sys.sct', help='Directory of .lisp* files to translate')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print LLM output to STDOUT in realtime as it is generated')
    # Context DB file paths
    parser.add_argument('--src', default='src_db.json', help='Path to save/load the SOURCE context DB (JSON)')
    parser.add_argument('--trg', default='trg_db.json', help='Path to save/load the TARGET context DB (JSON)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    translator = LispTranslationRAG(
        src_docs_path=args.src_docs,
        trg_docs_path=args.trg_docs,
        ollama_model=args.model
    )

    translator.prepare_context_dbs(args.src, args.trg)
    translator._load_done_db()  # optional: load done DB if present

    translator.translate_directory(args.input_dir, verbose=args.verbose)
