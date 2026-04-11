#!/usr/bin/env python3
"""
lexsub_pipeline.py  —  End-to-End Lexical Substitution Pipeline
================================================================
Usage:
    python lexsub_pipeline.py "<sentence>" "<target_word>"

Example:
    python lexsub_pipeline.py \
        "Dr. Ram, the scientist's experiment went out-of-order." \
        "went"

The pipeline:
  PHASE 1  (C binaries, run via subprocess)
    step1_tokeniser   → tokens.txt
    step3_lowercase   → lowercased.txt
    step4_target      → target_index.txt
    step5_pos         → pos_tags.txt + target_pos.txt
    step6_lemma       → lemma.txt

  PHASE 2  (Python modules)
    candidate_generation.py  → candidates.txt
    embedding_module.py      → embedding_scores.txt
    candidate_scoring.py     → output.txt   (final substituted sentence)
"""

import sys
import os
import subprocess
import platform
import textwrap

# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"   {title}")
    print("=" * width)

def section(title: str) -> None:
    print(f"\n[Pipeline] -- {title} --")

def write_input(sentence: str, target_word: str, path: str = 'input.txt', verbose: bool = True) -> None:
    """Write input.txt consumed by all C steps."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(sentence.strip() + '\n')
        f.write(target_word.strip() + '\n')
    if verbose:
        print(f"[Pipeline] input.txt written  (sentence + target word)")

def resolve_binary(name: str) -> str:
    """
    Return the path to a C binary, trying:
      1. bare name   (Unix compiled on Mac/Linux)
      2. name.exe    (Windows / pre-compiled .exe in repo)
    Exits if neither is found.
    """
    candidates = [name, name + '.exe']
    for path in candidates:
        if os.path.isfile(path):
            # Ensure executable bit is set on Unix
            if platform.system() != 'Windows':
                try:
                    os.chmod(path, 0o755)
                except OSError:
                    pass
            return path
    print(f"[Pipeline] ERROR: C binary '{name}' not found in {os.getcwd()!r}. "
          "Please compile it first.")
    sys.exit(1)

def run_c_step(binary_name: str, label: str, verbose: bool = True) -> None:
    """Execute one C preprocessing binary, streaming its stdout."""
    binary = resolve_binary(binary_name)
    cmd = [f'./{binary}'] if platform.system() != 'Windows' else [binary]
    if verbose:
        print(f"[Pipeline] Running {label}  ({binary}) ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if verbose:
        # Print a compact summary: skip blank lines, indent each line
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped:
                print(f"           {stripped}")
    
    if result.returncode != 0:
        print(f"[Pipeline] ERROR in {label}:\n{result.stderr}")
        sys.exit(result.returncode)

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Argument parsing ──────────────────────────────────────────────────────
    if len(sys.argv) < 3:
        print(textwrap.dedent("""
            Usage:
              python lexsub_pipeline.py "<sentence>" "<target_word>" [--embed sbert|lexconsub|xldurel]

            Example:
              python lexsub_pipeline.py \\
                  "Dr. Ram, the scientist's experiment went out-of-order." \\
                  "went" \\
                  --embed xldurel
        """))
        sys.exit(1)

    sentence    = sys.argv[1]
    target_word = sys.argv[2]
    
    embed_method = 'sbert'
    if '--embed' in sys.argv:
        idx = sys.argv.index('--embed')
        if idx + 1 < len(sys.argv):
            embed_method = sys.argv[idx + 1].lower()

    # ── Change to script directory so relative paths work ─────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    banner("LEXICAL SUBSTITUTION PIPELINE")
    print(f"  Sentence    : {sentence}")
    print(f"  Target word : {target_word!r}")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1 : C PREPROCESSING
    # ──────────────────────────────────────────────────────────────────────────
    banner("PHASE 1 — C PREPROCESSING")
    write_input(sentence, target_word)

    c_steps = [
        ('step1_tokeniser', 'Step 1 : Punctuation-Aware Tokenisation'),
        ('step3_lowercase', 'Step 3 : Lowercasing (non-NE tokens)'),
        ('step4_target',    'Step 4 : Target Index Identification'),
        ('step5_pos',       'Step 5 : POS Tagging'),
        ('step6_lemma',     'Step 6 : Lemmatisation'),
    ]
    for binary, label in c_steps:
        run_c_step(binary, label)
    print("\n[Pipeline] Phase 1 complete [DONE]")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2 : PYTHON MODULES
    # ──────────────────────────────────────────────────────────────────────────
    banner("PHASE 2 — PYTHON MODULES")

    # Module 1 — Candidate Generation
    section("Module 1 : Candidate Generation (WordNet + POS filter)")
    from candidate_generation import run as run_cg
    candidates = run_cg()
    if not candidates:
        print("[Pipeline] No candidates could be generated. "
              "Check that the target word has WordNet synonyms for its POS.")
        sys.exit(1)

    # Module 2 — Embedding
    section(f"Module 2 : Embedding Scoring ({embed_method.upper()})")
    from embedding_module import run as run_embed
    run_embed(method=embed_method)

    # Module 3 — Candidate Scoring & Output
    section("Module 3 : Candidate Scoring, Inflection & Output")
    from candidate_scoring import run as run_score
    output_sentence, top_candidates = run_score()

    return output_sentence, top_candidates

def evaluate_single(sentence: str, target_word: str, embed_method: str = 'sbert', verbose: bool = False):
    """
    Programmatic entry point for dataset evaluation.
    Silences stdout outputs natively from Python modules via os.devnull where possible,
    and returns the top candidates.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if verbose:
        banner("PHASE 1 — C PREPROCESSING")
        
    write_input(sentence, target_word, verbose=verbose)

    c_steps = [
        ('step1_tokeniser', 'Step 1 : Punctuation-Aware Tokenisation'),
        ('step3_lowercase', 'Step 3 : Lowercasing (non-NE tokens)'),
        ('step4_target',    'Step 4 : Target Index Identification'),
        ('step5_pos',       'Step 5 : POS Tagging'),
        ('step6_lemma',     'Step 6 : Lemmatisation'),
    ]
    for binary, label in c_steps:
        run_c_step(binary, label, verbose=verbose)

    # We redirect python stdout to null when not verbose to prevent massive console spam
    original_stdout = sys.stdout
    if not verbose:
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')

    try:
        from candidate_generation import run as run_cg
        candidates = run_cg()
        if not candidates:
            return None, []

        from embedding_module import run as run_embed
        run_embed(method=embed_method)

        from candidate_scoring import run as run_score
        output_sentence, top_candidates = run_score()
    finally:
        if not verbose:
            sys.stdout.close()
            sys.stdout = original_stdout

    return output_sentence, top_candidates

    # ── Final summary ─────────────────────────────────────────────────────────
    banner("PIPELINE COMPLETE")
    print(f"  Target word   : '{target_word}'")
    print(f"  Best substitute (inflected): '{top_candidates[0]['lemma']}'  "
          f"(combined score: {top_candidates[0]['combined']:.4f})")
    print(f"  Output sentence: {output_sentence}")
    print(f"\n  Full results saved → output.txt")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
