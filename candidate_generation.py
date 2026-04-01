"""
candidate_generation.py  —  Module 1: Candidate Generation
============================================================
Reads  : lemma.txt, target_pos.txt, target_index.txt, tokens.txt
Writes : candidates.txt  (format: "lemma|candidate_sentence" per line)

Steps:
  1. WordNet synonym extraction  (for target lemma + POS)
  2. POS-based filtering         (keep only same-POS synonyms)
  3. Generate candidate sentences (replace target token with each synonym)
"""

import os
import nltk
from nltk.corpus import wordnet as wn

# Download required NLTK data silently
for pkg in ['wordnet', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

# ── Penn Treebank → WordNet POS mapping ──────────────────────────────────────
PTB_TO_WN = {
    'JJ':  wn.ADJ,  'JJR': wn.ADJ,  'JJS': wn.ADJ,
    'NN':  wn.NOUN, 'NNS': wn.NOUN, 'NNP': wn.NOUN, 'NNPS': wn.NOUN,
    'RB':  wn.ADV,  'RBR': wn.ADV,  'RBS': wn.ADV,
    'VB':  wn.VERB, 'VBD': wn.VERB, 'VBG': wn.VERB,
    'VBN': wn.VERB, 'VBP': wn.VERB, 'VBZ': wn.VERB,
}

# ── File loaders ──────────────────────────────────────────────────────────────

def load_lemma(path='lemma.txt') -> str:
    """Read target lemma from line 1 of lemma.txt."""
    with open(path, encoding='utf-8') as f:
        return f.readline().strip()

def load_target_pos(path='target_pos.txt') -> str:
    """Read PTB POS tag from line 1 of target_pos.txt."""
    with open(path, encoding='utf-8') as f:
        return f.readline().strip()

def load_target_index(path='target_index.txt') -> int:
    """Read target token index from line 1 of target_index.txt."""
    with open(path, encoding='utf-8') as f:
        return int(f.readline().strip())

def load_tokens(path='tokens.txt') -> list:
    """Read comma-separated token list from tokens.txt."""
    with open(path, encoding='utf-8') as f:
        line = f.readline().strip()
    return [t.strip() for t in line.split(',')]

# ── Core helpers ──────────────────────────────────────────────────────────────

def get_wn_pos(ptb_tag: str):
    """Map a Penn Treebank POS tag to a WordNet POS constant, or None."""
    return PTB_TO_WN.get(ptb_tag.upper())

def _collect_lemmas(synsets, exclude: str, wn_pos) -> set:
    """
    Collect single-word lemma names from a list of synsets,
    excluding the target lemma itself and multi-word expressions.
    """
    found = set()
    for synset in synsets:
        # Only use synsets that match the target POS
        if synset.pos() != wn_pos and wn_pos != wn.ADJ:
            continue
        for l in synset.lemmas():
            name = l.name()
            if '_' in name or ' ' in name:   # skip multi-word
                continue
            normalised = name.lower()
            if normalised != exclude.lower():
                found.add(normalised)
    return found

def get_synonyms(lemma: str, wn_pos) -> list:
    """
    Extract single-word synonyms from WordNet for the given lemma + POS.

    Strategy (applied in order until at least one candidate is found):
      1. Direct synset lemmas          — same synset as the target word
      2. Hypernym synset lemmas        — one level up (more general terms)
      3. Second-level hypernym lemmas  — two levels up
      4. Sister synset lemmas          — co-hyponyms sharing the same hypernym

    This handles words like 'mug' (VBD) which have a synset but only one
    lemma (itself), giving 0 direct synonyms at level 1.
    """
    target_synsets = wn.synsets(lemma, pos=wn_pos)
    print(target_synsets)
    print(wn.synonyms(lemma))

    # ── Level 1: direct synset lemmas ────────────────────────────────────────
    synonyms = _collect_lemmas(target_synsets, lemma, wn_pos)
    if synonyms:
        return sorted(synonyms)

    # ── Level 2: hypernym lemmas (1 hop up) ──────────────────────────────────
    hypernym1 = []
    for s in target_synsets:
        hypernym1.extend(s.hypernyms())
    synonyms = _collect_lemmas(hypernym1, lemma, wn_pos)
    if synonyms:
        print(f"[CandGen] No direct synonyms; using hypernym lemmas "
              f"({[h.name() for h in hypernym1[:3]]})")
        return sorted(synonyms)

    # ── Level 3: hypernym lemmas (2 hops up) ─────────────────────────────────
    hypernym2 = []
    for s in hypernym1:
        hypernym2.extend(s.hypernyms())
    synonyms = _collect_lemmas(hypernym2, lemma, wn_pos)
    if synonyms:
        print(f"[CandGen] No direct/hypernym-1 synonyms; using level-2 hypernym lemmas.")
        return sorted(synonyms)

    # ── Level 4: sister synsets (co-hyponyms of hypernym) ────────────────────
    sisters = []
    for h in hypernym1:
        sisters.extend(h.hyponyms())
    synonyms = _collect_lemmas(sisters, lemma, wn_pos)
    if synonyms:
        print(f"[CandGen] Using sister synset lemmas (co-hyponyms of hypernym).")
        return sorted(synonyms)

    return []  # genuinely no related words found


def reconstruct_sentence(tokens: list) -> str:
    """
    Join tokens back into a readable sentence.
    Punctuation tokens are attached to the preceding token (no leading space).
    """
    PUNCT_NO_SPACE_BEFORE = {',', '.', '!', '?', ';', ':', ')', ']', '}'}
    PUNCT_NO_SPACE_AFTER  = {'(', '[', '{'}

    result = ''
    for i, tok in enumerate(tokens):
        if tok in PUNCT_NO_SPACE_BEFORE:
            result += tok
        elif i > 0 and tokens[i - 1] in PUNCT_NO_SPACE_AFTER:
            result += tok
        else:
            result = (result + ' ' + tok) if result else tok
    return result

def generate_candidate_sentences(tokens: list, target_index: int,
                                  candidates: list) -> list:
    """
    For every candidate lemma, swap it into the token list at target_index
    and reconstruct the sentence.
    Returns list of (lemma, candidate_sentence) tuples.
    """
    results = []
    for cand in candidates:
        new_tokens = tokens[:]
        new_tokens[target_index] = cand
        sentence = reconstruct_sentence(new_tokens)
        results.append((cand, sentence))
    return results

# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> list:
    """
    Run Module 1.  Returns list of (lemma, sentence) tuples,
    and writes candidates.txt.
    """
    lemma        = load_lemma()
    ptb_pos      = load_target_pos()
    target_index = load_target_index()
    tokens       = load_tokens()

    wn_pos = get_wn_pos(ptb_pos)
    if wn_pos is None:
        print(f"[CandGen] WARNING: PTB POS '{ptb_pos}' has no WordNet mapping. "
              "Cannot generate candidates.")
        return []

    print(f"[CandGen] Target lemma : '{lemma}'")
    print(f"[CandGen] PTB POS      : '{ptb_pos}'  →  WordNet POS: '{wn_pos}'")

    candidates = get_synonyms(lemma, wn_pos)
    preview = candidates[:10]
    ellipsis = '...' if len(candidates) > 10 else ''
    print(f"[CandGen] Synonyms found : {len(candidates)}  "
          f"({', '.join(preview)}{ellipsis})")

    if not candidates:
        print("[CandGen] No valid candidates after filtering. Exiting.")
        return []

    candidate_sentences = generate_candidate_sentences(tokens, target_index,
                                                        candidates)

    with open('candidates.txt', 'w', encoding='utf-8') as f:
        for cand, sent in candidate_sentences:
            f.write(f"{cand}|{sent}\n")

    print(f"[CandGen] Saved {len(candidate_sentences)} candidates → candidates.txt")
    return candidate_sentences


if __name__ == '__main__':
    run()
