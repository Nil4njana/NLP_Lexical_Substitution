"""
candidate_generation.py  —  Module 1: Candidate Generation
============================================================

UPDATED:
  - Ignores multi-word substitutes
  - Uses ALL synsets
"""

import nltk
from nltk.corpus import wordnet as wn

# Download required data
for pkg in ['wordnet', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

# ── POS Mapping ─────────────────────────────────────────────

PTB_TO_WN = {
    'JJ':  wn.ADJ,  'JJR': wn.ADJ,  'JJS': wn.ADJ,
    'NN':  wn.NOUN, 'NNS': wn.NOUN, 'NNP': wn.NOUN, 'NNPS': wn.NOUN,
    'RB':  wn.ADV,  'RBR': wn.ADV,  'RBS': wn.ADV,
    'VB':  wn.VERB, 'VBD': wn.VERB, 'VBG': wn.VERB,
    'VBN': wn.VERB, 'VBP': wn.VERB, 'VBZ': wn.VERB,
}

# ── File Loaders ────────────────────────────────────────────

def load_lemma(path='lemma.txt') -> str:
    with open(path, encoding='utf-8') as f:
        return f.readline().strip()

def load_target_pos(path='target_pos.txt') -> str:
    with open(path, encoding='utf-8') as f:
        return f.readline().strip()

def load_target_index(path='target_index.txt') -> int:
    with open(path, encoding='utf-8') as f:
        return int(f.readline().strip())

def load_tokens(path='tokens.txt') -> list:
    with open(path, encoding='utf-8') as f:
        line = f.readline().strip()
    return [t.strip() for t in line.split(',')]

# ── POS Conversion ──────────────────────────────────────────

def get_wn_pos(ptb_tag: str):
    return PTB_TO_WN.get(ptb_tag.upper())

# ── UPDATED SYNONYM FUNCTION (MULTI-WORD SUPPORT) ───────────

def get_synonyms(lemma: str, wn_pos) -> list:
    """
    Get ALL substitutes (including multi-word) from WordNet.
    """

    synsets = wn.synsets(lemma)

    candidates = set()

    for synset in synsets:
        if synset.pos() != wn_pos:
            continue

        for l in synset.lemmas():
            name = l.name().lower()

            # ignore multi-word substitutes
            if '_' in name:
                continue

            # skip original word
            if name == lemma.lower():
                continue

            candidates.add(name)

    return sorted(candidates)

# ── Sentence Reconstruction ─────────────────────────────────

def reconstruct_sentence(tokens: list) -> str:
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

# ── Generate Candidate Sentences ────────────────────────────

def generate_candidate_sentences(tokens: list, target_index: int,
                                 candidates: list) -> list:
    results = []

    for cand in candidates:
        new_tokens = tokens[:]

        # handle multi-word substitution
        cand_tokens = cand.split()

        new_tokens = (
            tokens[:target_index] +
            cand_tokens +
            tokens[target_index + 1:]
        )

        sentence = reconstruct_sentence(new_tokens)
        results.append((cand, sentence))

    return results

# ── MAIN FUNCTION ───────────────────────────────────────────

def run() -> list:
    lemma        = load_lemma()
    ptb_pos      = load_target_pos()
    target_index = load_target_index()
    tokens       = load_tokens()

    wn_pos = get_wn_pos(ptb_pos)
    if wn_pos is None:
        print(f"[CandGen] ERROR: POS '{ptb_pos}' not supported.")
        return []

    print(f"[CandGen] Target lemma : '{lemma}'")
    print(f"[CandGen] POS          : '{ptb_pos}' -> '{wn_pos}'")

    candidates = get_synonyms(lemma, wn_pos)

    print(f"[CandGen] Total candidates: {len(candidates)}")
    print(f"[CandGen] Sample: {candidates[:10]}")

    if not candidates:
        print("[CandGen] No candidates found.")
        return []

    candidate_sentences = generate_candidate_sentences(
        tokens, target_index, candidates
    )

    with open('candidates.txt', 'w', encoding='utf-8') as f:
        for cand, sent in candidate_sentences:
            f.write(f"{cand}|{sent}\n")

    print(f"[CandGen] Saved {len(candidate_sentences)} candidates -> candidates.txt")

    return candidate_sentences

# ── ENTRY POINT ─────────────────────────────────────────────

if __name__ == '__main__':
    run()