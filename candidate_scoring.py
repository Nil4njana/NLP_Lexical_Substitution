"""
candidate_scoring.py  —  Module 3: Candidate Scoring & Output
==============================================================
Reads  : embedding_scores.txt, target_pos.txt, target_index.txt,
         tokens.txt, lemma.txt
Writes : output.txt

Steps:
  1. Load SBERT cosine scores from embedding_scores.txt
  2. Compute WordNet frequency scores (proxy for corpus frequency)
  3. Min-max normalise both score sets to [0, 1]
  4. Combine: final = α × sbert_norm + (1−α) × freq_norm   (default α=0.7)
  5. Sort descending → Top-K candidates
  6. Morphological inflection of best lemma → target form  (via pyinflect)
  7. Replace target token in original sentence → output sentence
  8. Save output.txt + print ranked table
"""

import sys
import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ── Optional inflection (pyinflect + spaCy) ───────────────────────────────────
try:
    import spacy
    import pyinflect          # registers Token._.inflect via spaCy extension
    _nlp = spacy.load('en_core_web_sm')
    INFLECT_OK = True
except Exception as _e:
    INFLECT_OK = False
    _nlp = None
    print(f"[Score] NOTE: pyinflect/spaCy unavailable ({_e}). "
          "Best substitute will be used in base (lemma) form.")

# ── Penn Treebank → WordNet POS mapping ──────────────────────────────────────
PTB_TO_WN = {
    'JJ':  wn.ADJ,  'JJR': wn.ADJ,  'JJS': wn.ADJ,
    'NN':  wn.NOUN, 'NNS': wn.NOUN, 'NNP': wn.NOUN, 'NNPS': wn.NOUN,
    'RB':  wn.ADV,  'RBR': wn.ADV,  'RBS': wn.ADV,
    'VB':  wn.VERB, 'VBD': wn.VERB, 'VBG': wn.VERB,
    'VBN': wn.VERB, 'VBP': wn.VERB, 'VBZ': wn.VERB,
}

# ── File loaders ──────────────────────────────────────────────────────────────

def load_embedding_scores(path='embedding_scores.txt') -> dict:
    scores = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                lemma, score = line.split('|', 1)
                scores[lemma.strip()] = float(score.strip())
    return scores

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

def load_target_lemma(path='lemma.txt') -> str:
    with open(path, encoding='utf-8') as f:
        return f.readline().strip()

# ── Scoring helpers ───────────────────────────────────────────────────────────

def get_wn_freq(lemma: str, wn_pos) -> int:
    """
    Sum WordNet corpus frequency counts for all lemma objects that match
    the given surface form across synsets with the target POS.
    """
    total = 0
    if wn_pos is None:
        return total
    for synset in wn.synsets(lemma, pos=wn_pos):
        for l in synset.lemmas():
            if l.name().lower() == lemma.lower():
                total += l.count()
    return total

def min_max_normalise(values: list) -> list:
    """Scale values to [0, 1].  If all values are equal, return 0.5 for each."""
    if not values:
        return values
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]

# ── Morphological inflection ──────────────────────────────────────────────────

def inflect(lemma: str, ptb_tag: str) -> str:
    """
    Inflect a base-form lemma to match the grammatical tag of the target word.
    Falls back to the bare lemma if pyinflect is unavailable or fails.
    """
    if not INFLECT_OK or _nlp is None:
        return lemma
    try:
        doc = _nlp(lemma)
        inflected = doc[0]._.inflect(ptb_tag)
        return inflected if inflected else lemma
    except Exception:
        return lemma

# ── Sentence reconstruction ───────────────────────────────────────────────────

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

# ── Entry point ───────────────────────────────────────────────────────────────

def run(alpha: float = 0.7, top_k: int = 5):
    """
    Run Module 3.
    alpha  : weight for SBERT score  (1-alpha goes to WordNet freq score)
    top_k  : number of top candidates to report
    Returns: (output_sentence, top_candidates_list)
    """
    sbert_scores  = load_embedding_scores()
    ptb_pos       = load_target_pos()
    target_index  = load_target_index()
    tokens        = load_tokens()
    target_lemma  = load_target_lemma()         # for reference only

    wn_pos = PTB_TO_WN.get(ptb_pos.upper())
    candidates = list(sbert_scores.keys())

    if not candidates:
        print("[Score] No candidates to score. Was embedding_module.py run?")
        sys.exit(1)

    # ── Step 1 & 2: Raw scores ────────────────────────────────────────────────
    sbert_raw = [sbert_scores[c] for c in candidates]
    freq_raw  = [get_wn_freq(c, wn_pos) for c in candidates]

    # ── Step 3: Normalise ─────────────────────────────────────────────────────
    sbert_norm = min_max_normalise(sbert_raw)
    freq_norm  = min_max_normalise(freq_raw)

    # ── Step 4: Weighted combination ──────────────────────────────────────────
    rows = []
    for i, cand in enumerate(candidates):
        combined = alpha * sbert_norm[i] + (1 - alpha) * freq_norm[i]
        rows.append({
            'lemma':    cand,
            'combined': combined,
            'sbert':    sbert_raw[i],
            'freq':     freq_raw[i],
        })

    # ── Step 5: Sort → Top-K ──────────────────────────────────────────────────
    rows.sort(key=lambda r: (r['combined'], r['freq']), reverse=True)
    top = rows[:top_k]

    # ── Step 6: Print ranked table ────────────────────────────────────────────
    BAR = "─" * 64
    print(f"\n[Score] {BAR}")
    print(f"[Score]  TOP-{top_k} RANKED CANDIDATES   "
          f"(α={alpha} × SBERT  +  {1-alpha:.1f} × FreqScore)")
    print(f"[Score] {BAR}")
    header = f"  {'Rank':<5} {'Candidate':<20} {'SBERT':>8} {'Freq':>6} {'Combined':>10}"
    print(header)
    print(f"  {'----':<5} {'---------':<20} {'-----':>8} {'----':>6} {'--------':>10}")
    for rank, r in enumerate(top, 1):
        print(f"  {rank:<5} {r['lemma']:<20} "
              f"{r['sbert']:>8.4f} {r['freq']:>6} {r['combined']:>10.4f}")

    # ── Step 7: Morphological inflection of best candidate ───────────────────
    best_lemma    = top[0]['lemma']
    best_combined = top[0]['combined']
    inflected     = inflect(best_lemma, ptb_pos)

    print(f"\n[Score] Best candidate lemma   : '{best_lemma}'")
    if inflected != best_lemma:
        print(f"[Score] Inflected to ({ptb_pos})    : '{inflected}'")
    else:
        print(f"[Score] (Inflection unchanged / pyinflect not applied)")

    # ── Step 8: Reconstruct output sentence ──────────────────────────────────
    original_sentence = reconstruct_sentence(tokens)
    new_tokens = tokens[:]
    new_tokens[target_index] = inflected
    output_sentence = reconstruct_sentence(new_tokens)

    print(f"\n[Score] {'═'*56}")
    print(f"[Score]  FINAL OUTPUT")
    print(f"[Score] {'═'*56}")
    print(f"  Original : {original_sentence}")
    print(f"  Output   : {output_sentence}")
    print(f"  Replaced : '{tokens[target_index]}'  →  '{inflected}'")

    # ── Write output.txt ──────────────────────────────────────────────────────
    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("   LEXICAL SUBSTITUTION OUTPUT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Original sentence : {original_sentence}\n")
        f.write(f"Output sentence   : {output_sentence}\n\n")
        f.write(f"Target word       : {tokens[target_index]}\n")
        f.write(f"Target lemma      : {target_lemma}\n")
        f.write(f"Target POS        : {ptb_pos}\n")
        f.write(f"Best substitute   : {inflected}  (lemma: {best_lemma})\n")
        f.write(f"Combined score    : {best_combined:.4f}\n")
        f.write(f"  (α={alpha} × SBERT  +  {round(1-alpha,2)} × WordNet frequency)\n")
        f.write(f"\nFull top-{top_k} ranking:\n")
        f.write(f"  {'Rank':<5} {'Lemma':<20} {'Inflected':<20} "
                f"{'SBERT':>8} {'Freq':>6} {'Combined':>10}\n")
        f.write(f"  {'----':<5} {'-----':<20} {'---------':<20} "
                f"{'-----':>8} {'----':>6} {'--------':>10}\n")
        for rank, r in enumerate(top, 1):
            inf = inflect(r['lemma'], ptb_pos)
            f.write(f"  {rank:<5} {r['lemma']:<20} {inf:<20} "
                    f"{r['sbert']:>8.4f} {r['freq']:>6} {r['combined']:>10.4f}\n")

    print(f"\n[Score] Results saved → output.txt")
    return output_sentence, top


if __name__ == '__main__':
    run()
