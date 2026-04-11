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
import os
import urllib.request
import pickle

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

# ── Norvig bigram corpus config ───────────────────────────────────────────────
_DIR         = os.path.dirname(os.path.abspath(__file__))
BIGRAM_TXT   = os.path.join(_DIR, 'count_2w.txt')
BIGRAM_CACHE = os.path.join(_DIR, 'bigram_ctx_cache.pkl')
BIGRAM_URL   = 'https://norvig.com/ngrams/count_2w.txt'

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

def _ensure_bigram_file() -> bool:
    """Download Norvig count_2w.txt to the project directory if not present."""
    if os.path.isfile(BIGRAM_TXT):
        return True
    print(f"\n[Score] Bigram file not found locally.")
    print(f"[Score] Downloading from: {BIGRAM_URL}")
    print(f"[Score] Saving to       : {BIGRAM_TXT}")
    print("[Score] (~300 MB one-time download — subsequent runs use local copy)")
    part = BIGRAM_TXT + '.part'
    try:
        def _hook(blocks, block_size, total):
            if total > 0:
                pct = min(blocks * block_size / total * 100, 100)
                mb  = min(blocks * block_size, total) / 1e6
                print(f"\r[Score]   {pct:5.1f}%  ({mb:.1f} / {total/1e6:.1f} MB)",
                      end='', flush=True)
        urllib.request.urlretrieve(BIGRAM_URL, part, reporthook=_hook)
        os.rename(part, BIGRAM_TXT)
        print(f"\n[Score] Download complete.")
        return True
    except Exception as exc:
        print(f"\n[Score] Download failed: {exc}")
        if os.path.exists(part):
            os.remove(part)
        return False


def _stream_context_bigrams(prev_word, next_word):
    """
    Single streaming pass through count_2w.txt.
    Collects:
      left_counts  — {word: count}  for bigrams  (prev_word, word)
      right_counts — {word: count}  for bigrams  (word, next_word)
    """
    left_counts:  dict = {}
    right_counts: dict = {}
    pw = prev_word.lower() if prev_word else None
    nw = next_word.lower() if next_word else None

    if not os.path.isfile(BIGRAM_TXT):
        print("[Score] WARNING: count_2w.txt missing — bigram scores will be 0.")
        return left_counts, right_counts

    print(f"[Score] Scanning bigrams  (prev='{pw}', next='{nw}') ...",
          end='', flush=True)
    with open(BIGRAM_TXT, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            tab = line.find('\t')
            if tab < 0:
                continue
            sp = line.find(' ')
            if sp < 0 or sp >= tab:
                continue
            w1  = line[:sp].lower()
            w2  = line[sp + 1:tab].lower()
            try:
                cnt = int(line[tab + 1:])
            except ValueError:
                continue
            if pw and w1 == pw:
                left_counts[w2]  = left_counts.get(w2, 0)  + cnt
            if nw and w2 == nw:
                right_counts[w1] = right_counts.get(w1, 0) + cnt
    print(' done.')
    return left_counts, right_counts


def _get_context_bigrams_cached(prev_word, next_word):
    """Return (left_counts, right_counts) from pickle cache or by streaming."""
    cache: dict = {}
    if os.path.isfile(BIGRAM_CACHE):
        try:
            with open(BIGRAM_CACHE, 'rb') as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}
    key = (prev_word, next_word)
    if key not in cache:
        left, right = _stream_context_bigrams(prev_word, next_word)
        cache[key]  = (left, right)
        try:
            with open(BIGRAM_CACHE, 'wb') as f:
                pickle.dump(cache, f)
        except Exception:
            pass  # non-fatal — just won't cache
    return cache[key]


def get_bigram_score(candidate: str,
                     left_counts: dict,
                     right_counts: dict) -> int:
    """
    Contextual bigram score for a candidate.
      score = count(prev_word, candidate)  +  count(candidate, next_word)
    Multi-word candidates (e.g. 'cave in') use first token for left
    and last token for right to approximate the trigram boundary.
    """
    words = candidate.lower().split()
    first, last = words[0], words[-1]
    return left_counts.get(first, 0) + right_counts.get(last, 0)

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

    candidates = list(sbert_scores.keys())

    if not candidates:
        print("[Score] No candidates to score. Was embedding_module.py run?")
        sys.exit(1)

    # ── Step 0: Inflect all candidates upfront ─────────────────────────────────
    # Inflection happens BEFORE scoring so that bigram lookups use the exact
    # surface form that will appear in the sentence (e.g. 'made' not 'make').
    print(f"[Score] Inflecting {len(candidates)} candidates to POS '{ptb_pos}' ...")
    inflected_map = {c: inflect(c, ptb_pos) for c in candidates}

    # ── Bigram context setup ──────────────────────────────────────────────────
    prev_word = tokens[target_index - 1].lower() if target_index > 0 else None
    next_word = tokens[target_index + 1].lower() if target_index < len(tokens) - 1 else None
    _ensure_bigram_file()
    left_counts, right_counts = _get_context_bigrams_cached(prev_word, next_word)

    # ── Step 1 & 2: Raw scores (bigram uses inflected surface form) ───────────
    import math
    sbert_raw  = [sbert_scores[c] for c in candidates]
    bigram_raw = [get_bigram_score(inflected_map[c], left_counts, right_counts)
                  for c in candidates]
    bigram_log = [math.log1p(x) for x in bigram_raw]

    # ── Step 3: Normalise ─────────────────────────────────────────────────────
    sbert_norm  = min_max_normalise(sbert_raw)
    bigram_norm = min_max_normalise(bigram_log)

    # ── Step 4: Weighted combination ──────────────────────────────────────────
    rows = []
    for i, cand in enumerate(candidates):
        combined = alpha * sbert_norm[i] + (1 - alpha) * bigram_norm[i]
        rows.append({
            'lemma':     cand,
            'inflected': inflected_map[cand],
            'combined':  combined,
            'sbert':     sbert_raw[i],
            'ngram':     bigram_raw[i],
        })

    # ── Step 5: Sort → Top-K ──────────────────────────────────────────────────
    rows.sort(key=lambda r: (r['combined'], r['ngram']), reverse=True)
    top = rows[:top_k]

    # ── Step 6: Print ranked table ────────────────────────────────────────────
    BAR = "─" * 64
    print(f"\n[Score] {BAR}")
    print(f"[Score]  TOP-{top_k} RANKED CANDIDATES   "
          f"(α={alpha} × SBERT  +  {1-alpha:.1f} × N-gram)")
    print(f"[Score] {BAR}")
    header = f"  {'Rank':<5} {'Lemma':<15} {'Inflected':<15} {'SBERT':>8} {'N-gram':>10} {'Combined':>10}"
    print(header)
    print(f"  {'----':<5} {'-----':<15} {'---------':<15} {'-----':>8} {'------':>10} {'--------':>10}")
    for rank, r in enumerate(top, 1):
        print(f"  {rank:<5} {r['lemma']:<15} {r['inflected']:<15} "
              f"{r['sbert']:>8.4f} {r['ngram']:>10} {r['combined']:>10.4f}")

    # ── Step 7: Morphological inflection of best candidate ───────────────────
    best_lemma    = top[0]['lemma']
    best_combined = top[0]['combined']
    inflected     = top[0]['inflected']   # already computed before scoring

    print(f"\n[Score] Best candidate lemma   : '{best_lemma}'")
    if inflected != best_lemma:
        print(f"[Score] Inflected to ({ptb_pos})    : '{inflected}'")
    else:
        print(f"[Score] (Inflection unchanged)")

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
    print(f"  Replaced : '{tokens[target_index]}'  ->  '{inflected}'")

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
        f.write(f"  (α={alpha} × SBERT  +  {round(1-alpha,2)} × N-gram bigram)\n")
        f.write(f"\nFull top-{top_k} ranking:\n")
        f.write(f"  {'Rank':<5} {'Lemma':<20} {'Inflected':<20} "
                f"{'SBERT':>8} {'N-gram':>10} {'Combined':>10}\n")
        f.write(f"  {'----':<5} {'-----':<20} {'---------':<20} "
                f"{'-----':>8} {'------':>10} {'--------':>10}\n")
        for rank, r in enumerate(top, 1):
            f.write(f"  {rank:<5} {r['lemma']:<20} {r['inflected']:<20} "
                    f"{r['sbert']:>8.4f} {r['ngram']:>10} {r['combined']:>10.4f}\n")

    print(f"\n[Score] Results saved → output.txt")
    return output_sentence, top


if __name__ == '__main__':
    run()
