"""
embedding_word2vec.py  —  Module 2 (alt): Averaged Word Embedding (GloVe)
==========================================================================
Reads  : tokens.txt, candidates.txt
Writes : embedding_scores.txt  (format: "lemma|cosine_score" per line)

Uses pre-trained GloVe embeddings (Stanford, 100-d).
Sentence embedding = average of word vectors for all tokens in the sentence.
Cosine similarity is then computed between the original and each candidate.

Drop-in replacement for embedding_module.py  — same run() interface.

No gensim dependency — loads GloVe text file directly with numpy.
The model auto-downloads on first run (~128 MB) and is cached locally.
"""

import os
import re
import zipfile
import urllib.request
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

GLOVE_URL  = 'https://nlp.stanford.edu/data/glove.6B.zip'
GLOVE_DIM  = 100                      # choices: 50, 100, 200, 300
GLOVE_FILE = f'glove.6B.{GLOVE_DIM}d.txt'
CACHE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.glove_cache')

# ── Lazy model loading ────────────────────────────────────────────────────────

_embeddings = None   # dict: word → np.ndarray

def _download_glove():
    """Download and extract GloVe if not already cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    glove_path = os.path.join(CACHE_DIR, GLOVE_FILE)

    if os.path.isfile(glove_path):
        return glove_path

    zip_path = os.path.join(CACHE_DIR, 'glove.6B.zip')
    if not os.path.isfile(zip_path):
        print(f"[W2V-Embed] Downloading GloVe ({GLOVE_DIM}d) from Stanford...")
        print(f"[W2V-Embed] URL: {GLOVE_URL}")
        print(f"[W2V-Embed] This is ~128 MB — one-time download, cached afterwards.")
        urllib.request.urlretrieve(GLOVE_URL, zip_path,
                                   reporthook=_download_progress)
        print()  # newline after progress bar

    print(f"[W2V-Embed] Extracting {GLOVE_FILE} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extract(GLOVE_FILE, CACHE_DIR)

    return glove_path

def _download_progress(count, block_size, total_size):
    """Progress hook for urllib.request.urlretrieve."""
    pct = count * block_size * 100 // total_size
    mb  = count * block_size / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    print(f"\r[W2V-Embed]   {mb:.1f} / {total_mb:.1f} MB  ({pct}%)", end='', flush=True)

def _load_model():
    """Load GloVe vectors into a dict: word → numpy array."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    glove_path = _download_glove()
    print(f"[W2V-Embed] Loading GloVe vectors from {GLOVE_FILE} ...")

    _embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            word = parts[0]
            vec  = np.array(parts[1:], dtype=np.float32)
            _embeddings[word] = vec

    print(f"[W2V-Embed] Loaded {len(_embeddings)} word vectors ({GLOVE_DIM}-d)")
    return _embeddings

# ── File loaders (same interface as embedding_module.py) ──────────────────────

def reconstruct_sentence(tokens: list) -> str:
    """Join tokens back into readable text, respecting punctuation spacing."""
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

def load_original_sentence(path='tokens.txt') -> str:
    """Reconstruct the original sentence from tokens.txt."""
    with open(path, encoding='utf-8') as f:
        line = f.readline().strip()
    tokens = [t.strip() for t in line.split(',')]
    return reconstruct_sentence(tokens)

def load_candidates(path='candidates.txt') -> list:
    """
    Load (lemma, candidate_sentence) pairs from candidates.txt.
    Each line is formatted as "lemma|sentence".
    """
    candidates = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                lemma, sent = line.split('|', 1)
                candidates.append((lemma.strip(), sent.strip()))
    return candidates

# ── Core: averaged word embedding ─────────────────────────────────────────────

def _tokenize(sentence: str) -> list:
    """Simple whitespace + punctuation-strip tokeniser."""
    raw = sentence.lower().split()
    tokens = [re.sub(r'[^\w\'-]', '', tok) for tok in raw]
    return [t for t in tokens if t]

def sentence_embedding(sentence: str, model: dict) -> np.ndarray:
    """
    Compute averaged GloVe embedding for a sentence.
    Tokens not in the vocabulary are silently skipped.
    Returns a GLOVE_DIM-d vector, or zeros if nothing matches.
    """
    tokens = _tokenize(sentence)
    vectors = []
    for tok in tokens:
        if tok in model:
            vectors.append(model[tok])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(GLOVE_DIM)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ── Entry point (same interface as embedding_module.run) ──────────────────────

def run() -> list:
    """
    Run Module 2 (Word2Vec / GloVe variant).
    Returns list of (lemma, cosine_score) tuples
    and writes embedding_scores.txt.
    """
    original_sentence = load_original_sentence()
    candidates = load_candidates()

    if not candidates:
        print("[W2V-Embed] No candidates found in candidates.txt. Skipping.")
        return []

    print(f"[W2V-Embed] Original sentence  : {original_sentence}")
    print(f"[W2V-Embed] Candidates loaded  : {len(candidates)}")

    model = _load_model()

    print(f"[W2V-Embed] Computing averaged word embeddings for "
          f"{1 + len(candidates)} sentences...")
    orig_emb = sentence_embedding(original_sentence, model)

    scores = []
    for lemma, cand_sent in candidates:
        cand_emb = sentence_embedding(cand_sent, model)
        cos_sim = cosine_similarity(orig_emb, cand_emb)
        scores.append((lemma, cos_sim))

    with open('embedding_scores.txt', 'w', encoding='utf-8') as f:
        for lemma, score in scores:
            f.write(f"{lemma}|{score:.6f}\n")

    print(f"[W2V-Embed] Saved {len(scores)} GloVe scores → embedding_scores.txt")
    return scores


if __name__ == '__main__':
    run()
