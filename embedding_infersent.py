"""
embedding_infersent.py  —  Module 2 (alt): InferSent Sentence Embeddings
==========================================================================
Reads  : tokens.txt, candidates.txt
Writes : embedding_scores.txt  (format: "lemma|cosine_score" per line)

Uses Facebook Research's InferSent (V1, trained on GloVe 840B 300d).
Produces 4096-d sentence embeddings via a BiLSTM trained on NLI data.

Drop-in replacement for embedding_module.py  — same run() interface.

Requirements:
  - PyTorch (already installed with sentence-transformers)
  - NLTK punkt tokenizer
  - GloVe 840B 300d vectors (~2 GB download, one-time)
  - InferSent model weights (~147 MB download, one-time)

All downloads are cached in .infersent_cache/ within the project directory.
"""

import os
import sys
import urllib.request
import zipfile
import numpy as np
import torch
import nltk

# Ensure punkt tokenizer is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(SCRIPT_DIR, '.infersent_cache')

# InferSent V1 uses GloVe 840B 300d
GLOVE_URL    = 'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip'
GLOVE_FILE   = 'glove.840B.300d.txt'
MODEL_URL    = 'https://dl.fbaipublicfiles.com/infersent/infersent1.pkl'
MODEL_FILE   = 'infersent1.pkl'
MODELS_PY    = os.path.join(CACHE_DIR, 'models.py')  # downloaded from repo

INFERSENT_VERSION = 1

# ── Lazy model loading ────────────────────────────────────────────────────────

_infersent = None

def _download_progress(count, block_size, total_size):
    """Progress hook for urllib.request.urlretrieve."""
    pct = min(count * block_size * 100 // max(total_size, 1), 100)
    mb  = count * block_size / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    print(f"\r[InferSent] Downloading: {mb:.1f} / {total_mb:.1f} MB ({pct}%)",
          end='', flush=True)

def _ensure_glove():
    """Download and extract GloVe 840B 300d if not cached."""
    glove_path = os.path.join(CACHE_DIR, GLOVE_FILE)
    if os.path.isfile(glove_path):
        return glove_path

    os.makedirs(CACHE_DIR, exist_ok=True)
    zip_path = os.path.join(CACHE_DIR, 'glove.840B.300d.zip')

    if not os.path.isfile(zip_path):
        print(f"[InferSent] Downloading GloVe 840B 300d vectors (~2 GB)...")
        print(f"[InferSent] This is a one-time download.")
        urllib.request.urlretrieve(GLOVE_URL, zip_path,
                                   reporthook=_download_progress)
        print()  # newline

    print(f"[InferSent] Extracting {GLOVE_FILE} ...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract(GLOVE_FILE, CACHE_DIR)
    except Exception as e:
        print(f"[InferSent] ERROR: Corrupted zip file detected ({e}).")
        print(f"[InferSent] Deleting {zip_path} so a fresh download can be attempted.")
        os.remove(zip_path)
        sys.exit(1)

    return glove_path

def _ensure_model():
    """Download InferSent model weights if not cached."""
    model_path = os.path.join(CACHE_DIR, MODEL_FILE)
    if os.path.isfile(model_path):
        return model_path

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[InferSent] Downloading InferSent V1 model (~147 MB)...")
    urllib.request.urlretrieve(MODEL_URL, model_path,
                               reporthook=_download_progress)
    print()  # newline
    return model_path

def _load_model(sentences):
    """Load InferSent model, set word vectors, and build vocabulary."""
    global _infersent
    if _infersent is not None:
        # Update vocabulary with new sentences
        _infersent.update_vocab(sentences, tokenize=True)
        return _infersent

    # Ensure models.py is available
    if not os.path.isfile(MODELS_PY):
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"[InferSent] Downloading models.py helper from Facebook Research...")
        url = "https://raw.githubusercontent.com/facebookresearch/InferSent/main/models.py"
        try:
            urllib.request.urlretrieve(url, MODELS_PY)
        except Exception as e:
            print(f"[InferSent] ERROR downloading models.py: {e}")
            sys.exit(1)

    # Add cache dir to path so we can import InferSent class
    if CACHE_DIR not in sys.path:
        sys.path.insert(0, CACHE_DIR)
    from models import InferSent

    model_path = _ensure_model()
    glove_path = _ensure_glove()

    print(f"[InferSent] Loading InferSent V{INFERSENT_VERSION} model...")
    params = {
        'bsize': 64,
        'word_emb_dim': 300,
        'enc_lstm_dim': 2048,
        'pool_type': 'max',
        'dpout_model': 0.0,
        'version': INFERSENT_VERSION,
    }
    _infersent = InferSent(params)
    _infersent.load_state_dict(torch.load(model_path,
                                           map_location='cpu',
                                           weights_only=False))
    _infersent.eval()

    print(f"[InferSent] Setting word vector path: {GLOVE_FILE}")
    _infersent.set_w2v_path(glove_path)

    print(f"[InferSent] Building vocabulary from {len(sentences)} sentences...")
    _infersent.build_vocab(sentences, tokenize=True)

    return _infersent

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

# ── Core similarity ───────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ── Entry point (same interface as embedding_module.run) ──────────────────────

def run() -> list:
    """
    Run Module 2 (InferSent variant).
    Returns list of (lemma, cosine_score) tuples
    and writes embedding_scores.txt.
    """
    original_sentence = load_original_sentence()
    candidates = load_candidates()

    if not candidates:
        print("[InferSent] No candidates found in candidates.txt. Skipping.")
        return []

    print(f"[InferSent] Original sentence  : {original_sentence}")
    print(f"[InferSent] Candidates loaded  : {len(candidates)}")

    # Collect all sentences for vocabulary building
    all_sentences = [original_sentence] + [sent for _, sent in candidates]

    # Load model and build vocabulary
    model = _load_model(all_sentences)

    # Encode all sentences at once (batch)
    print(f"[InferSent] Encoding {len(all_sentences)} sentences (4096-d)...")
    embeddings = model.encode(all_sentences, tokenize=True, verbose=False)

    orig_emb  = embeddings[0]
    cand_embs = embeddings[1:]

    scores = []
    for i, (lemma, _) in enumerate(candidates):
        cos_sim = cosine_similarity(orig_emb, cand_embs[i])
        scores.append((lemma, cos_sim))

    with open('embedding_scores.txt', 'w', encoding='utf-8') as f:
        for lemma, score in scores:
            f.write(f"{lemma}|{score:.6f}\n")

    print(f"[InferSent] Saved {len(scores)} InferSent scores → embedding_scores.txt")
    return scores


if __name__ == '__main__':
    run()
