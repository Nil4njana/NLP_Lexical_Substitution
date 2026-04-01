"""
embedding_module.py  —  Module 2: SBERT Embedding & Similarity
===============================================================
Reads  : tokens.txt, candidates.txt
Writes : embedding_scores.txt  (format: "lemma|cosine_score" per line)

Uses the Sentence-BERT model  'all-MiniLM-L6-v2'  to encode:
  • the original (preprocessed) input sentence
  • every candidate substitution sentence

Then computes cosine similarity between the original embedding
and each candidate embedding.
"""

from sentence_transformers import SentenceTransformer, util

MODEL_NAME = 'all-MiniLM-L6-v2'

# ── File loaders ──────────────────────────────────────────────────────────────

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

# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> list:
    """
    Run Module 2.  Returns list of (lemma, cosine_score) tuples
    and writes embedding_scores.txt.
    """
    original_sentence = load_original_sentence()
    candidates = load_candidates()

    if not candidates:
        print("[Embed] No candidates found in candidates.txt. Skipping.")
        return []

    print(f"[Embed] Original sentence  : {original_sentence}")
    print(f"[Embed] Candidates loaded  : {len(candidates)}")
    print(f"[Embed] Loading SBERT model: {MODEL_NAME}  (downloads on first run)")

    model = SentenceTransformer(MODEL_NAME)

    # Encode original + all candidate sentences in one batch (efficient)
    all_sentences = [original_sentence] + [sent for _, sent in candidates]
    print(f"[Embed] Encoding {len(all_sentences)} sentences...")
    embeddings = model.encode(all_sentences, convert_to_tensor=True,
                               show_progress_bar=False)

    orig_emb  = embeddings[0]
    cand_embs = embeddings[1:]

    scores = []
    for i, (lemma, _) in enumerate(candidates):
        cos_sim = util.cos_sim(orig_emb, cand_embs[i]).item()
        scores.append((lemma, cos_sim))

    with open('embedding_scores.txt', 'w', encoding='utf-8') as f:
        for lemma, score in scores:
            f.write(f"{lemma}|{score:.6f}\n")

    print(f"[Embed] Saved {len(scores)} SBERT scores → embedding_scores.txt")
    return scores


if __name__ == '__main__':
    run()
