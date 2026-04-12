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

def run(method: str = 'sbert') -> list:
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
    print(f"[Embed] Using Embedding Method: {method.upper()}")

    scores = []
    substituted_sentences = [sent for _, sent in candidates]
    cand_lemmas = [lemma for lemma, _ in candidates]

    if method == 'sbert':
        global _sbert_model
        from sentence_transformers import SentenceTransformer, util
        if '_sbert_model' not in globals() or _sbert_model is None:
            print(f"[Embed] Loading SBERT model: {MODEL_NAME}  (downloads on first run)")
            _sbert_model = SentenceTransformer(MODEL_NAME)
        model = _sbert_model
        
        all_sentences = [original_sentence] + substituted_sentences
        print(f"[Embed] Encoding {len(all_sentences)} sentences...")
        embeddings = model.encode(all_sentences, convert_to_tensor=True,
                                   show_progress_bar=False)
        orig_emb  = embeddings[0]
        cand_embs = embeddings[1:]
        
        for i, lemma in enumerate(cand_lemmas):
            cos_sim = util.cos_sim(orig_emb, cand_embs[i]).item()
            scores.append((lemma, cos_sim))
            
    elif method == 'glove':
        import embedding_glove
        print(f"[Embed] Computing averaged word embeddings (GloVe 100d) for "
              f"{len(candidates)} candidates...")
        scores = embedding_glove.run()
        
    elif method == 'infersent':
        import embedding_infersent
        print(f"[Embed] Computing BiLSTM sentence embeddings (InferSent) for "
              f"{len(candidates)} candidates...")
        scores = embedding_infersent.run()
            
    elif method == 'xldurel':
        from embedding_module_xldurel import xl_durel_score
        
        with open('input.txt', 'r', encoding='utf-8') as f:
            target_word = f.readlines()[1].strip()
            
        print(f"[Embed] Encoding 1 original + {len(cand_lemmas)} candidates with XL-DURel...")
        
        results = xl_durel_score(original_sentence, target_word, cand_lemmas)
        
        # xl_durel_score returns a list of (candidate, score) tuples in the same order
        for item in results:
            # item is (candidate, score)
            scores.append(item)
            
    else:
        print(f"[Embed] Unknown embedding method: {method}")
        return []

    with open('embedding_scores.txt', 'w', encoding='utf-8') as f:
        for lemma, score in scores:
            f.write(f"{lemma}|{score:.6f}\n")

    print(f"[Embed] Saved {len(scores)} {method.upper()} scores -> embedding_scores.txt")
    return scores


if __name__ == '__main__':
    run()
