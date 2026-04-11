from WordTransformer import WordTransformer, InputExample
from sklearn.metrics.pairwise import cosine_similarity

# ── Load ONCE at module level ──────────────────────────────────────
model = None

def get_positions(sentence: str, word: str):
    """Get character-level [start, end] of word in sentence."""
    try:
        start = sentence.index(word)
        return [start, start + len(word)]
    except ValueError:
        return None

def xl_durel_score(input_sentence: str,
                   target_word: str,
                   candidates: list) -> list:
    global model
    if model is None:
        print("[Embed] Loading XL-DURel model (sachinn1/xl-durel)...")
        model = WordTransformer("sachinn1/xl-durel")

    # ── Step 1: embed target word in original sentence ─────────────
    orig_pos = get_positions(input_sentence, target_word)
    if orig_pos is None:
        raise ValueError(f"Target word '{target_word}' not in sentence.")

    orig_example = InputExample(texts=input_sentence, positions=orig_pos)
    orig_emb     = model.encode(orig_example).reshape(1, -1)  # (1, 1024)

    # ── Step 2: for each candidate, build substituted sentence ─────
    results = []
    for candidate in candidates:
        subst_sentence = input_sentence.replace(target_word, candidate, 1)

        cand_pos = get_positions(subst_sentence, candidate)
        if cand_pos is None:
            results.append((candidate, 0.0))
            continue

        cand_example = InputExample(texts=subst_sentence, positions=cand_pos)
        cand_emb     = model.encode(cand_example).reshape(1, -1)  # (1, 1024)

        score = cosine_similarity(orig_emb, cand_emb)[0][0]
        results.append((candidate, round(float(score), 4)))

    # ── Step 3: We don't rank here because embedding_module.py expects
    # raw scores aligned to the candidates list to save to embedding_scores.txt
    return results
