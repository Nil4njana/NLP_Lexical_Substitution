"""
embedding_module.py
─────────────────────────────────────────────────────────────────────────────
DROP-IN REPLACEMENT: LexSubCon Similarity Model instead of vanilla SBERT

The LexSubCon similarity model is fine-tuned on top of stsb-roberta-large
(as stated in the ACL 2022 paper) to score how much the substituted sentence
preserves the semantics of the original — which is EXACTLY what your task does.

USAGE in your existing pipeline:
    from embedding_module import LexSubConEmbedder

    embedder = LexSubConEmbedder()
    results  = embedder.rank_substitutes(input_sentence, target_word, candidates)
─────────────────────────────────────────────────────────────────────────────
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import os
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL OPTIONS  (in order of preference for your task)
# ─────────────────────────────────────────────────────────────────────────────
#
#  OPTION 1 [BEST]: LexSubCon pre-trained checkpoint (downloaded from repo)
#     path = "checkpoint/similarity_new_bert/"
#     This is the model fine-tuned specifically for lexical substitution scoring.
#
#  OPTION 2 [GOOD]: stsb-roberta-large (the BASE model LexSubCon was fine-tuned FROM)
#     path = "stsb-roberta-large"
#     Downloads automatically from HuggingFace. Same model family, no custom training.
#
#  OPTION 3 [FALLBACK]: all-mpnet-base-v2 (strong general SBERT)
#     path = "all-mpnet-base-v2"
#     Use if roberta-large is too large for your GPU.
#
# ─────────────────────────────────────────────────────────────────────────────

LEXSUBCON_CHECKPOINT = "checkpoint/similarity_new_bert/"   # ← after git clone + download_4.sh
ROBERTA_LARGE_MODEL  = "stsb-roberta-large"                # ← auto-downloads (~1.3 GB)
FALLBACK_MODEL       = "all-mpnet-base-v2"                 # ← auto-downloads (~420 MB)


class LexSubConEmbedder:
    """
    Embedding module that replicates and extends the LexSubCon similarity signal.

    Replaces your SBERT (all-MiniLM-L6-v2 / all-mpnet) with the model used
    by LexSubCon for its Similarity Score component.

    Parameters
    ----------
    model_path : str
        Path to LexSubCon checkpoint OR a HuggingFace model name.
        Defaults to the LexSubCon checkpoint if available, else stsb-roberta-large.
    device : str
        'cuda' or 'cpu'. Auto-detected if None.
    batch_size : int
        Encoding batch size. Reduce to 8 if GPU OOM on roberta-large.
    normalize : bool
        If True (default), L2-normalizes embeddings so cosine = dot product.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        # Auto-select model
        if model_path is None:
            if os.path.isdir(LEXSUBCON_CHECKPOINT):
                model_path = LEXSUBCON_CHECKPOINT
                print(f"[LexSubConEmbedder] ✅ Using LexSubCon checkpoint: {LEXSUBCON_CHECKPOINT}")
            else:
                model_path = ROBERTA_LARGE_MODEL
                print(f"[LexSubConEmbedder] ℹ️  LexSubCon checkpoint not found.")
                print(f"[LexSubConEmbedder] ✅ Using HuggingFace model: {ROBERTA_LARGE_MODEL}")
                print(f"[LexSubConEmbedder]    To use the actual LexSubCon model, run:")
                print(f"[LexSubConEmbedder]    bash checkpoint/similarity_new_bert/download_4.sh")
        else:
            print(f"[LexSubConEmbedder] ✅ Using model: {model_path}")

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[LexSubConEmbedder] Device: {device}")

        self.model      = SentenceTransformer(model_path, device=device)
        self.batch_size = batch_size
        self.normalize  = normalize
        self.model_path = model_path

    # ──────────────────────────────────────────────────────────────────────
    #  CORE: encode a list of sentences → embeddings matrix
    # ──────────────────────────────────────────────────────────────────────
    def encode(self, sentences: List[str]) -> np.ndarray:
        """
        Encode a list of sentences into embeddings.

        Parameters
        ----------
        sentences : List[str]
            Any list of sentences — original or substituted.

        Returns
        -------
        np.ndarray  shape (N, embedding_dim)
            One embedding per sentence.
        """
        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings

    # ──────────────────────────────────────────────────────────────────────
    #  MAIN TASK: rank substitute sentences by similarity to input
    # ──────────────────────────────────────────────────────────────────────
    def rank_substitutes(
        self,
        input_sentence: str,
        target_word: str,
        candidates: List[str],
        top_n: int = 5,
    ) -> List[Tuple[str, str, float]]:
        """
        Given an input sentence and a list of candidate substitute words,
        builds substituted sentences and ranks them by cosine similarity
        to the original — exactly replicating LexSubCon's Similarity Score signal.

        Parameters
        ----------
        input_sentence : str
            The original sentence containing the target word.
        target_word : str
            The word to be replaced.
        candidates : List[str]
            List of candidate substitute words.
        top_n : int
            How many top results to return.

        Returns
        -------
        List of (substitute_word, substituted_sentence, cosine_score)
            Sorted best → worst.
        """
        # Build substituted sentences
        substituted_sentences = [
            input_sentence.replace(target_word, cand, 1)
            for cand in candidates
        ]

        # Encode input + all substituted sentences in one batched pass (efficient)
        all_sentences   = [input_sentence] + substituted_sentences
        all_embeddings  = self.encode(all_sentences)

        input_emb = all_embeddings[0:1]          # shape (1, dim)
        cand_embs = all_embeddings[1:]            # shape (N, dim)

        # Cosine similarity (= dot product since normalized)
        scores = cosine_similarity(input_emb, cand_embs)[0]   # shape (N,)

        # Zip and sort
        results = list(zip(candidates, substituted_sentences, scores.tolist()))
        results.sort(key=lambda x: x[2], reverse=True)

        return results[:top_n]

    # ──────────────────────────────────────────────────────────────────────
    #  UTILITY: get raw similarity matrix for all pairs
    # ──────────────────────────────────────────────────────────────────────
    def similarity_matrix(
        self,
        input_sentence: str,
        substituted_sentences: List[str],
    ) -> np.ndarray:
        """
        Return a (1 × N) similarity matrix between the input sentence
        and every substituted sentence.

        Use this if you already have the substituted sentences pre-built
        and just need the similarity scores.

        Parameters
        ----------
        input_sentence : str
        substituted_sentences : List[str]

        Returns
        -------
        np.ndarray  shape (1, N)
        """
        all_sentences  = [input_sentence] + substituted_sentences
        all_embeddings = self.encode(all_sentences)
        input_emb      = all_embeddings[0:1]
        subst_embs     = all_embeddings[1:]
        return cosine_similarity(input_emb, subst_embs)   # (1, N)

    # ──────────────────────────────────────────────────────────────────────
    #  UTILITY: compare two single sentences
    # ──────────────────────────────────────────────────────────────────────
    def sentence_pair_score(self, sentence_a: str, sentence_b: str) -> float:
        """Return cosine similarity between a single pair of sentences."""
        embs = self.encode([sentence_a, sentence_b])
        return float(cosine_similarity(embs[0:1], embs[1:2])[0][0])


# ─────────────────────────────────────────────────────────────────────────────
#  SELF-TEST  (run: python embedding_module.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    embedder = LexSubConEmbedder()  # auto-selects best available model

    # ── TEST 1: rank_substitutes ─────────────────────────────────────────
    input_sentence = "The bright child solved the problem quickly."
    target_word    = "bright"
    candidates     = ["smart", "clever", "intelligent", "colorful", "noisy", "tall", "gifted"]

    print("\n" + "="*65)
    print(f"Input  : {input_sentence}")
    print(f"Target : '{target_word}'")
    print(f"{'='*65}")
    print(f"{'Rank':<6}{'Substitute':<18}{'Score':<10}  Substituted Sentence")
    print("-"*65)

    results = embedder.rank_substitutes(input_sentence, target_word, candidates, top_n=5)
    for rank, (sub, sent, score) in enumerate(results, 1):
        print(f"  {rank:<5}{sub:<18}{score:<10.4f}  {sent}")

    # ── TEST 2: similarity_matrix ────────────────────────────────────────
    print("\n── Similarity Matrix ──────────────────────────────────────────")
    substituted = [input_sentence.replace(target_word, c) for c in candidates]
    sim_matrix  = embedder.similarity_matrix(input_sentence, substituted)
    for cand, score in zip(candidates, sim_matrix[0]):
        bar = "█" * int(score * 20)
        print(f"  {cand:<15}  {score:.4f}  {bar}")

    # ── TEST 3: sentence pair ────────────────────────────────────────────
    print("\n── Sentence Pair Score ────────────────────────────────────────")
    s1 = "The bright child solved the problem."
    s2 = "The smart child solved the problem."
    s3 = "The dog sat on a red carpet."
    print(f"  Similar pair : {embedder.sentence_pair_score(s1, s2):.4f}  (should be HIGH)")
    print(f"  Unrelated    : {embedder.sentence_pair_score(s1, s3):.4f}  (should be LOW)")
