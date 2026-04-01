# NLP Lexical Substitution

A hybrid C/Python NLP pipeline for lexical substitution. It takes a target text and a target word, deeply processes the linguistic structure in C (Tokenization, POS tagging, Lemmatization), and then leverages Python with WordNet and SentenceTransformers to find, rank, and correctly inflect the best synonymous word replacements.

## Prerequisites

1. **C Compiler** (`gcc` or `clang` on macOS/Linux/Windows).
2. **Python 3.8+**

## Setup

### 1. Compile the C Modules

The pipeline uses five separate C executables which need to be compiled. Run these commands from the project directory:

**macOS/Linux:**
```bash
gcc -o step1_tokeniser step1_tokeniser.c
gcc -o step3_lowercase step3_lowercase.c
gcc -o step4_target step4_target.c
gcc -o step5_pos step5_pos.c
gcc -o step6_lemma step6_lemma.c
```

**Windows:**
```bash
gcc -o step1_tokeniser.exe step1_tokeniser.c
gcc -o step3_lowercase.exe step3_lowercase.c
gcc -o step4_target.exe step4_target.c
gcc -o step5_pos.exe step5_pos.c
gcc -o step6_lemma.exe step6_lemma.c
```

### 2. Install Python Dependencies

The Python part uses NLTK, spaCy, PyInflect, and SentenceTransformers. Install them by running:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Download Initial NLTK Data (Optional, but recommended)

To prevent the pipeline from downloading datasets on the very first run, you can pre-download WordNet:
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Running the Pipeline

You run the whole pipeline through a single Python script `lexsub_pipeline.py`, providing the sentence and the target word as arguments.

**Usage:**
```bash
python lexsub_pipeline.py "Sentence goes here." "target_word"
```

**Examples:**
```bash
python lexsub_pipeline.py "The thief mugged the old man on the street." "mugged"

python lexsub_pipeline.py "She reads aloud." "aloud"

python lexsub_pipeline.py "She writes neatly." "writes"
```

## How It Works Under The Hood

The script `lexsub_pipeline.py` sequences the data between the modules automatically using text files (`input.txt`, `output.txt`, etc.):

1. **Step 1 (C): Tokenizer** - Splits the raw sentence into clean lowercase/uppercase tokens.
2. **Step 3 (C): Lowercase** - Provides a fully lowercased structure for the word search.
3. **Step 4 (C): Target Identifier** - Finds the target token in the array to give its index.
4. **Step 5 (C): POS Tagger** - Tags the text using Penn Treebank tags (VBZ, NNS, RB, etc.).
5. **Step 6 (C): Lemmatizer** - A C module that bridges to WordNet's `morphy()` to get the uninflected dictionary root of the target word.
6. **CandGen (Py)**: Queries WordNet for synonyms matching the exact Part-Of-Speech, and traverses hypernyms optionally if the synset is sparse.
7. **Embedder (Py)**: SBERT `all-MiniLM-L6-v2` checks cosine similarity context-matching of the candidates substituted directly into the sentence.
8. **Scorer (Py)**: Ranks by `0.7 * Semantics + 0.3 * Frequency` and finally restructures the morphology using `pyinflect` to make the new word grammatically correct inline.
