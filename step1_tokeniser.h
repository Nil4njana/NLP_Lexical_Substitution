#ifndef STEP1_TOKENIZER_H
#define STEP1_TOKENIZER_H

/* =========================================================
 * STEP 1 : PUNCTUATION-AWARE TOKENIZER
 * ---------------------------------------------------------
 * INPUT  : raw sentence string + target word string
 * OUTPUT : Token array (text, start/end offset, flags)
 * FILE I/O:
 *   reads  -> input.txt   (line1: sentence, line2: target)
 *   writes -> tokens.txt  (TSV: index|token|start|end|is_punct|is_contraction)
 * ========================================================= */

#define MAX_TOKENS      256
#define MAX_TOKEN_LEN    64
#define MAX_SENT_LEN   1024
#define MAX_ABBREVS      32
#define MAX_CONTRACTIONS 20

/* One token produced by the tokenizer */
typedef struct {
    char text[MAX_TOKEN_LEN]; /* token text, original casing           */
    int  start_offset;        /* byte offset of first char in sentence */
    int  end_offset;          /* byte offset just past last char       */
    int  is_punct;            /* 1 = punctuation-only token            */
    int  is_contraction_part; /* 1 = this token came from a contraction*/
} Token;

/* Full output of the tokenizer – passed to every downstream module */
typedef struct {
    Token tokens[MAX_TOKENS];
    int   token_count;
    char  original_sentence[MAX_SENT_LEN];
    char  target_word[MAX_TOKEN_LEN];
} TokenizerOutput;

/* Public API */
int  tokenize(const char *sentence, const char *target_word,
              TokenizerOutput *out);
void save_tokens(const TokenizerOutput *out, const char *filename);
int  load_input (const char *filename,
                 char *sentence, char *target_word);

#endif /* STEP1_TOKENIZER_H */
