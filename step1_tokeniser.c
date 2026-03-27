#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "step1_tokeniser.h"

/* =============================================================
 * ABBREVIATION TABLE
 * Words that end with a period but are NOT sentence boundaries.
 * The period is kept attached to the token (e.g. "Dr." stays "Dr.")
 * ============================================================= */
static const char *ABBREVS[] = {
    "Mr","Mrs","Ms","Dr","Prof","Sr","Jr","St","Ave","Blvd",
    "vs","etc","al","eg","ie","approx","dept","est","govt",
    "U.S","U.K","U.N","Ph.D","M.D","B.Sc","fig","vol",
    NULL   /* sentinel – marks end of table */
};

/* =============================================================
 * CONTRACTION TABLE
 * Maps surface form → two parts
 * e.g. "don't" → "do" + "n't"
 * Both parts are flagged is_contraction_part = 1
 * ============================================================= */
typedef struct {
    const char *surface; /* full contraction as written   */
    const char *p1;      /* first part after the split    */
    const char *p2;      /* second part after the split   */
} Contraction;

static const Contraction CONTRACTIONS[] = {
    {"don't",   "do",    "n't"}, {"doesn't", "does",   "n't"},
    {"didn't",  "did",   "n't"}, {"can't",   "ca",     "n't"},
    {"won't",   "wo",    "n't"}, {"wouldn't","would",  "n't"},
    {"couldn't","could", "n't"}, {"shouldn't","should","n't"},
    {"isn't",   "is",    "n't"}, {"aren't",  "are",    "n't"},
    {"wasn't",  "was",   "n't"}, {"weren't", "were",   "n't"},
    {"haven't", "have",  "n't"}, {"hasn't",  "has",    "n't"},
    {"hadn't",  "had",   "n't"}, {"I'm",     "I",      "'m" },
    {"I've",    "I",     "'ve"}, {"I'll",    "I",      "'ll"},
    {"I'd",     "I",     "'d" }, {"he's",    "he",     "'s" },
    {"she's",   "she",   "'s" }, {"it's",    "it",     "'s" },
    {"that's",  "that",  "'s" }, {"there's", "there",  "'s" },
    {"they're", "they",  "'re"}, {"they've", "they",   "'ve"},
    {"they'll", "they",  "'ll"}, {"we're",   "we",     "'re"},
    {"we've",   "we",    "'ve"}, {"we'll",   "we",     "'ll"},
    {"you're",  "you",   "'re"}, {"you've",  "you",    "'ve"},
    {"you'll",  "you",   "'ll"}, {"let's",   "let",    "'s" },
    {NULL, NULL, NULL}           /* sentinel */
};

/* =============================================================
 * HELPER: case-insensitive string equality
 * Used so "DON'T" and "don't" both match the contraction table
 * ============================================================= */
static int streqi(const char *a, const char *b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b))
            return 0;
        a++; b++;
    }
    return *a == '\0' && *b == '\0';
}

/* =============================================================
 * HELPER: check if buf matches any entry in ABBREVS[]
 * ============================================================= */
static int is_abbreviation(const char *buf) {
    int i;
    for (i = 0; ABBREVS[i] != NULL; i++)
        if (streqi(buf, ABBREVS[i])) return 1;
    return 0;
}

/* =============================================================
 * HELPER: push one completed token into the output array
 * ============================================================= */
static void push_token(TokenizerOutput *out,
                       const char *text,
                       int start, int end,
                       int is_p, int is_c) {
    Token *t;
    if (out->token_count >= MAX_TOKENS) return; /* guard overflow */
    t = &out->tokens[out->token_count++];
    strncpy(t->text, text, MAX_TOKEN_LEN - 1);
    t->text[MAX_TOKEN_LEN - 1] = '\0';
    t->start_offset        = start;
    t->end_offset          = end;
    t->is_punct            = is_p;
    t->is_contraction_part = is_c;
}

/* =============================================================
 * HELPER: try to split buf as a known contraction.
 * Returns 1 and pushes TWO tokens if match found.
 * Returns 0 if buf is not a contraction.
 * ============================================================= */
static int try_split_contraction(TokenizerOutput *out,
                                 const char *buf,
                                 int buf_start) {
    int i, len1;
    for (i = 0; CONTRACTIONS[i].surface != NULL; i++) {
        if (streqi(buf, CONTRACTIONS[i].surface)) {
            len1 = (int)strlen(CONTRACTIONS[i].p1);
            push_token(out, CONTRACTIONS[i].p1,
                       buf_start,
                       buf_start + len1,
                       0, 1);
            push_token(out, CONTRACTIONS[i].p2,
                       buf_start + len1,
                       buf_start + (int)strlen(CONTRACTIONS[i].surface),
                       0, 1);
            return 1;
        }
    }
    return 0;
}

/* =============================================================
 * HELPER: flush the accumulated word buffer as a token.
 * Tries contraction split first; plain token if no match.
 * Resets buf_len to 0 after flush.
 * ============================================================= */
static void flush_word(TokenizerOutput *out,
                       char *buf, int *buf_len, int buf_start) {
    if (*buf_len == 0) return;       /* nothing to flush */
    buf[*buf_len] = '\0';

    if (!try_split_contraction(out, buf, buf_start))
        push_token(out, buf, buf_start, buf_start + *buf_len, 0, 0);

    *buf_len = 0;
}

/* helper: is char a digit? */
static int is_digit(char c) { return c >= '0' && c <= '9'; }

/* =============================================================
 * HELPER: period_should_attach()
 *
 * Called when we see a '.' and buf holds the preceding word.
 * Returns 1 → keep period attached to current token (abbreviation/decimal)
 * Returns 0 → emit period as a separate punctuation token
 *
 * RULES (applied in order):
 *  1. buf is a known abbreviation               → attach  (Dr. Mr. etc.)
 *  2. buf is all-digits AND next char is digit  → decimal (3.14)
 *  3. Next non-space char is uppercase AND
 *     buf length ≥ 3                            → sentence end, split
 *  4. buf length ≤ 2 AND space follows          → short abbrev, attach
 *  5. Default                                   → split
 * ============================================================= */
static int period_should_attach(const char *sentence, int pos,
                                const char *buf) {
    int        k, buf_len;
    char       next_nonspace;
    const char *p;

    buf_len = (int)strlen(buf);

    /* Rule 1 */
    if (is_abbreviation(buf)) return 1;

    /* Rule 2: check if buf is all digits */
    p = buf;
    while (*p) { if (!is_digit(*p)) break; p++; }
    if (*p == '\0') {                /* all digits */
        k = pos + 1;
        while (sentence[k] == ' ') k++;
        if (is_digit(sentence[k])) return 1;
    }

    /* Rule 3 */
    k = pos + 1;
    while (sentence[k] == ' ') k++;
    next_nonspace = sentence[k];
    if (isupper((unsigned char)next_nonspace) && buf_len >= 3) return 0;

    /* Rule 4 */
    if (buf_len <= 2 && sentence[pos + 1] == ' ') return 1;

    /* Rule 5 */
    return 0;
}

/* =============================================================
 * MAIN TOKENIZE FUNCTION
 * ---------------------------------------------------------
 * Single-pass scanner over the sentence string.
 *
 * Character handling:
 *   whitespace / end      → flush word buffer
 *   period '.'            → disambiguation logic (attach or split)
 *   apostrophe "'"        → keep if inside word (contraction), else punct
 *   hyphen '-'            → keep if between letters (compound), else punct
 *   other non-alnum       → flush buffer + emit as standalone punct token
 *   alnum / underscore    → accumulate into word buffer
 * ============================================================= */
int tokenize(const char *sentence, const char *target_word,
             TokenizerOutput *out) {
    char buf[MAX_TOKEN_LEN]; /* current word being built  */
    int  buf_len   = 0;
    int  buf_start = 0;      /* offset where current word started */
    int  i, len;

    memset(out, 0, sizeof(*out));
    strncpy(out->original_sentence, sentence, MAX_SENT_LEN - 1);
    strncpy(out->target_word,       target_word, MAX_TOKEN_LEN - 1);

    len = (int)strlen(sentence);

    for (i = 0; i <= len; i++) {       /* i == len → '\0' triggers final flush */
        char c = sentence[i];

        /* ── WHITESPACE or END-OF-STRING ──────────────────────── */
        if (c == '\0' || c == ' ' || c == '\t' || c == '\n') {
            flush_word(out, buf, &buf_len, buf_start);
            continue;
        }

        /* ── PERIOD ─────────────────────────────────────────────  */
        if (c == '.') {
            if (buf_len > 0) {
                buf[buf_len] = '\0';   /* null-terminate for look-up */
                if (period_should_attach(sentence, i, buf)) {
                    /* attach period to current word (e.g. "Dr.") */
                    if (buf_len < MAX_TOKEN_LEN - 1) buf[buf_len++] = '.';
                } else {
                    /* sentence-ending period: flush word, then emit "." */
                    flush_word(out, buf, &buf_len, buf_start);
                    push_token(out, ".", i, i + 1, 1, 0);
                }
            } else {
                /* period with no preceding word → standalone punct */
                push_token(out, ".", i, i + 1, 1, 0);
            }
            continue;
        }

        /* ── APOSTROPHE ─────────────────────────────────────────  */
        if (c == '\'') {
            /* keep apostrophe only if it sits between word characters */
            if (buf_len > 0 && i + 1 < len &&
                isalpha((unsigned char)sentence[i + 1])) {
                if (buf_len < MAX_TOKEN_LEN - 1) buf[buf_len++] = c;
            } else {
                /* standalone apostrophe / opening quote → punct */
                flush_word(out, buf, &buf_len, buf_start);
                push_token(out, "'", i, i + 1, 1, 0);
            }
            continue;
        }

        /* ── HYPHEN ──────────────────────────────────────────────  */
        if (c == '-') {
            /* keep hyphen inside compound words: well-known, self-aware */
            if (buf_len > 0 && i + 1 < len &&
                isalpha((unsigned char)sentence[i + 1])) {
                if (buf_len < MAX_TOKEN_LEN - 1) buf[buf_len++] = c;
            } else {
                flush_word(out, buf, &buf_len, buf_start);
                push_token(out, "-", i, i + 1, 1, 0);
            }
            continue;
        }

        /* ── ANY OTHER NON-ALPHANUMERIC  ( , ; : ! ? ( ) [ ] " ) ─ */
        if (!isalnum((unsigned char)c) && c != '_') {
            flush_word(out, buf, &buf_len, buf_start);
            { char pstr[2] = {c, '\0'};
              push_token(out, pstr, i, i + 1, 1, 0); }
            continue;
        }

        /* ── ALPHANUMERIC: accumulate into word buffer ───────────  */
        if (buf_len == 0) buf_start = i;   /* start of new word */
        if (buf_len < MAX_TOKEN_LEN - 1) buf[buf_len++] = c;
    }

    return out->token_count;
}

/* =============================================================
 * load_input()
 * Reads input.txt:
 *   Line 1 → sentence
 *   Line 2 → target word
 * ============================================================= */
int load_input(const char *filename,
               char *sentence, char *target_word) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", filename); return 0; }

    if (!fgets(sentence,    MAX_SENT_LEN,   fp)) { fclose(fp); return 0; }
    sentence[strcspn(sentence, "\r\n")] = '\0';  /* strip newline */

    if (!fgets(target_word, MAX_TOKEN_LEN, fp))  { fclose(fp); return 0; }
    target_word[strcspn(target_word, "\r\n")] = '\0';

    fclose(fp);
    return 1;
}

/* =============================================================
 * save_tokens()
 * Writes tokens.txt in TSV format.
 * Columns: INDEX | TOKEN | START | END | IS_PUNCT | IS_CONTRACTION
 * The file is read by every downstream preprocessing step.
 * ============================================================= */
void save_tokens(const TokenizerOutput *out, const char *filename) {
    FILE *fp;
    int   i;

    fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Cannot write %s\n", filename); return; }

    /* Write tokens as a comma-separated list, skip punctuation tokens */
    for (i = 0; i < out->token_count; i++) {
        const Token *t = &out->tokens[i];
        // if (t->is_punct) continue;           /* skip , . ! ? etc. */
        if (i > 0) fprintf(fp, ", ");
        fprintf(fp, "%s", t->text);
    }
    fprintf(fp, "\n");

    fclose(fp);
    printf("Tokens saved to %s\n", filename);
}



/* =============================================================
 * MAIN – Step 1 driver
 * Usage:  ./step1_tokenizer              (uses input.txt)
 *         ./step1_tokenizer myinput.txt
 * ============================================================= */
int main(int argc, char *argv[]) {
    const char     *infile  = (argc > 1) ? argv[1] : "input.txt";
    const char     *outfile = "tokens.txt";
    char            sentence[MAX_SENT_LEN];
    char            target[MAX_TOKEN_LEN];
    TokenizerOutput out;
    int             i;

    if (!load_input(infile, sentence, target)) {
        fprintf(stderr, "Failed to load input from %s\n", infile);
        return 1;
    }

    printf("=== STEP 1: PUNCTUATION-AWARE TOKENIZER ===\n");
    printf("Sentence : %s\n", sentence);
    printf("Target   : %s\n\n", target);

    tokenize(sentence, target, &out);


        /* Print comma-separated token list to console */
    printf("Tokens: ");
    for (i = 0; i < out.token_count; i++) {
        Token *t = &out.tokens[i];
        if (t->is_punct) continue;           /* skip punctuation */
        printf("%s", t->text);
        if (i < out.token_count - 1) printf(", ");
    }
    printf("\n");



    save_tokens(&out, outfile);
    return 0;
}
