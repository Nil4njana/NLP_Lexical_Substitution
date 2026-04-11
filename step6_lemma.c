/*
 * ============================================================
 * STEP 6: LEMMATIZATION  (WordNet morphy via Python / NLTK)
 * ============================================================
 * APPROACH:
 *   Delegates to wn.morphy(word, pos) — the morphological
 *   analyser built into NLTK's WordNet interface.
 *   morphy() uses WordNet's own exception files (verb.exc,
 *   noun.exc, ...) together with suffix-stripping rules, so
 *   it handles ALL inflected forms correctly:
 *       promised → promise    (not 'promis')
 *       walked   → walk
 *       went     → go         (irregular, from verb.exc)
 *       tried    → try
 *       better   → good       (irregular adjective)
 *
 * HOW IT WORKS:
 *   1. Read target word  from lowercased.txt + target_index.txt
 *   2. Read target POS   from target_pos.txt
 *   3. Map PTB POS tag   → WordNet POS short code (v/n/a/r)
 *   4. Write a tiny Python helper script to _morphy_tmp.py
 *   5. Run it with system("python3 _morphy_tmp.py > _morphy_out.txt")
 *   6. Read the result back from _morphy_out.txt
 *   7. Clean up temp files and write lemma.txt
 *
 * INPUT  : lowercased.txt   — comma-separated lowercase tokens
 *          target_index.txt — integer index of the target token
 *          target_pos.txt   — PTB POS tag of the target token
 *
 * OUTPUT : lemma.txt        — lemma (base form) of target word
 * ============================================================
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_LEN    64
#define MAX_LINE 4096
#define MAX_TOKENS 512

/* ── Helper: null-terminated strncpy ─────────────────────── */
static void safe_copy(char *dst, const char *src, int maxlen) {
    strncpy(dst, src, maxlen - 1);
    dst[maxlen - 1] = '\0';
}

/* ── PTB POS → WordNet POS short code ────────────────────── */
static const char *ptb_to_wn(const char *ptb) {
    if (strncmp(ptb, "VB", 2) == 0) return "v";  /* all verb tags */
    if (strncmp(ptb, "NN", 2) == 0) return "n";  /* all noun tags */
    if (strncmp(ptb, "JJ", 2) == 0) return "a";  /* all adj  tags */
    if (strncmp(ptb, "RB", 2) == 0) return "r";  /* all adv  tags */
    return NULL;                                  /* no mapping    */
}

/* ── Load comma-separated token line into array ──────────── */
static int load_tokens(const char *fname,
                        char arr[][MAX_LEN], int max) {
    FILE *fp = fopen(fname, "r");
    if (!fp) return -1;
    char line[MAX_LINE];
    if (!fgets(line, MAX_LINE, fp)) { fclose(fp); return 0; }
    fclose(fp);
    line[strcspn(line, "\r\n")] = '\0';
    int n = 0;
    char *tok = strtok(line, ",");
    while (tok && n < max) {
        while (*tok == ' ') tok++;
        safe_copy(arr[n++], tok, MAX_LEN);
        tok = strtok(NULL, ",");
    }
    return n;
}

/*
 * ── morphy_via_python ─────────────────────────────────────
 * Calls NLTK's wn.morphy(word, wn_pos) through a temp Python
 * script.  Returns 1 + fills `out` on success, 0 on failure.
 *
 * The script also tries the adjective-satellite pos ('s') as
 * a fallback for adjectives, matching NLTK morphy behaviour.
 */
static int morphy_via_python(const char *word, const char *wn_pos,
                              char *out, int outlen) {

    /* ── Write temporary Python helper ───────────────────── */
    FILE *script = fopen("_morphy_tmp.py", "w");
    if (!script) return 0;

    fprintf(script,
        "import sys\n"
        "try:\n"
        "    import nltk\n"
        "    nltk.download('wordnet', quiet=True)\n"
        "    nltk.download('omw-1.4', quiet=True)\n"
        "    from nltk.corpus import wordnet as wn\n"
        "    result = wn.morphy('%s', '%s')\n"          /* primary POS   */
        "    if result is None and '%s' == 'a':\n"      /* adj satellite */
        "        result = wn.morphy('%s', 's')\n"
        "    print(result if result else '%s')\n"       /* fallback      */
        "except Exception as e:\n"
        "    sys.stderr.write('morphy error: ' + str(e) + '\\n')\n"
        "    print('%s')\n",                            /* error fallback */
        word, wn_pos,   /* args 1-2: wn.morphy('%s', '%s')          */
        wn_pos,         /* arg  3  : if '%s' == 'a' satellite check  */
        word,           /* arg  4  : wn.morphy('%s', 's') satellite  */
        word,           /* arg  5  : print(result else '%s') fallback */
        word            /* arg  6  : except block print('%s') fallback */
    );
    fclose(script);

    /* ── Execute ──────────────────────────────────────────── */
    int rc = system("python _morphy_tmp.py > _morphy_out.txt 2>NUL");

    /* ── Read result ──────────────────────────────────────── */
    int success = 0;
    if (rc == 0 || 1) {   /* check file even if rc != 0 */
        FILE *res = fopen("_morphy_out.txt", "r");
        if (res) {
            if (fgets(out, outlen, res)) {
                out[strcspn(out, "\r\n")] = '\0';
                if (strlen(out) > 0) success = 1;
            }
            fclose(res);
        }
    }

    /* ── Cleanup temp files ───────────────────────────────── */
    remove("_morphy_tmp.py");
    remove("_morphy_out.txt");

    return success;
}

/* ── Main ─────────────────────────────────────────────────── */
int main(void) {
    char tokens[MAX_TOKENS][MAX_LEN];
    char target_word[MAX_LEN] = {0};
    char target_pos[16]       = {0};
    char lemma[MAX_LEN]       = {0};
    int  target_idx = -1;
    const char *method = "WordNet morphy (wn.morphy)";

    /* 1. Load all lowercased tokens */
    int n = load_tokens("lowercased.txt", tokens, MAX_TOKENS);
    if (n <= 0) {
        fprintf(stderr, "ERROR: cannot read lowercased.txt\n");
        return 1;
    }

    /* 2. Load target index */
    FILE *ti = fopen("target_index.txt", "r");
    if (!ti) {
        fprintf(stderr, "ERROR: cannot read target_index.txt\n");
        return 1;
    }
    fscanf(ti, "%d", &target_idx);
    fclose(ti);

    if (target_idx < 0 || target_idx >= n) {
        fprintf(stderr, "ERROR: target_index %d out of range (0..%d)\n",
                target_idx, n - 1);
        return 1;
    }
    safe_copy(target_word, tokens[target_idx], MAX_LEN);

    /* 3. Load target POS */
    FILE *tp = fopen("target_pos.txt", "r");
    if (!tp) {
        fprintf(stderr, "ERROR: cannot read target_pos.txt\n");
        return 1;
    }
    if (!fgets(target_pos, sizeof(target_pos), tp)) {
        fclose(tp);
        fprintf(stderr, "ERROR: target_pos.txt is empty\n");
        return 1;
    }
    fclose(tp);
    target_pos[strcspn(target_pos, "\r\n# ")] = '\0';

    /* 4. Map PTB POS → WordNet POS */
    const char *wn_pos = ptb_to_wn(target_pos);

    printf("=== STEP 6: LEMMATIZATION (WordNet morphy) ===\n\n");
    printf("Target word : %s\n", target_word);
    printf("PTB POS tag : %s", target_pos);

    if (!wn_pos) {
        printf("  (no WordNet mapping — returning word as-is)\n");
        safe_copy(lemma, target_word, MAX_LEN);
        method = "no-op (unsupported POS)";
        goto write_output;
    }

    printf("  →  WordNet POS: '%s'\n", wn_pos);
    printf("Calling     : wn.morphy('%s', '%s') ...\n", target_word, wn_pos);

    /* 5. Run wn.morphy via Python */
    if (morphy_via_python(target_word, wn_pos, lemma, MAX_LEN)) {
        printf("Lemma       : %s\n", lemma);
    } else {
        /* Python or NLTK not available: return word unchanged */
        printf("WARNING: Python/NLTK call failed — "
               "returning target word as-is.\n");
        printf("         (Install NLTK: pip install nltk)\n");
        safe_copy(lemma, target_word, MAX_LEN);
        method = "fallback (Python/NLTK unavailable)";
    }

write_output:
    printf("Method      : %s\n", method);

    /* 6. Write lemma.txt */
    FILE *out_fp = fopen("lemma.txt", "w");
    if (!out_fp) {
        fprintf(stderr, "ERROR: cannot write lemma.txt\n");
        return 1;
    }
    fprintf(out_fp, "%s\n", lemma);
    fprintf(out_fp, "# target_word : %s\n", target_word);
    fprintf(out_fp, "# target_pos  : %s\n", target_pos);
    fprintf(out_fp, "# target_index: %d\n", target_idx);
    fclose(out_fp);

    printf("\nSaved → lemma.txt\n");
    return 0;
}