/*
 * ============================================================
 * STEP 6: LEMMATIZATION
 * ============================================================
 * PROCESS:
 *   1. Read target word from lowercased.txt (using target_index.txt)
 *   2. Read its POS tag from target_pos.txt
 *   3. Try irregular lookup table first
 *   4. If not found, apply POS-specific suffix stripping rules
 *   5. Save result to lemma.txt
 *
 * INPUT  : lowercased.txt    — comma-separated tokens
 *          target_index.txt  — integer index of target word
 *          target_pos.txt    — POS tag of target word
 *
 * OUTPUT : lemma.txt         — lemma of the target word
 * ============================================================
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_LEN    64
#define MAX_LINE 4096
#define MAX_TOKENS 512

/* ── Irregular word table ────────────────────────────────────
 * Format: {inflected_form, lemma}
 * Covers irregular verbs, nouns, adjectives from standard
 * English grammar references.                               */
typedef struct { const char *form; const char *lemma; } IrregEntry;

static const IrregEntry irregulars[] = {
    /* ── Irregular VERBS (past tense VBD) ── */
    {"was","be"},{"were","be"},{"been","be"},{"is","be"},{"are","be"},{"am","be"},
    {"had","have"},{"has","have"},
    {"did","do"},{"done","do"},
    {"went","go"},{"gone","go"},
    {"got","get"},{"gotten","get"},
    {"said","say"},
    {"made","make"},
    {"took","take"},{"taken","take"},
    {"came","come"},{"come","come"},
    {"saw","see"},{"seen","see"},
    {"knew","know"},{"known","know"},
    {"gave","give"},{"given","give"},
    {"found","find"},
    {"thought","think"},
    {"told","tell"},
    {"became","become"},{"become","become"},
    {"left","leave"},
    {"felt","feel"},
    {"brought","bring"},
    {"kept","keep"},
    {"began","begin"},{"begun","begin"},
    {"held","hold"},
    {"wrote","write"},{"written","write"},
    {"stood","stand"},
    {"heard","hear"},
    {"let","let"},
    {"meant","mean"},
    {"set","set"},
    {"met","meet"},
    {"ran","run"},{"run","run"},
    {"paid","pay"},
    {"sat","sit"},
    {"spoke","speak"},{"spoken","speak"},
    {"cut","cut"},
    {"built","build"},
    {"lost","lose"},
    {"won","win"},
    {"fell","fall"},{"fallen","fall"},
    {"led","lead"},
    {"broke","break"},{"broken","break"},
    {"drew","draw"},{"drawn","draw"},
    {"spent","spend"},
    {"grew","grow"},{"grown","grow"},
    {"sent","send"},
    {"read","read"},
    {"sold","sell"},
    {"told","tell"},
    {"chose","choose"},{"chosen","choose"},
    {"drove","drive"},{"driven","drive"},
    {"rose","rise"},{"risen","rise"},
    {"rode","ride"},{"ridden","ride"},
    {"wore","wear"},{"worn","wear"},
    {"flew","fly"},{"flown","fly"},
    {"threw","throw"},{"thrown","throw"},
    {"bought","buy"},
    {"caught","catch"},
    {"taught","teach"},
    {"fought","fight"},
    {"sought","seek"},
    {"brought","bring"},
    {"thought","think"},
    {"slept","sleep"},
    {"wept","weep"},
    {"kept","keep"},
    {"crept","creep"},
    {"swept","sweep"},
    {"leapt","leap"},
    {"dealt","deal"},
    {"felt","feel"},
    {"knelt","kneel"},
    {"smelt","smell"},
    {"spelt","spell"},
    {"spilt","spill"},
    {"spoilt","spoil"},
    {"learnt","learn"},
    {"dreamt","dream"},
    {"burnt","burn"},
    {"hit","hit"},
    {"put","put"},
    {"shut","shut"},
    {"spread","spread"},
    {"hurt","hurt"},
    {"quit","quit"},
    {"cast","cast"},
    {"cost","cost"},
    {"bet","bet"},
    {"bid","bid"},
    {"burst","burst"},
    {"fit","fit"},
    {"split","split"},
    {"knit","knit"},
    /* ── Irregular NOUNS (NNS) ── */
    {"men","man"},{"women","woman"},{"children","child"},
    {"people","person"},{"feet","foot"},{"teeth","tooth"},
    {"geese","goose"},{"mice","mouse"},{"lice","louse"},
    {"oxen","ox"},{"criteria","criterion"},{"phenomena","phenomenon"},
    {"data","datum"},{"media","medium"},{"alumni","alumnus"},
    {"analyses","analysis"},{"bases","basis"},{"crises","crisis"},
    {"theses","thesis"},{"hypotheses","hypothesis"},
    {"indices","index"},{"appendices","appendix"},
    {"matrices","matrix"},{"vertices","vertex"},
    {"leaves","leaf"},{"knives","knife"},{"lives","life"},
    {"wives","wife"},{"wolves","wolf"},{"halves","half"},
    {"shelves","shelf"},{"themselves","themselves"},
    /* ── Irregular ADJECTIVES (JJR/JJS) ── */
    {"better","good"},{"best","good"},
    {"worse","bad"},{"worst","bad"},
    {"more","many"},{"most","many"},
    {"less","little"},{"least","little"},
    {"further","far"},{"furthest","far"},
    {"farther","far"},{"farthest","far"},
    {"elder","old"},{"eldest","old"},
    {"latter","late"},{"last","late"},
    {NULL, NULL}
};

/* ── Helper: does string end with suffix? ─────────────────── */
static int endswith(const char *s, const char *sfx) {
    int sl = (int)strlen(s), fl = (int)strlen(sfx);
    if (sl < fl) return 0;
    return strcmp(s + sl - fl, sfx) == 0;
}

/* ── Helper: safe string copy to fixed buffer ─────────────── */
static void safe_copy(char *dst, const char *src, int maxlen) {
    strncpy(dst, src, maxlen - 1);
    dst[maxlen - 1] = '\0';
}

/* ── Layer 1: Irregular lookup ────────────────────────────── */
static const char *lookup_irregular(const char *word) {
    for (int i = 0; irregulars[i].form != NULL; i++)
        if (strcmp(irregulars[i].form, word) == 0)
            return irregulars[i].lemma;
    return NULL;
}

/* ── Layer 2: POS-guided suffix stripping ─────────────────── */
static void suffix_lemma(const char *word, const char *pos,
                          char *out, int outlen) {
    int len = (int)strlen(word);
    safe_copy(out, word, outlen);   /* default: return as-is */

    /* ── VERBS ── */
    if (strncmp(pos, "VB", 2) == 0) {

        /* VBG: running→run, stopping→stop, making→make */
        if (strcmp(pos, "VBG") == 0) {
            /* -nning → -n (running→run, beginning→begin) */
            if (len > 5 && endswith(word,"nning")) {
                safe_copy(out, word, outlen);
                out[len - 4] = '\0';   /* strip 'ning', keep one 'n' */
                return;
            }
            /* -tting → -t (hitting→hit, sitting→sit) */
            if (len > 5 && endswith(word,"tting")) {
                safe_copy(out, word, outlen);
                out[len - 4] = '\0';
                return;
            }
            /* -pping → -p (stopping→stop, wrapping→wrap) */
            if (len > 5 && endswith(word,"pping")) {
                safe_copy(out, word, outlen);
                out[len - 4] = '\0';
                return;
            }
            /* -ing after 'e' was dropped: making→make, taking→take */
            if (len > 4 && endswith(word,"ing")) {
                /* try stripping -ing and adding -e */
                char tmp[MAX_LEN];
                safe_copy(tmp, word, MAX_LEN);
                tmp[len - 3] = 'e';
                tmp[len - 2] = '\0';
                /* simple heuristic: if result ends in consonant+e, use it */
                int tl = strlen(tmp);
                char c = (tl > 1) ? tmp[tl - 2] : 0;
                if (c && strchr("bcdfghjklmnpqrstvwxyz", c)) {
                    safe_copy(out, tmp, outlen);
                    return;
                }
                /* else just strip -ing */
                safe_copy(out, word, outlen);
                out[len - 3] = '\0';
                return;
            }
        }

        /* VBD / VBN: walked→walk, loved→love, stopped→stop */
        if (strcmp(pos,"VBD")==0 || strcmp(pos,"VBN")==0) {
            if (endswith(word,"ed")) {
                /* -doubled consonant + ed: stopped→stop */
                if (len > 4 && word[len-3] == word[len-4]) {
                    safe_copy(out, word, outlen);
                    out[len - 3] = '\0';   /* strip last char + 'ed' */
                    return;
                }
                /* -ied → -y: tried→try, carried→carry */
                if (endswith(word,"ied")) {
                    safe_copy(out, word, outlen);
                    out[len - 3] = 'y';
                    out[len - 2] = '\0';
                    return;
                }
                /* -e already there: loved→love, saved→save */
                if (len > 3 && word[len-3] == 'e') {
                    safe_copy(out, word, outlen);
                    out[len - 1] = '\0';   /* strip just 'd' */
                    return;
                }
                /* regular: walked→walk */
                safe_copy(out, word, outlen);
                out[len - 2] = '\0';
                return;
            }
        }

        /* VBZ: runs→run, goes→go, watches→watch */
        if (strcmp(pos,"VBZ") == 0) {
            if (endswith(word,"ies")) {
                /* carries→carry */
                safe_copy(out, word, outlen);
                out[len - 3] = 'y';
                out[len - 2] = '\0';
                return;
            }
            if (endswith(word,"es") &&
                (endswith(word,"ches") || endswith(word,"shes") ||
                 endswith(word,"xes")  || endswith(word,"zes")  ||
                 endswith(word,"ses"))) {
                /* watches→watch, washes→wash */
                safe_copy(out, word, outlen);
                out[len - 2] = '\0';
                return;
            }
            if (endswith(word,"s") && len > 2) {
                safe_copy(out, word, outlen);
                out[len - 1] = '\0';
                return;
            }
        }
    }

    /* ── NOUNS (NNS) ── */
    if (strcmp(pos,"NNS") == 0) {
        /* -ies → -y: puppies→puppy, cities→city */
        if (endswith(word,"ies") && len > 4) {
            safe_copy(out, word, outlen);
            out[len - 3] = 'y';
            out[len - 2] = '\0';
            return;
        }
        /* -ves → -f: leaves→leaf, knives→knife */
        if (endswith(word,"ves") && len > 4) {
            safe_copy(out, word, outlen);
            out[len - 3] = 'f';
            out[len - 2] = '\0';
            return;
        }
        /* -es after sibilant: watches→watch, boxes→box */
        if (endswith(word,"es") && len > 3 &&
            (endswith(word,"ches") || endswith(word,"shes") ||
             endswith(word,"xes")  || endswith(word,"zes")  ||
             endswith(word,"ses"))) {
            safe_copy(out, word, outlen);
            out[len - 2] = '\0';
            return;
        }
        /* regular -s: dogs→dog, cats→cat */
        if (endswith(word,"s") && len > 2) {
            safe_copy(out, word, outlen);
            out[len - 1] = '\0';
            return;
        }
    }

    /* ── ADJECTIVES (JJR comparative) ── */
    if (strcmp(pos,"JJR") == 0) {
        /* -ier → -y: happier→happy */
        if (endswith(word,"ier") && len > 4) {
            safe_copy(out, word, outlen);
            out[len - 3] = 'y';
            out[len - 2] = '\0';
            return;
        }
        /* -doubled+er: bigger→big, hotter→hot */
        if (len > 4 && endswith(word,"er") &&
            word[len-3] == word[len-4]) {
            safe_copy(out, word, outlen);
            out[len - 3] = '\0';
            return;
        }
        /* -er: faster→fast, older→old */
        if (endswith(word,"er") && len > 4) {
            safe_copy(out, word, outlen);
            out[len - 2] = '\0';
            return;
        }
    }

    /* ── ADJECTIVES (JJS superlative) ── */
    if (strcmp(pos,"JJS") == 0) {
        /* -iest → -y: happiest→happy */
        if (endswith(word,"iest") && len > 5) {
            safe_copy(out, word, outlen);
            out[len - 4] = 'y';
            out[len - 3] = '\0';
            return;
        }
        /* -doubled+est: biggest→big */
        if (len > 5 && endswith(word,"est") &&
            word[len-4] == word[len-5]) {
            safe_copy(out, word, outlen);
            out[len - 4] = '\0';
            return;
        }
        /* -est: fastest→fast */
        if (endswith(word,"est") && len > 5) {
            safe_copy(out, word, outlen);
            out[len - 3] = '\0';
            return;
        }
    }

    /* Everything else (JJ, NN, VB, RB, etc.) → return as-is */
}

/* ── Load comma-separated token line ─────────────────────── */
static int load_tokens(const char *fname,
                        char arr[][MAX_LEN], int max) {
    FILE *fp = fopen(fname, "r");
    if (!fp) return -1;
    char line[MAX_LINE];
    if (!fgets(line, MAX_LINE, fp)) { fclose(fp); return 0; }
    fclose(fp);
    line[strcspn(line,"\r\n")] = '\0';
    int n = 0;
    char *tok = strtok(line, ",");
    while (tok && n < max) {
        while (*tok == ' ') tok++;
        safe_copy(arr[n++], tok, MAX_LEN);
        tok = strtok(NULL, ",");
    }
    return n;
}

/* ── Main ─────────────────────────────────────────────────── */
int main(void) {
    char tokens[MAX_TOKENS][MAX_LEN];
    char target_word[MAX_LEN] = {0};
    char target_pos[16]       = {0};
    char lemma[MAX_LEN]       = {0};
    int  target_idx = -1;

    /* 1. Load all tokens */
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

    /* 4. Lemmatize */
    const char *irr = lookup_irregular(target_word);
    if (irr) {
        safe_copy(lemma, irr, MAX_LEN);
    } else {
        suffix_lemma(target_word, target_pos, lemma, MAX_LEN);
    }

    /* 5. Print results */
    printf("=== STEP 6: LEMMATIZATION ===\n\n");
    printf("Target word : %s\n", target_word);
    printf("POS tag     : %s\n", target_pos);
    printf("Lemma       : %s\n", lemma);
    printf("Method      : %s\n",
           irr ? "irregular lookup table" : "POS-guided suffix stripping");

    /* 6. Save lemma.txt */
    FILE *out = fopen("lemma.txt", "w");
    if (!out) {
        fprintf(stderr, "ERROR: cannot write lemma.txt\n");
        return 1;
    }
    fprintf(out, "%s\n", lemma);
    fprintf(out, "# target_word : %s\n", target_word);
    fprintf(out, "# target_pos  : %s\n", target_pos);
    fprintf(out, "# target_index: %d\n", target_idx);
    fclose(out);

    printf("\nSaved → lemma.txt\n");
    return 0;
}