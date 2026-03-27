#include <stdio.h>
#include <string.h>
#include <ctype.h>

/* =========================================================
 * STEP 3: LOWERCASING
 * ---------------------------------------------------------
 * INPUT  : tokens.txt      — CSV token list from Step 1
 * OUTPUT : lowercased.txt  — same CSV, non-NE tokens lowercased
 *
 * RULE:
 *   - Punctuation tokens   → leave unchanged
 *   - Everything else      → lowercase
 *
 * NER HOOK (fill in later):
 *   When NER is ready, load ner.txt alongside tokens.txt.
 *   Any token whose NER_TAG != "O" → skip lowercasing.
 *   That single line is the ONLY change needed here.
 * ========================================================= */

#define MAX_TOKENS    256
#define MAX_TOKEN_LEN  64
#define MAX_LINE     4096

static void str_tolower(char *s) {
    while (*s) { *s = (char)tolower((unsigned char)*s); s++; }
}

/* returns 1 if token has no alphanumeric chars (pure punctuation) */
static int is_punct_token(const char *s) {
    while (*s) {
        if (isalnum((unsigned char)*s)) return 0;
        s++;
    }
    return 1;
}

int main(void) {
    char  tokens[MAX_TOKENS][MAX_TOKEN_LEN];
    char  lowered[MAX_TOKENS][MAX_TOKEN_LEN];
    int   n = 0, i;
    char  line[MAX_LINE];
    FILE *fp;

    /* ── 1. Load tokens.txt ──────────────────────────────── */
    fp = fopen("tokens.txt", "r");
    if (!fp) { fprintf(stderr, "Cannot open tokens.txt\n"); return 1; }
    if (!fgets(line, MAX_LINE, fp)) { fclose(fp); return 1; }
    fclose(fp);

    line[strcspn(line, "\r\n")] = '\0';

    char *tok = strtok(line, ",");
    while (tok && n < MAX_TOKENS) {
        while (*tok == ' ') tok++;
        strncpy(tokens[n], tok, MAX_TOKEN_LEN - 1);
        tokens[n][MAX_TOKEN_LEN - 1] = '\0';
        n++;
        tok = strtok(NULL, ",");
    }

    /* ── 2. Lowercase ────────────────────────────────────── */
    for (i = 0; i < n; i++) {
        strncpy(lowered[i], tokens[i], MAX_TOKEN_LEN - 1);
        lowered[i][MAX_TOKEN_LEN - 1] = '\0';

        if (is_punct_token(lowered[i])) continue;  /* skip punctuation */

        /* === NER HOOK ======================================
         * FUTURE: if (ner_tag[i] != O) continue;
         * That one line protects named entities from
         * being lowercased once NER is integrated.
         * ==================================================*/

        str_tolower(lowered[i]);
    }

    /* ── 3. Print ────────────────────────────────────────── */
    printf("=== STEP 3: LOWERCASING ===\n\n");
    printf("%-22s  %-22s\n", "ORIGINAL", "LOWERCASED");
    printf("------------------------------------------\n");
    for (i = 0; i < n; i++)
        printf("%-22s  %s\n", tokens[i], lowered[i]);

    /* ── 4. Save lowercased.txt ──────────────────────────── */
    fp = fopen("lowercased.txt", "w");
    if (!fp) { fprintf(stderr, "Cannot write lowercased.txt\n"); return 1; }
    for (i = 0; i < n; i++) {
        if (i > 0) fprintf(fp, ", ");
        fprintf(fp, "%s", lowered[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);

    printf("\nSaved to lowercased.txt\n");
    return 0;
}