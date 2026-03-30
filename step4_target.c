#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

/* =========================================================
 * STEP 4: TARGET INDEX IDENTIFICATION
 * ---------------------------------------------------------
 * INPUT  : lowercased.txt — CSV tokens from Step 3
 *          input.txt      — line 2 is the target word
 * OUTPUT : target_index.txt — the token index of target word
 *
 * AMBIGUITY HANDLING:
 *   If the target appears multiple times (e.g. "the"),
 *   the FIRST occurrence is used by default.
 *   To manually pick a different occurrence:
 *     step4_target <word> <index>
 * ========================================================= */

#define MAX_TOKENS    256
#define MAX_TOKEN_LEN  64
#define MAX_LINE     4096

static void str_tolower(char *s) {
    while (*s) { *s = (char)tolower((unsigned char)*s); s++; }
}

int main(int argc, char *argv[]) {
    char  tokens[MAX_TOKENS][MAX_TOKEN_LEN];
    int   n = 0, i;
    char  line[MAX_LINE];
    char  target[MAX_TOKEN_LEN];
    char  target_low[MAX_TOKEN_LEN];
    int   found_index = -1;
    int   count       = 0;
    FILE *fp;

    /* ── 1. Load lowercased tokens ───────────────────────── */
    fp = fopen("lowercased.txt", "r");
    if (!fp) { fprintf(stderr, "Cannot open lowercased.txt\n"); return 1; }
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

    /* ── 2. Load target word: from arg or input.txt line 2 ─ */
    if (argc >= 2) {
        strncpy(target, argv[1], MAX_TOKEN_LEN - 1);
        target[MAX_TOKEN_LEN - 1] = '\0';
    } else {
        fp = fopen("input.txt", "r");
        if (!fp) { fprintf(stderr, "Cannot open input.txt\n"); return 1; }
        fgets(line, MAX_LINE, fp);          /* skip sentence line */
        if (!fgets(target, MAX_TOKEN_LEN, fp)) { fclose(fp); return 1; }
        fclose(fp);
        target[strcspn(target, "\r\n")] = '\0';
    }

    /* ── 3. Lowercase target for comparison ──────────────── */
    strncpy(target_low, target, MAX_TOKEN_LEN - 1);
    target_low[MAX_TOKEN_LEN - 1] = '\0';
    str_tolower(target_low);

    /* ── 4. Scan for match ───────────────────────────────── */
    for (i = 0; i < n; i++) {
        if (strcmp(tokens[i], target_low) == 0) {
            count++;
            if (found_index == -1)
                found_index = i;    /* first occurrence */
        }
    }

    /* ── 5. Print ────────────────────────────────────────── */
    printf("=== STEP 4: TARGET INDEX IDENTIFICATION ===\n\n");
    printf("Target word  : \"%s\"\n", target_low);
    printf("Total tokens : %d\n\n", n);

    if (found_index == -1) {
        printf("ERROR: \"%s\" not found in token list.\n", target_low);
        return 1;
    }

    for (i = 0; i < n; i++) {
        if (i == found_index)
            printf("  [%d] >>> %s <<<  <- TARGET\n", i, tokens[i]);
        else
            printf("  [%d] %s\n", i, tokens[i]);
    }

    /* ── 6. Ambiguity warning ────────────────────────────── */
    if (count > 1) {
        printf("\nWARNING: \"%s\" appears %d times.\n", target_low, count);
        printf("Using FIRST occurrence at index %d.\n", found_index);
        printf("To override: run as  step4_target <word> <index>\n");
    }

    /* ── 7. Optional manual index override via arg 2 ─────── */
    if (argc >= 3) {
        int override = atoi(argv[2]);
        if (override >= 0 && override < n &&
            strcmp(tokens[override], target_low) == 0) {
            found_index = override;
            printf("Manual override: using index %d.\n", found_index);
        } else {
            printf("WARNING: override index %d invalid. Using %d.\n",
                   override, found_index);
        }
    }

    printf("\nTarget index : %d\n", found_index);

    /* ── 8. Save target_index.txt ────────────────────────── */
    fp = fopen("target_index.txt", "w");
    if (!fp) { fprintf(stderr, "Cannot write target_index.txt\n"); return 1; }
    fprintf(fp, "%d\n", found_index);
    fprintf(fp, "# target word  : %s\n", target_low);
    fprintf(fp, "# total tokens : %d\n", n);
    fprintf(fp, "# occurrences  : %d\n", count);
    fclose(fp);

    printf("\nSaved to target_index.txt\n");
    return 0;
}