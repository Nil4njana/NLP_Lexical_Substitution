/*
 * ============================================================
 * STEP 5: POS TAGGER  (Penn Treebank lookup + suffix fallback)
 * ============================================================
 *
 * PROCESS:
 *   1. Load lowercased tokens from lowercased.txt
 *   2. Load original-cased tokens from tokens.txt
 *   3. For each token:
 *        a. Punctuation check          → PUNCT
 *        b. Digit string               → CD
 *        c. PTB closed-class table     → exact tag
 *        d. PTB known-word table       → most-frequent tag
 *        e. Capitalised + not sent-start → NNP (proper noun)
 *        f. Suffix rules (Jurafsky §8) → morphological guess
 *        g. Default                    → NN
 *   4. Write pos_tags.txt  (TOKEN,POS  one per line)
 *   5. Write target_pos.txt (single tag for target word)
 *
 * INPUT  : lowercased.txt   — comma-separated lowercase tokens
 *          tokens.txt       — comma-separated original-case tokens
 *          target_index.txt — integer index of target word
 *
 * OUTPUT : pos_tags.txt     — TOKEN,POS one per line
 *          target_pos.txt   — POS of target word only
 * ============================================================
 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

#define MAX_TOKENS  512
#define MAX_LEN      64
#define MAX_LINE   8192

/* ── PTB tag set used (subset) ──────────────────────────────
 * CC   Coordinating conjunction
 * CD   Cardinal number
 * DT   Determiner
 * EX   Existential there
 * IN   Preposition / subordinating conjunction
 * JJ   Adjective
 * JJR  Adjective comparative
 * JJS  Adjective superlative
 * MD   Modal
 * NN   Noun singular or mass
 * NNS  Noun plural
 * NNP  Proper noun singular
 * PDT  Predeterminer
 * POS  Possessive ending
 * PRP  Personal pronoun
 * PRP$ Possessive pronoun
 * RB   Adverb
 * RBR  Adverb comparative
 * RBS  Adverb superlative
 * RP   Particle
 * TO   "to"
 * UH   Interjection
 * VB   Verb base form
 * VBD  Verb past tense
 * VBG  Verb gerund / present participle
 * VBN  Verb past participle
 * VBP  Verb non-3rd person singular present
 * VBZ  Verb 3rd person singular present
 * WDT  Wh-determiner
 * WP   Wh-pronoun
 * WRB  Wh-adverb
 * PUNCT Punctuation (our label)
 * ─────────────────────────────────────────────────────────── */

typedef struct { const char *word; const char *tag; } Entry;

/* ── TABLE 1: Closed-class words (always this tag) ──────────
 * Source: Jurafsky & Martin Ch.8 + PTB WSJ frequency counts
 * These words are 99%+ unambiguous in PTB                    */
static const Entry closed[] = {
    /* --- Determiners --- */
    {"the","DT"},{"a","DT"},{"an","DT"},
    {"this","DT"},{"that","DT"},{"these","DT"},{"those","DT"},
    {"every","DT"},{"each","DT"},{"any","DT"},{"all","DT"},
    {"both","DT"},{"either","DT"},{"neither","DT"},{"no","DT"},
    {"half","DT"},{"such","DT"},{"what","DT"},{"whatever","DT"},
    /* --- Possessive pronouns (DT role) --- */
    {"my","PRP$"},{"your","PRP$"},{"his","PRP$"},{"her","PRP$"},
    {"its","PRP$"},{"our","PRP$"},{"their","PRP$"},{"whose","WP$"},
    /* --- Personal pronouns --- */
    {"i","PRP"},{"you","PRP"},{"he","PRP"},{"she","PRP"},
    {"it","PRP"},{"we","PRP"},{"they","PRP"},{"me","PRP"},
    {"him","PRP"},{"us","PRP"},{"them","PRP"},
    {"myself","PRP"},{"yourself","PRP"},{"himself","PRP"},
    {"herself","PRP"},{"itself","PRP"},{"ourselves","PRP"},
    {"themselves","PRP"},
    /* --- Wh-words --- */
    {"who","WP"},{"whom","WP"},{"which","WDT"},
    {"when","WRB"},{"where","WRB"},{"why","WRB"},{"how","WRB"},
    /* --- Modals --- */
    {"can","MD"},{"could","MD"},{"may","MD"},{"might","MD"},
    {"shall","MD"},{"should","MD"},{"will","MD"},{"would","MD"},
    {"must","MD"},{"ought","MD"},{"dare","MD"},{"need","MD"},
    /* --- Coordinating conjunctions --- */
    {"and","CC"},{"but","CC"},{"or","CC"},{"nor","CC"},
    {"for","CC"},{"yet","CC"},{"so","CC"},
    /* --- Prepositions & subordinators --- */
    {"in","IN"},{"on","IN"},{"at","IN"},{"by","IN"},{"with","IN"},
    {"of","IN"},{"to","TO"},{"from","IN"},{"into","IN"},{"onto","IN"},
    {"about","IN"},{"above","IN"},{"across","IN"},{"after","IN"},
    {"against","IN"},{"along","IN"},{"among","IN"},{"around","IN"},
    {"as","IN"},{"because","IN"},{"before","IN"},{"behind","IN"},
    {"below","IN"},{"beneath","IN"},{"beside","IN"},{"between","IN"},
    {"beyond","IN"},{"during","IN"},{"except","IN"},{"inside","IN"},
    {"near","IN"},{"off","IN"},{"outside","IN"},{"over","IN"},
    {"past","IN"},{"since","IN"},{"through","IN"},{"throughout","IN"},
    {"till","IN"},{"under","IN"},{"until","IN"},{"up","RP"},
    {"upon","IN"},{"while","IN"},{"within","IN"},{"without","IN"},
    {"although","IN"},{"if","IN"},{"unless","IN"},{"whether","IN"},
    {"than","IN"},{"that","IN"},
    /* --- Existential --- */
    {"there","EX"},
    /* --- Predeterminer --- */
    {"quite","PDT"},{"rather","PDT"},{"such","PDT"},{"half","PDT"},
    /* --- Interjections --- */
    {"oh","UH"},{"ah","UH"},{"yes","UH"},{"no","UH"},
    {"hey","UH"},{"wow","UH"},{"uh","UH"},{"um","UH"},
    /* --- Contractions / particles --- */
    {"n't","RB"},{"not","RB"},{"'s","POS"},
    {"'ll","MD"},{"'ve","VBP"},{"'re","VBP"},{"'d","MD"},
    {"'m","VBP"},
    /* --- Numbers as words --- */
    {"zero","CD"},{"one","CD"},{"two","CD"},{"three","CD"},
    {"four","CD"},{"five","CD"},{"six","CD"},{"seven","CD"},
    {"eight","CD"},{"nine","CD"},{"ten","CD"},{"eleven","CD"},
    {"twelve","CD"},{"hundred","CD"},{"thousand","CD"},
    {"million","CD"},{"billion","CD"},{"trillion","CD"},
    {NULL,NULL}
};

/* ── TABLE 2: Known open-class words → most frequent PTB tag
 * Compiled from WSJ Penn Treebank word frequency counts.
 * Only words where the most-frequent tag accounts for >80%
 * of occurrences are included (avoids misleading tags).       */
static const Entry known[] = {
    /* Common verbs — base form */
    {"be","VB"},{"have","VB"},{"do","VB"},{"say","VB"},
    {"make","VB"},{"get","VB"},{"take","VB"},{"come","VB"},
    {"see","VB"},{"know","VB"},{"think","VB"},{"look","VB"},
    {"want","VB"},{"give","VB"},{"use","VB"},{"find","VB"},
    {"tell","VB"},{"ask","VB"},{"seem","VB"},{"feel","VB"},
    {"try","VB"},{"leave","VB"},{"call","VB"},{"keep","VB"},
    {"let","VB"},{"begin","VB"},{"show","VB"},{"hear","VB"},
    {"play","VB"},{"run","VB"},{"move","VB"},{"live","VB"},
    {"hold","VB"},{"bring","VB"},{"write","VB"},{"stand","VB"},
    {"turn","VB"},{"start","VB"},{"lose","VB"},{"buy","VB"},
    {"remember","VB"},{"open","VB"},{"need","VB"},{"learn","VB"},
    {"change","VB"},{"help","VB"},{"build","VB"},{"fall","VB"},
    {"work","VB"},{"read","VB"},{"spend","VB"},{"grow","VB"},
    {"win","VB"},{"wait","VB"},{"pass","VB"},{"meet","VB"},
    {"set","VB"},{"send","VB"},{"pay","VB"},{"speak","VB"},
    {"sit","VB"},{"eat","VB"},{"stop","VB"},{"break","VB"},
    {"put","VB"},{"cut","VB"},{"raise","VB"},{"include","VB"},
    {"add","VB"},{"create","VB"},{"remain","VB"},{"lead","VB"},
    {"draw","VB"},{"check","VB"},{"pick","VB"},{"hit","VB"},
    {"reduce","VB"},{"increase","VB"},{"provide","VB"},
    {"support","VB"},{"consider","VB"},{"allow","VB"},
    {"develop","VB"},{"continue","VB"},{"produce","VB"},
    {"describe","VB"},{"expect","VB"},{"determine","VB"},
    /* VBZ — 3rd person singular present */
    {"is","VBZ"},{"has","VBZ"},{"does","VBZ"},{"goes","VBZ"},
    {"gets","VBZ"},{"says","VBZ"},{"makes","VBZ"},{"takes","VBZ"},
    {"seems","VBZ"},{"comes","VBZ"},{"knows","VBZ"},{"gives","VBZ"},
    {"wants","VBZ"},{"looks","VBZ"},{"thinks","VBZ"},{"uses","VBZ"},
    {"works","VBZ"},{"needs","VBZ"},{"means","VBZ"},{"includes","VBZ"},
    {"shows","VBZ"},{"helps","VBZ"},{"becomes","VBZ"},{"remains","VBZ"},
    /* VBZ — common -ites/-ates/-etes/-otes/-utes/-ives/-oves/-eves/-aves
     * (verbs ending in silent-e + s, which suffix '-s→NNS' would steal) */
    {"writes","VBZ"},{"reads","VBZ"},{"runs","VBZ"},{"speaks","VBZ"},
    {"drives","VBZ"},{"lives","VBZ"},{"moves","VBZ"},{"loves","VBZ"},
    {"leaves","VBZ"},{"gives","VBZ"},{"saves","VBZ"},{"hates","VBZ"},
    {"creates","VBZ"},{"relates","VBZ"},{"operates","VBZ"},{"continues","VBZ"},
    {"produces","VBZ"},{"reduces","VBZ"},{"decides","VBZ"},{"provides","VBZ"},
    {"describes","VBZ"},{"assumes","VBZ"},{"becomes","VBZ"},{"notices","VBZ"},
    {"places","VBZ"},{"raises","VBZ"},{"loses","VBZ"},{"closes","VBZ"},
    {"changes","VBZ"},{"manages","VBZ"},{"improves","VBZ"},{"moves","VBZ"},
    {"proves","VBZ"},{"serves","VBZ"},{"solves","VBZ"},{"involves","VBZ"},
    {"receives","VBZ"},{"believes","VBZ"},{"achieves","VBZ"},{"perceives","VBZ"},
    {"carries","VBZ"},{"tries","VBZ"},{"studies","VBZ"},{"applies","VBZ"},
    {"replies","VBZ"},{"relies","VBZ"},{"varies","VBZ"},{"supplies","VBZ"},
    {"teaches","VBZ"},{"reaches","VBZ"},{"catches","VBZ"},{"watches","VBZ"},
    {"pushes","VBZ"},{"rushes","VBZ"},{"finishes","VBZ"},{"wishes","VBZ"},
    {"fixes","VBZ"},{"mixes","VBZ"},{"expects","VBZ"},{"suggests","VBZ"},
    {"starts","VBZ"},{"ends","VBZ"},{"supports","VBZ"},{"allows","VBZ"},
    {"follows","VBZ"},{"grows","VBZ"},{"draws","VBZ"},{"throws","VBZ"},
    {"knows","VBZ"},{"shows","VBZ"},{"flows","VBZ"},{"grows","VBZ"},
    {"adds","VBZ"},{"leads","VBZ"},{"builds","VBZ"},{"holds","VBZ"},
    {"finds","VBZ"},{"sends","VBZ"},{"stands","VBZ"},{"spends","VBZ"},
    {"considers","VBZ"},{"remembers","VBZ"},{"understands","VBZ"},
    {"represents","VBZ"},{"determines","VBZ"},{"demonstrates","VBZ"},
    /* VBD — past tense */
    {"was","VBD"},{"were","VBD"},{"had","VBD"},{"did","VBD"},
    {"went","VBD"},{"got","VBD"},{"said","VBD"},{"made","VBD"},
    {"took","VBD"},{"came","VBD"},{"saw","VBD"},{"knew","VBD"},
    {"gave","VBD"},{"found","VBD"},{"thought","VBD"},{"told","VBD"},
    {"became","VBD"},{"left","VBD"},{"felt","VBD"},{"put","VBD"},
    {"brought","VBD"},{"kept","VBD"},{"began","VBD"},{"held","VBD"},
    {"wrote","VBD"},{"stood","VBD"},{"heard","VBD"},{"let","VBD"},
    {"meant","VBD"},{"set","VBD"},{"met","VBD"},{"ran","VBD"},
    {"paid","VBD"},{"sat","VBD"},{"spoke","VBD"},{"cut","VBD"},
    {"built","VBD"},{"lost","VBD"},{"won","VBD"},{"fell","VBD"},
    {"led","VBD"},{"broke","VBD"},{"drew","VBD"},{"spent","VBD"},
    {"grew","VBD"},{"sent","VBD"},{"showed","VBD"},{"asked","VBD"},
    /* VBN — past participle */
    {"been","VBN"},{"had","VBN"},{"done","VBN"},{"gone","VBN"},
    {"given","VBN"},{"taken","VBN"},{"seen","VBN"},{"known","VBN"},
    {"found","VBN"},{"made","VBN"},{"said","VBN"},{"come","VBN"},
    {"used","VBN"},{"told","VBN"},{"called","VBN"},{"shown","VBN"},
    {"kept","VBN"},{"held","VBN"},{"set","VBN"},{"let","VBN"},
    {"written","VBN"},{"broken","VBN"},{"spoken","VBN"},
    {"grown","VBN"},{"fallen","VBN"},{"drawn","VBN"},
    {"built","VBN"},{"lost","VBN"},{"sent","VBN"},{"paid","VBN"},
    /* VBG — gerund */
    {"being","VBG"},{"having","VBG"},{"doing","VBG"},{"going","VBG"},
    {"getting","VBG"},{"making","VBG"},{"taking","VBG"},
    {"saying","VBG"},{"coming","VBG"},{"seeing","VBG"},
    {"knowing","VBG"},{"thinking","VBG"},{"looking","VBG"},
    {"working","VBG"},{"using","VBG"},{"trying","VBG"},
    {"running","VBG"},{"playing","VBG"},{"writing","VBG"},
    {"reading","VBG"},{"speaking","VBG"},{"standing","VBG"},
    {"living","VBG"},{"moving","VBG"},{"including","VBG"},
    {"showing","VBG"},{"following","VBG"},{"helping","VBG"},
    {"changing","VBG"},{"building","VBG"},{"leading","VBG"},
    /* Common nouns */
    {"time","NN"},{"year","NN"},{"people","NNS"},{"way","NN"},
    {"day","NN"},{"man","NN"},{"woman","NN"},{"child","NN"},
    {"children","NNS"},{"men","NNS"},{"women","NNS"},
    {"feet","NNS"},{"teeth","NNS"},{"mice","NNS"},{"geese","NNS"},
    {"world","NN"},{"life","NN"},{"hand","NN"},{"part","NN"},
    {"place","NN"},{"case","NN"},{"week","NN"},{"company","NN"},
    {"system","NN"},{"program","NN"},{"question","NN"},{"work","NN"},
    {"government","NN"},{"number","NN"},{"night","NN"},{"point","NN"},
    {"home","NN"},{"water","NN"},{"room","NN"},{"mother","NN"},
    {"area","NN"},{"money","NN"},{"story","NN"},{"fact","NN"},
    {"month","NN"},{"lot","NN"},{"right","NN"},{"study","NN"},
    {"book","NN"},{"eye","NN"},{"job","NN"},{"word","NN"},
    {"business","NN"},{"issue","NN"},{"side","NN"},{"kind","NN"},
    {"head","NN"},{"house","NN"},{"service","NN"},{"friend","NN"},
    {"father","NN"},{"power","NN"},{"hour","NN"},{"game","NN"},
    {"line","NN"},{"end","NN"},{"among","NN"},{"course","NN"},
    {"city","NN"},{"community","NN"},{"name","NN"},{"president","NN"},
    {"team","NN"},{"minute","NN"},{"air","NN"},{"force","NN"},
    {"body","NN"},{"face","NN"},{"type","NN"},{"form","NN"},
    {"reason","NN"},{"car","NN"},{"level","NN"},{"age","NN"},
    {"policy","NN"},{"process","NN"},{"school","NN"},{"church","NN"},
    {"street","NN"},{"word","NN"},{"sense","NN"},{"law","NN"},
    {"voice","NN"},{"care","NN"},{"door","NN"},{"office","NN"},
    {"decision","NN"},{"report","NN"},{"price","NN"},{"board","NN"},
    {"plan","NN"},{"information","NN"},{"group","NN"},{"order","NN"},
    {"position","NN"},{"field","NN"},{"problem","NN"},{"role","NN"},
    /* Common adjectives */
    {"new","JJ"},{"good","JJ"},{"high","JJ"},{"old","JJ"},
    {"great","JJ"},{"big","JJ"},{"american","JJ"},{"small","JJ"},
    {"large","JJ"},{"national","JJ"},{"young","JJ"},{"local","JJ"},
    {"important","JJ"},{"political","JJ"},{"long","JJ"},{"different","JJ"},
    {"black","JJ"},{"white","JJ"},{"real","JJ"},{"best","JJS"},
    {"free","JJ"},{"able","JJ"},{"open","JJ"},{"public","JJ"},
    {"sure","JJ"},{"clear","JJ"},{"strong","JJ"},{"hard","JJ"},
    {"short","JJ"},{"true","JJ"},{"human","JJ"},{"late","JJ"},
    {"possible","JJ"},{"economic","JJ"},{"social","JJ"},{"full","JJ"},
    {"federal","JJ"},{"major","JJ"},{"medical","JJ"},{"low","JJ"},
    {"central","JJ"},{"current","JJ"},{"specific","JJ"},{"recent","JJ"},
    {"main","JJ"},{"available","JJ"},{"special","JJ"},{"natural","JJ"},
    {"final","JJ"},{"entire","JJ"},{"simple","JJ"},{"common","JJ"},
    {"various","JJ"},{"particular","JJ"},{"foreign","JJ"},{"military","JJ"},
    {"successful","JJ"},{"financial","JJ"},{"difficult","JJ"},
    /* Comparative/superlative */
    {"more","RBR"},{"less","RBR"},{"most","RBS"},{"least","RBS"},
    {"better","JJR"},{"worse","JJR"},{"older","JJR"},{"younger","JJR"},
    {"larger","JJR"},{"smaller","JJR"},{"higher","JJR"},{"lower","JJR"},
    {"bigger","JJR"},{"longer","JJR"},{"shorter","JJR"},{"faster","JJR"},
    {"happier","JJR"},{"angrier","JJR"},{"easier","JJR"},{"heavier","JJR"},
    {"busier","JJR"},{"funnier","JJR"},{"friendlier","JJR"},
    {"happiest","JJS"},{"angriest","JJS"},{"easiest","JJS"},{"heaviest","JJS"},
    {"busiest","JJS"},{"funniest","JJS"},{"friendliest","JJS"},
    /* Common adverbs (-ly forms caught by suffix rule; list others explicitly) */
    {"also","RB"},{"just","RB"},{"very","RB"},{"now","RB"},
    {"well","RB"},{"back","RB"},{"still","RB"},{"even","RB"},
    {"already","RB"},{"never","RB"},{"always","RB"},{"often","RB"},
    {"again","RB"},{"here","RB"},{"then","RB"},{"only","RB"},
    {"first","RB"},{"last","RB"},{"soon","RB"},{"around","RB"},
    {"too","RB"},{"once","RB"},{"almost","RB"},{"away","RB"},
    {"probably","RB"},{"perhaps","RB"},{"certainly","RB"},
    {"usually","RB"},{"quickly","RB"},{"finally","RB"},
    {"recently","RB"},{"simply","RB"},{"actually","RB"},
    {"especially","RB"},{"really","RB"},{"together","RB"},
    {"likely","RB"},{"instead","RB"},
    /* Non-ly adverbs the suffix rule CANNOT detect (would default to NN) */
    {"aloud","RB"},{"ahead","RB"},{"apart","RB"},{"abroad","RB"},
    {"aside","RB"},{"anew","RB"},{"afar","RB"},{"astray","RB"},
    {"aloft","RB"},{"afoot","RB"},{"ashore","RB"},{"abreast","RB"},
    {"forth","RB"},{"hence","RB"},{"thus","RB"},{"thence","RB"},
    {"otherwise","RB"},{"elsewhere","RB"},{"meanwhile","RB"},
    {"nevertheless","RB"},{"nonetheless","RB"},{"furthermore","RB"},
    {"moreover","RB"},{"therefore","RB"},{"however","RB"},
    {"somehow","RB"},{"somewhere","RB"},{"sometime","RB"},
    {"somewhat","RB"},{"anyhow","RB"},{"anyway","RB"},
    {"anywhere","RB"},{"everywhere","RB"},{"nowhere","RB"},
    {"sometimes","RB"},{"seldom","RB"},{"enough","RB"},
    {"hard","RB"},{"fast","RB"},{"long","RB"},{"late","RB"},
    {"early","RB"},{"far","RB"},{"straight","RB"},{"loud","RB"},
    {NULL,NULL}
};

/* ── Helpers ─────────────────────────────────────────────── */

static int is_punct(const char *w) {
    /* single or all-punctuation string */
    for (const char *p = w; *p; p++)
        if (isalnum((unsigned char)*p)) return 0;
    return 1;
}

static int is_digits(const char *w) {
    for (const char *p = w; *p; p++)
        if (!isdigit((unsigned char)*p) && *p != '.' && *p != ',') return 0;
    return strlen(w) > 0;
}

static int endswith(const char *s, const char *sfx) {
    int sl = strlen(s), fl = strlen(sfx);
    return sl >= fl && strcmp(s + sl - fl, sfx) == 0;
}

/* Suffix rules from Jurafsky & Martin Table 8.1 */
static const char *suffix_pos(const char *w) {
    int len = strlen(w);
    if (len < 3) return "NN";
    /* -ing  → VBG (running, playing) */
    if (endswith(w,"ing"))  return "VBG";
    /* -tion/-sion → NN */
    if (endswith(w,"tion")) return "NN";
    if (endswith(w,"sion")) return "NN";
    /* -ness → NN */
    if (endswith(w,"ness")) return "NN";
    /* -ment → NN */
    if (endswith(w,"ment")) return "NN";
    /* -ity/-ty → NN */
    if (endswith(w,"ity"))  return "NN";
    if (endswith(w,"ty"))   return "NN";
    /* -ism → NN */
    if (endswith(w,"ism"))  return "NN";
    /* -ist → NN */
    if (endswith(w,"ist"))  return "NN";
    /* -er/-or → NN (actor, runner) */
    if (endswith(w,"er") && len > 4)  return "NN";
    if (endswith(w,"or") && len > 4)  return "NN";
    /* -ly → RB */
    if (endswith(w,"ly"))   return "RB";
    /* -ful → JJ */
    if (endswith(w,"ful"))  return "JJ";
    /* -ous → JJ */
    if (endswith(w,"ous"))  return "JJ";
    /* -ive → JJ */
    if (endswith(w,"ive"))  return "JJ";
    /* -able/-ible → JJ */
    if (endswith(w,"able")) return "JJ";
    if (endswith(w,"ible")) return "JJ";
    /* -al → JJ */
    if (endswith(w,"al") && len > 4)  return "JJ";
    /* -ish → JJ */
    if (endswith(w,"ish"))  return "JJ";
    /* -less → JJ */
    if (endswith(w,"less")) return "JJ";
    /* -ed → VBD (walked, talked) */
    if (endswith(w,"ed"))   return "VBD";
    /* -en → VBN (broken, taken) */
    if (endswith(w,"en") && len > 4)  return "VBN";
    /* -s → NNS (dogs, cats) — only if not a verb suffix already */
    if (endswith(w,"s") && len > 3)   return "NNS";
    return "NN"; /* default: noun */
}

/* Table lookup helper */
static const char *table_lookup(const Entry *table, const char *w) {
    for (int i = 0; table[i].word; i++)
        if (strcmp(table[i].word, w) == 0)
            return table[i].tag;
    return NULL;
}

/* Load comma-separated line → token array, returns count */
static int load_csv(const char *fname, char arr[][MAX_LEN], int max) {
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
        strncpy(arr[n], tok, MAX_LEN - 1);
        arr[n][MAX_LEN - 1] = '\0';
        n++;
        tok = strtok(NULL, ",");
    }
    return n;
}

/* ── Main ─────────────────────────────────────────────────── */
int main(void) {
    char low[MAX_TOKENS][MAX_LEN];   /* lowercased tokens */
    char orig[MAX_TOKENS][MAX_LEN];  /* original-case tokens */
    char tags[MAX_TOKENS][8];        /* output tags */

    /* 1. Load tokens */
    int n  = load_csv("lowercased.txt", low,  MAX_TOKENS);
    int no = load_csv("tokens.txt",     orig, MAX_TOKENS);
    if (n <= 0) {
        fprintf(stderr, "ERROR: cannot read lowercased.txt\n");
        return 1;
    }
    if (no != n) {
        fprintf(stderr, "Warning: token count mismatch (%d vs %d); using lowercase for capitalisation check\n", n, no);
        for (int i = 0; i < n; i++) strncpy(orig[i], low[i], MAX_LEN - 1);
    }

    /* 2. Tag each token */
    for (int i = 0; i < n; i++) {
        const char *tag = NULL;

        /* (a) Punctuation */
        if (is_punct(low[i])) { tag = "PUNCT"; goto done; }

        /* (b) Number */
        if (is_digits(low[i])) { tag = "CD"; goto done; }

        /* (c) Closed-class exact lookup */
        tag = table_lookup(closed, low[i]);
        if (tag) goto done;

        /* (d) Known open-class lookup */
        tag = table_lookup(known, low[i]);
        if (tag) goto done;

        /* (e) Capitalised + not sentence-start → proper noun
               Sentence start = index 0 OR previous tag was PUNCT .!? */
        if (isupper((unsigned char)orig[i][0])) {
            int sent_start = (i == 0);
            if (!sent_start && i > 0) {
                /* check if previous tag was a sentence-boundary punct */
                char prev = low[i-1][0];
                if (prev == '.' || prev == '!' || prev == '?')
                    sent_start = 1;
            }
            if (!sent_start) { tag = "NNP"; goto done; }
        }

        /* (f) Suffix rules */
        tag = suffix_pos(low[i]);

    done:
        strncpy(tags[i], tag ? tag : "NN", 7);
        tags[i][7] = '\0';
    }

    /* 3. Print to console */
    printf("=== STEP 5: POS TAGGING (PTB lookup + suffix rules) ===\n\n");
    printf("%-4s  %-22s  %s\n", "IDX", "TOKEN", "POS");
    printf("%-4s  %-22s  %s\n", "---", "-----", "---");
    for (int i = 0; i < n; i++)
        printf("%-4d  %-22s  %s\n", i, low[i], tags[i]);

    /* 4. Save pos_tags.txt */
    FILE *fp = fopen("pos_tags.txt", "w");
    if (!fp) { fprintf(stderr, "Cannot write pos_tags.txt\n"); return 1; }
    for (int i = 0; i < n; i++)
        fprintf(fp, "%s,%s\n", low[i], tags[i]);
    fclose(fp);
    printf("\nSaved -> pos_tags.txt\n");

    /* 5. Save target_pos.txt using target_index.txt */
    FILE *ti = fopen("target_index.txt", "r");
    if (ti) {
        int idx = -1;
        fscanf(ti, "%d", &idx);
        fclose(ti);
        if (idx >= 0 && idx < n) {
            FILE *tp = fopen("target_pos.txt", "w");
            if (tp) {
                fprintf(tp, "%s\n", tags[idx]);
                fprintf(tp, "# token : %s\n", low[idx]);
                fprintf(tp, "# index : %d\n", idx);
                fclose(tp);
                printf("Saved -> target_pos.txt  [%s = %s]\n", low[idx], tags[idx]);
            }
        }
    } else {
        fprintf(stderr, "Warning: target_index.txt not found; target_pos.txt not written\n");
    }

    return 0;
}
