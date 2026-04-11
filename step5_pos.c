/*
 * ============================================================
 * STEP 5: POS TAGGER  (PTB lookup + context rules + suffix)
 * ============================================================
 *
 * PROCESS:
 *   1. Load lowercased tokens from lowercased.txt
 *   2. Load original-cased tokens from tokens.txt
 *   3. For each token:
 *        a. Punctuation check              -> PUNCT
 *        b. Digit string                   -> CD
 *        c. Closed-class table             -> exact tag
 *        d. Known open-class table         -> most-frequent tag
 *        e. Context rules (two-pass)       -> neighbour-based tag
 *        f. Capitalised + not sent-start   -> NNP
 *        g. Suffix rules                   -> morphological guess
 *        h. Default                        -> NN
 *   4. Write pos_tags.txt   (TOKEN,POS one per line)
 *   5. Write target_pos.txt (single tag for target word)
 *
 * INPUT :  lowercased.txt  — comma-separated lowercase tokens
 *          tokens.txt      — comma-separated original-case tokens
 *          target_index.txt — integer index of target word
 *
 * OUTPUT:  pos_tags.txt    — TOKEN,POS one per line
 *          target_pos.txt  — POS of target word only
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKENS 512
#define MAX_LEN    64
#define MAX_LINE   8192

/* ── PTB tag set (subset) ───────────────────────────────────
 * CC   Coordinating conjunction         DT   Determiner
 * CD   Cardinal number                  EX   Existential there
 * IN   Preposition / sub-conj           JJ   Adjective
 * JJR  Adjective comparative            JJS  Adjective superlative
 * MD   Modal                            NN   Noun singular
 * NNS  Noun plural                      NNP  Proper noun
 * PDT  Predeterminer                    POS  Possessive ending
 * PRP  Personal pronoun                 PRP$ Possessive pronoun
 * RB   Adverb                           RBR  Adverb comparative
 * RBS  Adverb superlative               RP   Particle
 * TO   "to"                             UH   Interjection
 * VB   Verb base                        VBD  Verb past tense
 * VBG  Verb gerund/pres-part            VBN  Verb past participle
 * VBP  Verb non-3rd-sg present          VBZ  Verb 3rd-sg present
 * WDT  Wh-determiner                    WP   Wh-pronoun
 * WRB  Wh-adverb                        PUNCT Punctuation
 * ─────────────────────────────────────────────────────────── */

typedef struct { const char *word; const char *tag; } Entry;

/* ── TABLE 1: Closed-class words (always this tag) ──────────
 * Source: Jurafsky & Martin Ch.8 + PTB WSJ frequency counts  */
static const Entry closed[] = {
/* Determiners */
{"the","DT"},{"a","DT"},{"an","DT"},
{"this","DT"},{"these","DT"},{"those","DT"},
{"every","DT"},{"each","DT"},{"any","DT"},{"all","DT"},
{"both","DT"},{"either","DT"},{"neither","DT"},{"no","DT"},
{"such","DT"},{"whatever","DT"},{"whichever","DT"},
/* Possessive pronouns */
{"my","PRP$"},{"your","PRP$"},{"his","PRP$"},{"her","PRP$"},
{"its","PRP$"},{"our","PRP$"},{"their","PRP$"},{"whose","WP$"},
/* Personal pronouns */
{"i","PRP"},{"you","PRP"},{"he","PRP"},{"she","PRP"},
{"it","PRP"},{"we","PRP"},{"they","PRP"},{"me","PRP"},
{"him","PRP"},{"us","PRP"},{"them","PRP"},
{"myself","PRP"},{"yourself","PRP"},{"himself","PRP"},
{"herself","PRP"},{"itself","PRP"},{"ourselves","PRP"},
{"themselves","PRP"},
/* Wh-words */
{"who","WP"},{"whom","WP"},{"which","WDT"},
{"when","WRB"},{"where","WRB"},{"why","WRB"},{"how","WRB"},
/* Modals */
{"can","MD"},{"could","MD"},{"may","MD"},{"might","MD"},
{"shall","MD"},{"should","MD"},{"will","MD"},{"would","MD"},
{"must","MD"},{"ought","MD"},
/* Coordinating conjunctions — "for"/"so" removed (see IN/RB below) */
{"and","CC"},{"but","CC"},{"or","CC"},{"nor","CC"},{"yet","CC"},
/* Prepositions & subordinators
   NOTE: "for" -> IN (preposition ~85% in PTB; CC only in "bread and butter, for …" idiom)
         "so"  -> RB  (adverb ~70%; CC only in formal "so … that" constructions)        */
{"in","IN"},{"on","IN"},{"at","IN"},{"by","IN"},{"with","IN"},
{"of","IN"},{"to","TO"},{"from","IN"},{"into","IN"},{"onto","IN"},
{"for","IN"},{"about","IN"},{"above","IN"},{"across","IN"},{"after","IN"},
{"against","IN"},{"along","IN"},{"among","IN"},{"around","IN"},
{"as","IN"},{"because","IN"},{"before","IN"},{"behind","IN"},
{"below","IN"},{"beneath","IN"},{"beside","IN"},{"between","IN"},
{"beyond","IN"},{"during","IN"},{"except","IN"},{"inside","IN"},
{"near","IN"},{"off","IN"},{"outside","IN"},{"over","IN"},
{"past","IN"},{"since","IN"},{"through","IN"},{"throughout","IN"},
{"till","IN"},{"under","IN"},{"until","IN"},
{"upon","IN"},{"while","IN"},{"within","IN"},{"without","IN"},
{"although","IN"},{"if","IN"},{"unless","IN"},{"whether","IN"},
{"than","IN"},{"that","IN"},
/* Particles (verbal) */
{"up","RP"},{"out","RP"},{"down","RP"},{"away","RP"},
/* Existential */
{"there","EX"},
/* Predeterminer — only "half" and "both" fit true PTB PDT usage */
{"half","PDT"},
/* Interjections */
{"oh","UH"},{"ah","UH"},{"hey","UH"},{"wow","UH"},{"uh","UH"},{"um","UH"},
/* Contractions */
{"n't","RB"},{"not","RB"},{"'s","POS"},
{"'ll","MD"},{"'ve","VBP"},{"'re","VBP"},{"'d","MD"},{"'m","VBP"},
/* Numbers as words */
{"zero","CD"},{"one","CD"},{"two","CD"},{"three","CD"},
{"four","CD"},{"five","CD"},{"six","CD"},{"seven","CD"},
{"eight","CD"},{"nine","CD"},{"ten","CD"},{"eleven","CD"},
{"twelve","CD"},{"hundred","CD"},{"thousand","CD"},
{"million","CD"},{"billion","CD"},{"trillion","CD"},
{NULL,NULL}
};

/* ── TABLE 2: Known open-class words -> most-frequent PTB tag ─
 * Only words where dominant tag > 80% of PTB occurrences.     */
static const Entry known[] = {
/* ── Common verbs: base form ── */
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
{"bank","VB"},{"walk","VB"},{"talk","VB"},{"jump","VB"},
{"swim","VB"},{"fly","VB"},{"drive","VB"},{"ride","VB"},
{"climb","VB"},{"sleep","VB"},{"wake","VB"},{"drink","VB"},
{"cook","VB"},{"clean","VB"},{"wash","VB"},{"fix","VB"},
{"push","VB"},{"pull","VB"},{"carry","VB"},{"throw","VB"},
{"catch","VB"},{"drop","VB"},{"lift","VB"},{"touch","VB"},
{"watch","VB"},{"listen","VB"},{"sing","VB"},{"dance","VB"},
{"laugh","VB"},{"cry","VB"},{"smile","VB"},{"shout","VB"},
{"happen","VB"},{"occur","VB"},{"exist","VB"},{"appear","VB"},
{"matter","VB"},{"wonder","VB"},{"decide","VB"},{"choose","VB"},
{"accept","VB"},{"refuse","VB"},{"offer","VB"},{"promise","VB"},
{"agree","VB"},{"disagree","VB"},{"believe","VB"},{"doubt","VB"},
{"forget","VB"},{"forgive","VB"},{"realize","VB"},{"notice","VB"},
{"imagine","VB"},{"suppose","VB"},{"pretend","VB"},{"suggest","VB"},
{"explain","VB"},{"discuss","VB"},{"mention","VB"},{"report","VB"},
/* ── VBZ: 3rd-person singular present ── */
{"is","VBZ"},{"has","VBZ"},{"does","VBZ"},{"goes","VBZ"},
{"gets","VBZ"},{"says","VBZ"},{"makes","VBZ"},{"takes","VBZ"},
{"seems","VBZ"},{"comes","VBZ"},{"knows","VBZ"},{"gives","VBZ"},
{"wants","VBZ"},{"looks","VBZ"},{"thinks","VBZ"},{"uses","VBZ"},
{"works","VBZ"},{"needs","VBZ"},{"means","VBZ"},{"includes","VBZ"},
{"shows","VBZ"},{"helps","VBZ"},{"becomes","VBZ"},{"remains","VBZ"},
{"writes","VBZ"},{"reads","VBZ"},{"runs","VBZ"},{"speaks","VBZ"},
{"drives","VBZ"},{"lives","VBZ"},{"moves","VBZ"},{"loves","VBZ"},
{"leaves","VBZ"},{"saves","VBZ"},{"hates","VBZ"},
{"creates","VBZ"},{"provides","VBZ"},{"decides","VBZ"},
{"produces","VBZ"},{"reduces","VBZ"},{"describes","VBZ"},
{"carries","VBZ"},{"tries","VBZ"},{"studies","VBZ"},{"applies","VBZ"},
{"replies","VBZ"},{"relies","VBZ"},{"varies","VBZ"},{"supplies","VBZ"},
{"teaches","VBZ"},{"reaches","VBZ"},{"catches","VBZ"},{"watches","VBZ"},
{"pushes","VBZ"},{"rushes","VBZ"},{"finishes","VBZ"},{"wishes","VBZ"},
{"fixes","VBZ"},{"mixes","VBZ"},{"expects","VBZ"},{"suggests","VBZ"},
{"starts","VBZ"},{"ends","VBZ"},{"supports","VBZ"},{"allows","VBZ"},
{"follows","VBZ"},{"grows","VBZ"},{"draws","VBZ"},{"throws","VBZ"},
{"adds","VBZ"},{"leads","VBZ"},{"builds","VBZ"},{"holds","VBZ"},
{"finds","VBZ"},{"sends","VBZ"},{"stands","VBZ"},{"spends","VBZ"},
{"considers","VBZ"},{"remembers","VBZ"},{"understands","VBZ"},
{"represents","VBZ"},{"determines","VBZ"},{"demonstrates","VBZ"},
{"banks","VBZ"},{"walks","VBZ"},{"talks","VBZ"},{"jumps","VBZ"},
/* ── VBD: past tense ── */
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
{"banked","VBD"},{"walked","VBD"},{"talked","VBD"},{"jumped","VBD"},
{"smiled","VBD"},{"laughed","VBD"},{"cried","VBD"},{"shouted","VBD"},
/* ── VBN: past participle ── */
{"been","VBN"},{"done","VBN"},{"gone","VBN"},
{"given","VBN"},{"taken","VBN"},{"seen","VBN"},{"known","VBN"},
{"used","VBN"},{"called","VBN"},{"shown","VBN"},
{"written","VBN"},{"broken","VBN"},{"spoken","VBN"},
{"grown","VBN"},{"fallen","VBN"},{"drawn","VBN"},
{"built","VBN"},{"sent","VBN"},{"paid","VBN"},
/* ── VBG: gerund / present participle ── */
{"being","VBG"},{"having","VBG"},{"doing","VBG"},{"going","VBG"},
{"getting","VBG"},{"making","VBG"},{"taking","VBG"},
{"saying","VBG"},{"coming","VBG"},{"seeing","VBG"},
{"knowing","VBG"},{"thinking","VBG"},{"looking","VBG"},
{"working","VBG"},{"using","VBG"},{"trying","VBG"},
{"running","VBG"},{"playing","VBG"},{"writing","VBG"},
{"reading","VBG"},{"speaking","VBG"},{"standing","VBG"},
{"living","VBG"},{"moving","VBG"},{"including","VBG"},
{"showing","VBG"},{"following","VBG"},{"helping","VBG"},
{"changing","VBG"},{"leading","VBG"},
{"walking","VBG"},{"talking","VBG"},{"banking","VBG"},
{"laughing","VBG"},{"smiling","VBG"},{"crying","VBG"},
/* ── Common nouns ── */
{"time","NN"},{"year","NN"},{"people","NNS"},{"way","NN"},
{"day","NN"},{"man","NN"},{"woman","NN"},{"child","NN"},
{"children","NNS"},{"men","NNS"},{"women","NNS"},
{"feet","NNS"},{"teeth","NNS"},{"mice","NNS"},{"geese","NNS"},
{"world","NN"},{"life","NN"},{"hand","NN"},{"part","NN"},
{"place","NN"},{"case","NN"},{"week","NN"},{"company","NN"},
{"system","NN"},{"program","NN"},{"question","NN"},
{"government","NN"},{"number","NN"},{"night","NN"},{"point","NN"},
{"home","NN"},{"water","NN"},{"room","NN"},{"mother","NN"},
{"area","NN"},{"money","NN"},{"story","NN"},{"fact","NN"},
{"month","NN"},{"lot","NN"},{"study","NN"},
{"book","NN"},{"eye","NN"},{"job","NN"},
{"business","NN"},{"issue","NN"},{"side","NN"},{"kind","NN"},
{"head","NN"},{"house","NN"},{"service","NN"},{"friend","NN"},
{"father","NN"},{"power","NN"},{"hour","NN"},{"game","NN"},
{"line","NN"},{"course","NN"},{"city","NN"},{"community","NN"},
{"name","NN"},{"president","NN"},{"team","NN"},{"minute","NN"},
{"air","NN"},{"force","NN"},{"body","NN"},{"face","NN"},
{"type","NN"},{"form","NN"},{"reason","NN"},{"car","NN"},
{"level","NN"},{"age","NN"},{"policy","NN"},{"process","NN"},
{"school","NN"},{"church","NN"},{"street","NN"},{"sense","NN"},
{"law","NN"},{"voice","NN"},{"care","NN"},{"door","NN"},
{"office","NN"},{"decision","NN"},{"price","NN"},{"board","NN"},
{"plan","NN"},{"information","NN"},{"group","NN"},{"order","NN"},
{"position","NN"},{"field","NN"},{"problem","NN"},{"role","NN"},
/* -ing nouns (would be caught as VBG by suffix rule otherwise) */
{"ring","NN"},{"king","NN"},{"thing","NN"},{"spring","NN"},
{"morning","NN"},{"evening","NN"},{"ceiling","NN"},{"string","NN"},
{"wing","NN"},{"swing","NN"},{"sting","NN"},{"bring","NN"},
{"warning","NN"},{"meaning","NN"},{"feeling","NN"},{"opening","NN"},
{"reading","NN"},{"painting","NN"},{"building","NN"},{"meeting","NN"},
{"training","NN"},{"testing","NN"},{"finding","NN"},{"setting","NN"},
{"listing","NN"},{"pricing","NN"},{"ranking","NN"},{"rating","NN"},
/* ── Common adjectives ── */
/* Shape/size */
{"big","JJ"},{"small","JJ"},{"large","JJ"},{"little","JJ"},
{"tall","JJ"},{"short","JJ"},{"long","JJ"},{"wide","JJ"},
{"narrow","JJ"},{"thick","JJ"},{"thin","JJ"},{"deep","JJ"},
{"shallow","JJ"},{"flat","JJ"},{"round","JJ"},{"square","JJ"},
/* Quality */
{"good","JJ"},{"bad","JJ"},{"great","JJ"},{"poor","JJ"},
{"nice","JJ"},{"fine","JJ"},{"fair","JJ"},{"right","JJ"},
{"wrong","JJ"},{"true","JJ"},{"false","JJ"},{"real","JJ"},
{"fake","JJ"},{"pure","JJ"},{"raw","JJ"},{"ripe","JJ"},
/* Speed/strength */
{"fast","JJ"},{"slow","JJ"},{"quick","JJ"},{"strong","JJ"},
{"weak","JJ"},{"tough","JJ"},{"soft","JJ"},{"hard","JJ"},
{"firm","JJ"},{"loose","JJ"},{"tight","JJ"},{"stiff","JJ"},
/* Temperature/light */
{"hot","JJ"},{"cold","JJ"},{"warm","JJ"},{"cool","JJ"},
{"bright","JJ"},{"dark","JJ"},{"dim","JJ"},{"pale","JJ"},
{"dull","JJ"},{"shiny","JJ"},{"clear","JJ"},{"clean","JJ"},
{"dirty","JJ"},{"dry","JJ"},{"wet","JJ"},{"damp","JJ"},
/* Emotion/character */
{"happy","JJ"},{"sad","JJ"},{"angry","JJ"},{"scared","JJ"},
{"brave","JJ"},{"calm","JJ"},{"kind","JJ"},{"mean","JJ"},
{"wild","JJ"},{"tame","JJ"},{"gentle","JJ"},{"rude","JJ"},
{"smart","JJ"},{"wise","JJ"},{"silly","JJ"},{"crazy","JJ"},
{"lazy","JJ"},{"busy","JJ"},{"ready","JJ"},{"tired","JJ"},
{"sick","JJ"},{"well","JJ"},{"safe","JJ"},{"sure","JJ"},
{"glad","JJ"},{"proud","JJ"},{"shy","JJ"},{"bold","JJ"},
/* Existing entries kept */
{"new","JJ"},{"high","JJ"},{"old","JJ"},
{"american","JJ"},{"national","JJ"},{"young","JJ"},{"local","JJ"},
{"important","JJ"},{"political","JJ"},{"different","JJ"},
{"black","JJ"},{"white","JJ"},{"best","JJS"},
{"free","JJ"},{"able","JJ"},{"open","JJ"},{"public","JJ"},
{"human","JJ"},{"late","JJ"},{"possible","JJ"},
{"economic","JJ"},{"social","JJ"},{"full","JJ"},
{"federal","JJ"},{"major","JJ"},{"medical","JJ"},{"low","JJ"},
{"central","JJ"},{"current","JJ"},{"specific","JJ"},{"recent","JJ"},
{"main","JJ"},{"available","JJ"},{"special","JJ"},{"natural","JJ"},
{"final","JJ"},{"entire","JJ"},{"simple","JJ"},{"common","JJ"},
{"various","JJ"},{"particular","JJ"},{"foreign","JJ"},{"military","JJ"},
{"successful","JJ"},{"financial","JJ"},{"difficult","JJ"},
{"heavy","JJ"},{"light","JJ"},{"rough","JJ"},{"smooth","JJ"},
{"sharp","JJ"},{"blunt","JJ"},{"loud","JJ"},{"quiet","JJ"},
{"rich","JJ"},{"cheap","JJ"},{"expensive","JJ"},{"easy","JJ"},
{"complex","JJ"},{"simple","JJ"},{"normal","JJ"},{"strange","JJ"},
{"odd","JJ"},{"even","JJ"},{"plain","JJ"},{"fancy","JJ"},
{"fresh","JJ"},{"stale","JJ"},{"sweet","JJ"},{"bitter","JJ"},
{"sour","JJ"},{"salty","JJ"},{"spicy","JJ"},{"bland","JJ"},
/* Comparative / superlative */
{"more","RBR"},{"less","RBR"},{"most","RBS"},{"least","RBS"},
{"better","JJR"},{"worse","JJR"},{"older","JJR"},{"younger","JJR"},
{"larger","JJR"},{"smaller","JJR"},{"higher","JJR"},{"lower","JJR"},
{"bigger","JJR"},{"longer","JJR"},{"shorter","JJR"},{"faster","JJR"},
{"slower","JJR"},{"stronger","JJR"},{"weaker","JJR"},{"warmer","JJR"},
{"colder","JJR"},{"brighter","JJR"},{"darker","JJR"},{"smarter","JJR"},
{"happier","JJR"},{"angrier","JJR"},{"easier","JJR"},{"heavier","JJR"},
{"busier","JJR"},{"funnier","JJR"},{"friendlier","JJR"},
{"happiest","JJS"},{"angriest","JJS"},{"easiest","JJS"},
{"heaviest","JJS"},{"busiest","JJS"},{"funniest","JJS"},
{"friendliest","JJS"},{"brightest","JJS"},{"darkest","JJS"},
{"smartest","JJS"},{"strongest","JJS"},{"fastest","JJS"},
{"slowest","JJS"},{"warmest","JJS"},{"coldest","JJS"},
/* Common adverbs */
{"also","RB"},{"just","RB"},{"very","RB"},{"now","RB"},
{"still","RB"},{"even","RB"},{"already","RB"},{"never","RB"},
{"always","RB"},{"often","RB"},{"again","RB"},{"here","RB"},
{"then","RB"},{"only","RB"},{"first","RB"},{"last","RB"},
{"soon","RB"},{"too","RB"},{"once","RB"},{"almost","RB"},
{"probably","RB"},{"perhaps","RB"},{"certainly","RB"},
{"usually","RB"},{"quickly","RB"},{"finally","RB"},
{"recently","RB"},{"simply","RB"},{"actually","RB"},
{"especially","RB"},{"really","RB"},{"together","RB"},
{"likely","RB"},{"instead","RB"},{"so","RB"},
{"quite","RB"},{"rather","RB"},{"fairly","RB"},{"pretty","RB"},
{"well","RB"},{"back","RB"},{"forward","RB"},{"ahead","RB"},
{"apart","RB"},{"abroad","RB"},{"aside","RB"},{"forth","RB"},
{"hence","RB"},{"thus","RB"},{"otherwise","RB"},{"elsewhere","RB"},
{"meanwhile","RB"},{"nevertheless","RB"},{"nonetheless","RB"},
{"furthermore","RB"},{"moreover","RB"},{"therefore","RB"},
{"however","RB"},{"somehow","RB"},{"somewhere","RB"},{"sometime","RB"},
{"somewhat","RB"},{"anyway","RB"},{"anywhere","RB"},
{"everywhere","RB"},{"nowhere","RB"},{"sometimes","RB"},
{"seldom","RB"},{"enough","RB"},{"far","RB"},{"straight","RB"},
{"aloud","RB"},{"anew","RB"},{"afar","RB"},{"astray","RB"},
{"aloft","RB"},{"afoot","RB"},{"ashore","RB"},{"abreast","RB"},
{NULL,NULL}
};

/* ── Helpers ─────────────────────────────────────────────── */

static int is_punct(const char *w) {
    for (const char *p = w; *p; p++)
        if (isalnum((unsigned char)*p)) return 0;
    return 1;
}

static int is_digits(const char *w) {
    if (!*w) return 0;
    for (const char *p = w; *p; p++)
        if (!isdigit((unsigned char)*p) && *p != '.' && *p != ',') return 0;
    return 1;
}

static int endswith(const char *s, const char *sfx) {
    int sl = strlen(s), fl = strlen(sfx);
    return sl >= fl && strcmp(s + sl - fl, sfx) == 0;
}

static const char *table_lookup(const Entry *tbl, const char *w) {
    for (int i = 0; tbl[i].word; i++)
        if (strcmp(tbl[i].word, w) == 0) return tbl[i].tag;
    return NULL;
}

/* Return 1 if tag is a linking/copula verb */
static int is_linking_verb(const char *w) {
    static const char *lv[] = {
        "is","are","was","were","be","been","being","am",
        "seem","seems","seemed","appear","appears","appeared",
        "become","becomes","became","feel","feels","felt",
        "look","looks","looked","sound","sounds","sounded",
        "smell","smells","smelled","taste","tastes","tasted",
        "remain","remains","remained","stay","stays","stayed",
        "get","got","grow","grew",NULL
    };
    for (int i = 0; lv[i]; i++)
        if (strcmp(w, lv[i]) == 0) return 1;
    return 0;
}

/* ── Suffix rules (Jurafsky §8 + extended) ──────────────── */
static const char *suffix_pos(const char *w) {
    int len = strlen(w);
    if (len < 3) return "NN";

    /* Superlative -est */
    if (endswith(w,"est") && len > 5) return "JJS";
    /* Comparative -er (only if likely adjective-like root)
       Heuristic: if stem ends in consonant cluster -> JJR, else NN */
    /* -ing -> VBG (but -ing nouns handled in known[] table above) */
    if (endswith(w,"ing") && len > 5) return "VBG";
    /* -tion / -sion -> NN */
    if (endswith(w,"tion")) return "NN";
    if (endswith(w,"sion")) return "NN";
    /* -ness -> NN */
    if (endswith(w,"ness")) return "NN";
    /* -ment -> NN */
    if (endswith(w,"ment")) return "NN";
    /* -ity -> NN */
    if (endswith(w,"ity")) return "NN";
    /* -ism -> NN */
    if (endswith(w,"ism")) return "NN";
    /* -ist -> NN */
    if (endswith(w,"ist")) return "NN";
    /* -ly -> RB */
    if (endswith(w,"ly") && len > 4) return "RB";
    /* -ful -> JJ */
    if (endswith(w,"ful")) return "JJ";
    /* -ous -> JJ */
    if (endswith(w,"ous")) return "JJ";
    /* -ive -> JJ */
    if (endswith(w,"ive")) return "JJ";
    /* -able / -ible -> JJ */
    if (endswith(w,"able")) return "JJ";
    if (endswith(w,"ible")) return "JJ";
    /* -al -> JJ */
    if (endswith(w,"al") && len > 4) return "JJ";
    /* -ish -> JJ */
    if (endswith(w,"ish")) return "JJ";
    /* -less -> JJ */
    if (endswith(w,"less")) return "JJ";
    /* -ic -> JJ (e.g. magic, comic, classic) */
    if (endswith(w,"ic") && len > 4) return "JJ";
    /* -ed -> VBD (walked, talked) */
    if (endswith(w,"ed")) return "VBD";
    /* -en -> VBN (broken, taken) */
    if (endswith(w,"en") && len > 4) return "VBN";
    /* -er -> NN (actor, runner) — but not comparatives; those are in known[] */
    if (endswith(w,"er") && len > 4) return "NN";
    if (endswith(w,"or") && len > 4) return "NN";
    /* -s -> NNS */
    if (endswith(w,"s") && len > 3) return "NNS";
    return "NN";
}

/* Load comma-separated line -> token array, returns count */
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

/* ── Context-based tag correction (two-pass) ──────────────
 * Pass 1: forward pass using left neighbour's tag
 * Pass 2: backward pass using right neighbour's tag
 * Rules are applied only when the initial tag is uncertain
 * (i.e. came from suffix fallback or default NN).           */
static void apply_context_rules(int n,
    char low[][MAX_LEN], char tags[][8], int from_table[])
{
    /* Pass 1: left-to-right */
    for (int i = 1; i < n; i++) {
        const char *prev_tag = tags[i-1];
        const char *prev_w   = low[i-1];
        const char *cur_tag  = tags[i];
        const char *cur_w    = low[i];

        /* Rule 1: DT/PDT/PRP$ + unknown -> likely JJ or NN
           If the word right-after DT is not already JJ/NN/NNS/NNP
           and not a closed-class word, and was tagged NN by default,
           keep as NN (determiners precede nouns — correct default).
           BUT if i+1 exists and is NN/NNS, re-tag current as JJ.  */
        if ((strcmp(prev_tag,"DT")==0 || strcmp(prev_tag,"PDT")==0 ||
             strcmp(prev_tag,"PRP$")==0) && !from_table[i]) {
            /* look ahead: if next token is a noun -> current is adjective */
            if (i + 1 < n) {
                const char *next_tag = tags[i+1];
                if (strcmp(next_tag,"NN")==0 || strcmp(next_tag,"NNS")==0 ||
                    strcmp(next_tag,"NNP")==0) {
                    strncpy(tags[i], "JJ", 7);
                }
            }
        }

        /* Rule 2: Linking verb + unknown word -> predicate adjective JJ
           e.g. "is bright", "was calm", "seems tired"                  */
        if (is_linking_verb(prev_w) && !from_table[i]) {
            if (strcmp(cur_tag,"NN")==0 || strcmp(cur_tag,"NNS")==0) {
                strncpy(tags[i], "JJ", 7);
            }
        }

        /* Rule 3: MD/TO + word -> VB (modal/to always precedes base verb)
           e.g. "will bank", "to bank"                                   */
        if ((strcmp(prev_tag,"MD")==0 || strcmp(prev_tag,"TO")==0) &&
            !from_table[i]) {
            strncpy(tags[i], "VB", 7);
        }

        /* Rule 4: RB/JJ + unknown word -> possible JJ (adverb intensifier)
           e.g. "very bright", "quite dark"                              */
        if ((strcmp(prev_tag,"RB")==0 || strcmp(prev_tag,"JJ")==0) &&
            !from_table[i] &&
            (strcmp(cur_tag,"NN")==0 || strcmp(cur_tag,"NNS")==0)) {
            /* Only promote to JJ if next token is a noun or nothing */
            if (i + 1 >= n ||
                strcmp(tags[i+1],"NN")==0 || strcmp(tags[i+1],"NNS")==0 ||
                strcmp(tags[i+1],"PUNCT")==0) {
                strncpy(tags[i], "JJ", 7);
            }
        }

        /* Rule 5: PRP/NNP/NN + unknown -> likely VB (subject precedes verb)
           Only if current word is NOT already a known-table word          */
        if ((strcmp(prev_tag,"PRP")==0 || strcmp(prev_tag,"NNP")==0 ||
             strcmp(prev_tag,"NN")==0  || strcmp(prev_tag,"NNS")==0) &&
            !from_table[i] && strcmp(cur_tag,"NN")==0) {
            /* Promote to VBP (non-3rd-sg) only if next is IN/DT/PRP/RB */
            if (i + 1 < n) {
                const char *nt = tags[i+1];
                if (strcmp(nt,"IN")==0 || strcmp(nt,"DT")==0 ||
                    strcmp(nt,"PRP")==0|| strcmp(nt,"RB")==0 ||
                    strcmp(nt,"TO")==0) {
                    strncpy(tags[i], "VBP", 7);
                }
            }
        }

        /* Rule 6: CC/IN + PRP/NNP/NN -> that pronoun/noun is fine; skip */

        /* Rule 7: VBD/VBN/VBZ + unknown -> consider predicate adj/NN     */
        (void)cur_w; /* suppress unused warning */
    }

    /* Pass 2: right-to-left (catch words missed in pass 1) */
    for (int i = n - 2; i >= 0; i--) {
        const char *next_tag = tags[i+1];

        /* Rule 8: unknown + NN/NNS -> likely JJ (noun-modifier)          */
        if (!from_table[i] &&
            (strcmp(tags[i],"NN")==0 || strcmp(tags[i],"NNS")==0) &&
            (strcmp(next_tag,"NN")==0 || strcmp(next_tag,"NNS")==0 ||
             strcmp(next_tag,"NNP")==0)) {
            /* Only if left neighbour is DT/PDT/PRP$ or start of sentence */
            if (i == 0 || strcmp(tags[i-1],"DT")==0 ||
                strcmp(tags[i-1],"PDT")==0 || strcmp(tags[i-1],"PRP$")==0) {
                strncpy(tags[i], "JJ", 7);
            }
        }
    }
}

/* ── Main ─────────────────────────────────────────────────── */
int main(void) {
    char low [MAX_TOKENS][MAX_LEN];
    char orig[MAX_TOKENS][MAX_LEN];
    char tags[MAX_TOKENS][8];
    int  from_table[MAX_TOKENS]; /* 1 = tag from closed/known table */

    /* 1. Load tokens */
    int n  = load_csv("lowercased.txt", low,  MAX_TOKENS);
    int no = load_csv("tokens.txt",     orig, MAX_TOKENS);
    if (n <= 0) {
        fprintf(stderr, "ERROR: cannot read lowercased.txt\n");
        return 1;
    }
    if (no != n) {
        fprintf(stderr,
            "Warning: token count mismatch (%d vs %d); "
            "using lowercase for capitalisation check\n", n, no);
        for (int i = 0; i < n; i++) strncpy(orig[i], low[i], MAX_LEN-1);
    }

    /* 2. First-pass tagging (no context yet) */
    for (int i = 0; i < n; i++) {
        const char *tag = NULL;
        from_table[i] = 0;

        /* (a) Punctuation */
        if (is_punct(low[i])) { tag = "PUNCT"; goto done; }

        /* (b) Number */
        if (is_digits(low[i])) { tag = "CD"; goto done; }

        /* (c) Closed-class exact match */
        tag = table_lookup(closed, low[i]);
        if (tag) { from_table[i] = 1; goto done; }

        /* (d) Known open-class match */
        tag = table_lookup(known, low[i]);
        if (tag) { from_table[i] = 1; goto done; }

        /* (e) Capitalised + not sentence-start -> proper noun */
        if (isupper((unsigned char)orig[i][0])) {
            int sent_start = (i == 0);
            if (!sent_start && i > 0) {
                char p = low[i-1][0];
                if (p == '.' || p == '!' || p == '?') sent_start = 1;
            }
            if (!sent_start) { tag = "NNP"; goto done; }
        }

        /* (f) Suffix rules */
        tag = suffix_pos(low[i]);

    done:
        strncpy(tags[i], tag ? tag : "NN", 7);
        tags[i][7] = '\0';
    }

    /* 3. Context-based correction pass */
    apply_context_rules(n, low, tags, from_table);

    /* 4. Print to console */
    printf("=== STEP 5: POS TAGGING (PTB lookup + context + suffix) ===\n\n");
    printf("%-4s %-22s %s\n", "IDX", "TOKEN", "POS");
    printf("%-4s %-22s %s\n", "---", "-----", "---");
    for (int i = 0; i < n; i++)
        printf("%-4d %-22s %s\n", i, low[i], tags[i]);

    /* 5. Save pos_tags.txt */
    FILE *fp = fopen("pos_tags.txt", "w");
    if (!fp) { fprintf(stderr, "Cannot write pos_tags.txt\n"); return 1; }
    for (int i = 0; i < n; i++)
        fprintf(fp, "%s,%s\n", low[i], tags[i]);
    fclose(fp);
    printf("\nSaved -> pos_tags.txt\n");

    /* 6. Save target_pos.txt */
    FILE *ti = fopen("target_index.txt", "r");
    if (ti) {
        int idx = -1;
        fscanf(ti, "%d", &idx);
        fclose(ti);
        if (idx >= 0 && idx < n) {
            FILE *tp = fopen("target_pos.txt", "w");
            if (tp) {
                fprintf(tp, "%s\n",      tags[idx]);
                fprintf(tp, "# token : %s\n", low[idx]);
                fprintf(tp, "# index : %d\n", idx);
                fclose(tp);
                printf("Saved -> target_pos.txt [%s = %s]\n", low[idx], tags[idx]);
            }
        }
    } else {
        fprintf(stderr,
            "Warning: target_index.txt not found; target_pos.txt not written\n");
    }

    return 0;
}