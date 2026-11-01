/*
 * eval: shared face-classifier evaluator.
 *
 *   eval SCORES KEY --fp-rates  r1,r2,...   (FP rate targets, e.g. 0.001,0.01,0.1)
 *   eval SCORES KEY --fp-counts c1,c2,...   (FP count targets, e.g. 10,50,100)
 *
 * SCORES: one "<id> <score>" line per image, from the classifier's predict.
 *         Higher score = more face-like. Any finite double.
 * KEY:    one "<id> <label>" line per image, label 1 = face, 0 = non-face.
 *         Owned by the harness; classifiers never see it.
 *
 * The ids in SCORES must match the ids in KEY exactly (no missing, duplicate
 * or extra ids, no NaN/inf). Otherwise eval prints why to stderr and exits 1
 * without scoring anything.
 *
 * Scoring: predict face when score >= threshold. Thresholds are swept over
 * groups of tied scores, so tied images are always all in or all out. For
 * each FP target, the reported operating point is the one with the most true
 * positives whose false positives stay <= the target (a rate target r allows
 * floor(r * n_neg) false positives); among points with equal true positives,
 * the one with the fewest false positives.
 *
 * Output (stdout), fixed key=value lines:
 *   n_pos=... n_neg=... auc=...
 *   fp_rate=... fp_allowed=... threshold=... tp=... fp=... tpr=... fpr=... precision=...
 * (fp_count= instead of fp_rate= with --fp-counts; precision=NA when nothing
 * is predicted positive; threshold=inf when no threshold meets the target.)
 */
#define _POSIX_C_SOURCE 200809L /* strtok_r under -std=c99 on Linux */
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ID 64
#define MAX_LINE 512
#define MAX_TARGETS 16

typedef struct {
    char id[MAX_ID];
    int label;
    double score;
    int seen;
} Item;

__attribute__((format(printf, 1, 2), noreturn)) static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "eval: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

static int cmp_id(const void *a, const void *b) {
    return strcmp(((const Item *)a)->id, ((const Item *)b)->id);
}

static int cmp_score_desc(const void *a, const void *b) {
    double x = ((const Item *)a)->score, y = ((const Item *)b)->score;
    return (x < y) - (x > y);
}

/* Split a line into exactly two whitespace-separated tokens. Returns 0 if the
 * line is blank, 1 on success, -1 if malformed. */
static int two_tokens(char *line, char **t1, char **t2) {
    char *save, *extra;
    *t1 = strtok_r(line, " \t\r\n", &save);
    if (!*t1) return 0;
    *t2 = strtok_r(NULL, " \t\r\n", &save);
    extra = strtok_r(NULL, " \t\r\n", &save);
    return (*t2 && !extra) ? 1 : -1;
}

static Item *read_key(const char *path, long *n_out) {
    FILE *f = fopen(path, "r");
    if (!f) die("cannot open key %s", path);
    long n = 0, cap = 1024, lineno = 0;
    Item *items = malloc(cap * sizeof *items);
    char line[MAX_LINE], *id, *lab;
    while (fgets(line, sizeof line, f)) {
        lineno++;
        int r = two_tokens(line, &id, &lab);
        if (r == 0) continue;
        if (r < 0) die("key %s line %ld: expected '<id> <label>'", path, lineno);
        if (strlen(id) >= MAX_ID) die("key %s line %ld: id too long", path, lineno);
        if (strcmp(lab, "0") && strcmp(lab, "1")) die("key %s line %ld: label must be 0 or 1", path, lineno);
        if (n == cap) items = realloc(items, (cap *= 2) * sizeof *items);
        strcpy(items[n].id, id);
        items[n].label = lab[0] - '0';
        items[n].seen = 0;
        n++;
    }
    fclose(f);
    qsort(items, n, sizeof *items, cmp_id);
    for (long i = 1; i < n; i++)
        if (!strcmp(items[i].id, items[i - 1].id)) die("key %s: duplicate id %s", path, items[i].id);
    *n_out = n;
    return items;
}

static void read_scores(const char *path, Item *items, long n) {
    FILE *f = fopen(path, "r");
    if (!f) die("cannot open scores %s", path);
    long lineno = 0, matched = 0;
    char line[MAX_LINE], *id, *tok, *end;
    while (fgets(line, sizeof line, f)) {
        lineno++;
        if (!strchr(line, '\n') && !feof(f)) die("scores %s line %ld: line too long", path, lineno);
        int r = two_tokens(line, &id, &tok);
        if (r == 0) continue;
        if (r < 0) die("scores %s line %ld: expected '<id> <score>'", path, lineno);
        if (strlen(id) >= MAX_ID) die("scores %s line %ld: id too long", path, lineno);
        errno = 0;
        double s = strtod(tok, &end);
        if (*end || end == tok || errno == ERANGE || !isfinite(s))
            die("scores %s line %ld: score is not a finite number", path, lineno);
        Item probe;
        strcpy(probe.id, id);
        Item *it = bsearch(&probe, items, n, sizeof *items, cmp_id);
        if (!it) die("scores %s line %ld: id not in key", path, lineno);
        if (it->seen) die("scores %s line %ld: duplicate id", path, lineno);
        it->seen = 1;
        it->score = s;
        matched++;
    }
    fclose(f);
    if (matched != n) {
        for (long i = 0; i < n; i++)
            if (!items[i].seen) die("scores %s: missing id %s (%ld ids missing in total)", path, items[i].id, n - matched);
    }
}

static int parse_targets(const char *s, double *out) {
    int k = 0;
    char buf[MAX_LINE], *save, *tok, *end;
    snprintf(buf, sizeof buf, "%s", s);
    for (tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        if (k == MAX_TARGETS) die("too many FP targets (max %d)", MAX_TARGETS);
        out[k] = strtod(tok, &end);
        if (*end || end == tok || !isfinite(out[k]) || out[k] < 0) die("bad FP target '%s'", tok);
        k++;
    }
    if (k == 0) die("no FP targets given");
    return k;
}

int main(int argc, char **argv) {
    if (argc != 5 || (strcmp(argv[3], "--fp-rates") && strcmp(argv[3], "--fp-counts"))) {
        fprintf(stderr, "usage: eval SCORES KEY (--fp-rates r1,r2,... | --fp-counts c1,c2,...)\n");
        return 2;
    }
    int by_rate = !strcmp(argv[3], "--fp-rates");
    double targets[MAX_TARGETS];
    int n_targets = parse_targets(argv[4], targets);

    long n;
    Item *items = read_key(argv[2], &n);
    read_scores(argv[1], items, n);

    long n_pos = 0;
    for (long i = 0; i < n; i++) n_pos += items[i].label;
    long n_neg = n - n_pos;
    if (n_pos == 0 || n_neg == 0) die("key needs both classes (%ld positives, %ld negatives)", n_pos, n - n_pos);

    qsort(items, n, sizeof *items, cmp_score_desc);

    /* Operating points after each tie group: cum_tp[g], cum_fp[g], thresh[g].
     * Point 0 is "predict nothing" (threshold = +inf). */
    long *cum_tp = malloc((n + 1) * sizeof *cum_tp);
    long *cum_fp = malloc((n + 1) * sizeof *cum_fp);
    double *thresh = malloc((n + 1) * sizeof *thresh);
    long g = 0, tp = 0, fp = 0;
    double auc_num = 0;
    cum_tp[0] = cum_fp[0] = 0;
    thresh[0] = INFINITY;
    for (long i = 0; i < n;) {
        long j = i, gp = 0, gn = 0;
        while (j < n && items[j].score == items[i].score) {
            if (items[j].label) gp++; else gn++;
            j++;
        }
        tp += gp;
        fp += gn;
        /* Positives in this group outrank every negative below it; ties count half. */
        auc_num += gp * (double)(n_neg - fp) + 0.5 * gp * gn;
        g++;
        cum_tp[g] = tp;
        cum_fp[g] = fp;
        thresh[g] = items[i].score;
        i = j;
    }

    printf("n_pos=%ld n_neg=%ld auc=%.6f\n", n_pos, n_neg, auc_num / ((double)n_pos * n_neg));
    for (int t = 0; t < n_targets; t++) {
        long allowed = by_rate ? (long)floor(targets[t] * n_neg + 1e-9) : (long)floor(targets[t] + 1e-9);
        /* Most TPs within the FP budget; among equal TPs, the earliest point (fewest FPs). */
        long best = 0;
        for (long k = 1; k <= g && cum_fp[k] <= allowed; k++)
            if (cum_tp[k] > cum_tp[best]) best = k;
        long btp = cum_tp[best], bfp = cum_fp[best];
        if (by_rate) printf("fp_rate=%g", targets[t]);
        else printf("fp_count=%g", targets[t]);
        printf(" fp_allowed=%ld threshold=%.17g tp=%ld fp=%ld tpr=%.6f fpr=%.6f", allowed, thresh[best], btp, bfp,
               (double)btp / n_pos, (double)bfp / n_neg);
        if (btp + bfp) printf(" precision=%.6f\n", (double)btp / (btp + bfp));
        else printf(" precision=NA\n");
    }
    free(cum_tp);
    free(cum_fp);
    free(thresh);
    free(items);
    return 0;
}
