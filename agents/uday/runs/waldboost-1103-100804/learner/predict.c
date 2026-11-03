#include "feat.h"
typedef struct { Feat f; double e[NB - 1], h[NB], thr; } W1;
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    int describe = !strcmp(argv[1], "--describe");
    FILE *f = fopen(describe ? argv[2] : argv[1], "r");
    if (!f) return 1;
    int T;
    if (fscanf(f, "waldboost %d", &T) != 1) return 1;
    W1 *w = calloc(T, sizeof *w);
    for (int t = 0; t < T; t++) {
        if (fscanf(f, "%d", &w[t].f.n) != 1) return 1;
        for (int r = 0; r < w[t].f.n; r++)
            if (fscanf(f, "%d %d %d %d %lf", &w[t].f.r[r][0], &w[t].f.r[r][1], &w[t].f.r[r][2], &w[t].f.r[r][3], &w[t].f.wt[r]) != 5) return 1;
        for (int b = 0; b < NB - 1; b++) if (fscanf(f, "%lf", &w[t].e[b]) != 1) return 1;
        for (int b = 0; b < NB; b++) if (fscanf(f, "%lf", &w[t].h[b]) != 1) return 1;
        if (fscanf(f, "%lf", &w[t].thr) != 1) return 1;
    }
    fclose(f);
    if (describe) { for (int t = 1; t <= T; t++) printf("stage=%d weak=1\n", t); return 0; }
    int n = 0;
    char **ids = list_ids(argv[2], &n);
    if (!ids) return 1;
    char path[4096];
    double raw[NPIX], ii[(W + 1) * (H + 1)];
    for (int i = 0; i < n; i++) {
        snprintf(path, sizeof path, "%s/%s.pgm", argv[2], ids[i]);
        if (read_pgm(path, raw)) return 1;
        prep(raw, ii);
        double sum = 0, margin = 0; int st = T - 1;
        for (int t = 0; t < T; t++) {
            sum += w[t].h[bin_of(w[t].e, feat_eval(&w[t].f, ii))];
            margin = sum - w[t].thr;
            if (sum < w[t].thr) { st = t; break; }
        }
        double m = margin / 4.0; if (m > 12) m = 12; if (m < -12) m = -12;
        printf("%s %.17g\n", ids[i], cascade_score(st, m));
    }
    return 0;
}
