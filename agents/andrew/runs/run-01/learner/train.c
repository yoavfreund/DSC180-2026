/* Viola-Jones style AdaBoost with Haar features on 19x19 patches.
 * Feature values are pre-quantized into 256 per-feature quantile bins so each
 * boosting round is a weighted histogram scan over all features. */
#include "common.h"
#include <pthread.h>

#define NB 256
#define NTH 10

static int NF, N;
static Feat *feats;
static unsigned char *bins;  /* NF x N */
static float *edges;         /* NF x (NB-1) */
static double *iis;          /* N x II*II */
static int *lab;
static double *wt;

typedef struct { int f0, f1; double err; int f, k, pol; } Job;

static int cmpf(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

static void *prep_worker(void *arg) {
    Job *j = arg;
    float *v = malloc(sizeof(float) * N), *s = malloc(sizeof(float) * N);
    for (int f = j->f0; f < j->f1; f++) {
        for (int i = 0; i < N; i++) v[i] = (float)feat_val(&feats[f], iis + (size_t)i * II * II);
        memcpy(s, v, sizeof(float) * N);
        qsort(s, N, sizeof(float), cmpf);
        float *e = edges + (size_t)f * (NB - 1);
        for (int k = 0; k < NB - 1; k++) {
            int idx = (int)((double)(k + 1) * N / NB);
            if (idx >= N) idx = N - 1;
            e[k] = s[idx];
        }
        unsigned char *b = bins + (size_t)f * N;
        for (int i = 0; i < N; i++) {
            /* bin = number of edges <= v */
            int lo = 0, hi = NB - 1;
            while (lo < hi) { int mid = (lo + hi) / 2; if (e[mid] <= v[i]) lo = mid + 1; else hi = mid; }
            b[i] = (unsigned char)lo;
        }
    }
    free(v); free(s);
    return NULL;
}

static void *round_worker(void *arg) {
    Job *j = arg;
    double hp[NB], hn[NB];
    double tp = 0, tn = 0;
    for (int i = 0; i < N; i++) { if (lab[i]) tp += wt[i]; else tn += wt[i]; }
    j->err = 1e300;
    for (int f = j->f0; f < j->f1; f++) {
        memset(hp, 0, sizeof hp); memset(hn, 0, sizeof hn);
        const unsigned char *b = bins + (size_t)f * N;
        for (int i = 0; i < N; i++) { if (lab[i]) hp[b[i]] += wt[i]; else hn[b[i]] += wt[i]; }
        double cp = 0, cn = 0;
        for (int k = 0; k < NB - 1; k++) {
            cp += hp[k]; cn += hn[k];
            /* pol +1: face if bin<=k (v < edge[k]) */
            double e1 = cn + (tp - cp);
            double e2 = cp + (tn - cn);
            if (e1 < j->err) { j->err = e1; j->f = f; j->k = k; j->pol = 1; }
            if (e2 < j->err) { j->err = e2; j->f = f; j->k = k; j->pol = -1; }
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: train <images_dir> <labels.txt> <model_out>\n"); return 1; }
    int T = 600;
    const char *env = getenv("VJ_ROUNDS");
    if (env) T = atoi(env);

    FILE *lf = fopen(argv[2], "r");
    if (!lf) { perror("labels"); return 1; }
    int cap = 1024, n0 = 0;
    char (*ids)[64] = malloc(cap * 64);
    int *l0 = malloc(cap * sizeof(int));
    char idb[64]; int lv;
    while (fscanf(lf, "%63s %d", idb, &lv) == 2) {
        if (n0 == cap) { cap *= 2; ids = realloc(ids, cap * 64); l0 = realloc(l0, cap * sizeof(int)); }
        strcpy(ids[n0], idb); l0[n0] = lv ? 1 : 0; n0++;
    }
    fclose(lf);
    int npos = 0; for (int i = 0; i < n0; i++) npos += l0[i];
    int ntot = n0 + npos; /* faces augmented with mirror */
    iis = malloc(sizeof(double) * II * II * ntot);
    lab = malloc(sizeof(int) * ntot);
    N = 0;
    char path[4096];
    unsigned char px[NPIX];
    for (int i = 0; i < n0; i++) {
        snprintf(path, sizeof path, "%s/%s.pgm", argv[1], ids[i]);
        if (read_pgm(path, px)) { fprintf(stderr, "skip %s\n", path); continue; }
        make_integral(px, 0, iis + (size_t)N * II * II); lab[N++] = l0[i];
        if (l0[i]) { make_integral(px, 1, iis + (size_t)N * II * II); lab[N++] = 1; }
    }
    int P = 0; for (int i = 0; i < N; i++) P += lab[i];
    fprintf(stderr, "samples %d (pos %d neg %d)\n", N, P, N - P);

    NF = enum_features(NULL);
    feats = malloc(sizeof(Feat) * NF);
    enum_features(feats);
    fprintf(stderr, "features %d\n", NF);
    bins = malloc((size_t)NF * N);
    edges = malloc(sizeof(float) * (size_t)NF * (NB - 1));

    pthread_t th[NTH]; Job jobs[NTH];
    for (int t = 0; t < NTH; t++) {
        jobs[t].f0 = (int)((long)NF * t / NTH); jobs[t].f1 = (int)((long)NF * (t + 1) / NTH);
        pthread_create(&th[t], NULL, prep_worker, &jobs[t]);
    }
    for (int t = 0; t < NTH; t++) pthread_join(th[t], NULL);
    fprintf(stderr, "quantized\n");

    wt = malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) wt[i] = lab[i] ? 0.5 / P : 0.5 / (N - P);

    FILE *mf = fopen(argv[3], "w");
    if (!mf) { perror("model"); return 1; }
    fprintf(mf, "%d\n", T);
    for (int r = 0; r < T; r++) {
        double sw = 0; for (int i = 0; i < N; i++) sw += wt[i];
        for (int i = 0; i < N; i++) wt[i] /= sw;
        for (int t = 0; t < NTH; t++) pthread_create(&th[t], NULL, round_worker, &jobs[t]);
        for (int t = 0; t < NTH; t++) pthread_join(th[t], NULL);
        Job best = jobs[0];
        for (int t = 1; t < NTH; t++) if (jobs[t].err < best.err) best = jobs[t];
        double err = best.err;
        if (err < 1e-10) err = 1e-10;
        double beta = err / (1 - err);
        double alpha = log(1 / beta);
        const unsigned char *b = bins + (size_t)best.f * N;
        for (int i = 0; i < N; i++) {
            int pred = best.pol > 0 ? (b[i] <= best.k) : (b[i] > best.k);
            if (pred == lab[i]) wt[i] *= beta;
        }
        Feat *f = &feats[best.f];
        float thr = edges[(size_t)best.f * (NB - 1) + best.k];
        fprintf(mf, "%d %d %d %d %d %.9g %d %.9g\n", f->type, f->x, f->y, f->w, f->h, thr, best.pol, alpha);
        if (r % 50 == 0) fprintf(stderr, "round %d err %.4f feat %d\n", r, err, best.f);
    }
    fclose(mf);
    return 0;
}
