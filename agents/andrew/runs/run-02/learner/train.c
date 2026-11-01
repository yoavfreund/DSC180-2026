/* Viola-Jones style attentional cascade trained with Real AdaBoost stumps.
 * See README.md for the design. */
#include "common.h"
#include <pthread.h>
#include <unistd.h>
#include <float.h>

/* ---------------- hyperparameters ---------------- */
#define NBINS 64          /* quantisation of feature values for stump search */
#define FMAX 32000        /* random subset of the Haar feature pool */
#define CALIB_FRAC 0.15   /* faces held out to set stage thresholds */
#define STAGE_DET 0.998   /* per-stage detection quantile on held-out faces */
#define SAFETY 1.0        /* extra slack below that quantile, in calib-margin std units */
#define MAX_EARLY 4       /* thresholded stages before the final one */
static const int   early_max[MAX_EARLY]  = {6, 12, 30, 60};
static const float early_fp[MAX_EARLY]   = {0.50f, 0.45f, 0.40f, 0.40f};
#define MIN_NEG 1500      /* stop adding early stages when fewer survivors remain */
#define NEG_CAP 6000      /* negatives per stage training set */
#define FINAL_WEAK 250    /* weak classifiers in the final (unthresholded) stage */

static int nthreads = 4;
static unsigned long long rng = 88172645463325252ULL;
static unsigned rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (unsigned)(rng >> 11); }

/* ---------------- parallel for ---------------- */
typedef void (*job_fn)(int lo, int hi, int tid, void *ctx);
typedef struct { job_fn fn; int lo, hi, tid; void *ctx; } Job;
static void *job_run(void *a) { Job *j = a; j->fn(j->lo, j->hi, j->tid, j->ctx); return NULL; }
static void pfor(int n, job_fn fn, void *ctx) {
    pthread_t th[64]; Job jb[64];
    int T = nthreads;
    for (int t = 0; t < T; t++) {
        jb[t] = (Job){fn, (int)((long)n * t / T), (int)((long)n * (t + 1) / T), t, ctx};
        pthread_create(&th[t], NULL, job_run, &jb[t]);
    }
    for (int t = 0; t < T; t++) pthread_join(th[t], NULL);
}

/* ---------------- features ---------------- */
static Feature *feats; static int nfeat;
static void add_feat(Feature f) { feats[nfeat++] = f; }
static void gen_features(void) {
    int cap = 200000; feats = malloc(cap * sizeof(Feature)); nfeat = 0;
    for (int h = 1; h <= W; h++) for (int w = 1; w <= W; w++)
        for (int y = 0; y + h <= W; y++) for (int x = 0; x + w <= W; x++) {
            Feature f;
            if (w % 2 == 0) { /* left/right */
                f.nr = 2; f.r[0] = (Rect){x, y, w, h, 1}; f.r[1] = (Rect){x, y, w / 2, h, -2}; add_feat(f); }
            if (h % 2 == 0) { /* top/bottom */
                f.nr = 2; f.r[0] = (Rect){x, y, w, h, 1}; f.r[1] = (Rect){x, y, w, h / 2, -2}; add_feat(f); }
            if (w % 3 == 0) { /* horizontal three-band */
                f.nr = 2; f.r[0] = (Rect){x, y, w, h, 1}; f.r[1] = (Rect){x + w / 3, y, w / 3, h, -3}; add_feat(f); }
            if (h % 3 == 0) { /* vertical three-band */
                f.nr = 2; f.r[0] = (Rect){x, y, w, h, 1}; f.r[1] = (Rect){x, y + h / 3, w, h / 3, -3}; add_feat(f); }
            if (w % 2 == 0 && h % 2 == 0) { /* checkerboard */
                f.nr = 3; f.r[0] = (Rect){x, y, w, h, 1}; f.r[1] = (Rect){x, y, w / 2, h / 2, -2};
                f.r[2] = (Rect){x + w / 2, y + h / 2, w / 2, h / 2, -2}; add_feat(f); }
            if (w % 3 == 0 && h % 3 == 0 && w >= 3) { /* centre-surround */
                f.nr = 2; f.r[0] = (Rect){x, y, w, h, 1}; f.r[1] = (Rect){x + w / 3, y + h / 3, w / 3, h / 3, -9}; add_feat(f); }
        }
    fprintf(stderr, "full Haar pool: %d features\n", nfeat);
    for (int i = nfeat - 1; i > 0; i--) { int j = rnd() % (i + 1); Feature t = feats[i]; feats[i] = feats[j]; feats[j] = t; }
    if (nfeat > FMAX) nfeat = FMAX;
}

/* ---------------- data ---------------- */
typedef struct { unsigned char p[NPIX]; } Img;
static void transform(const unsigned char *src, unsigned char *dst, int d, int dx, int dy) {
    for (int y = 0; y < W; y++) for (int x = 0; x < W; x++) {
        int sx = x + dx, sy = y + dy;
        if (sx < 0) sx = 0; if (sx >= W) sx = W - 1; if (sy < 0) sy = 0; if (sy >= W) sy = W - 1;
        int u = sx, v = sy, t;
        if (d & 1) u = W - 1 - u;
        if (d & 2) v = W - 1 - v;
        if (d & 4) { t = u; u = v; v = t; }
        dst[y * W + x] = src[v * W + u];
    }
}

/* ---------------- stage training ---------------- */
static float *edges; /* nfeat x (NBINS-1) */
typedef struct { Integ *I; int n; unsigned char *bins; } BinCtx;
static void bin_job(int lo, int hi, int tid, void *c) {
    (void)tid; BinCtx *B = c;
    for (int f = lo; f < hi; f++) {
        const float *e = edges + (size_t)f * (NBINS - 1);
        unsigned char *row = B->bins + (size_t)f * B->n;
        for (int i = 0; i < B->n; i++) {
            float v = feat_val(&feats[f], &B->I[i]);
            int a = 0, b = NBINS - 1; /* count edges <= v */
            while (a < b) { int m = (a + b) >> 1; if (e[m] <= v) a = m + 1; else b = m; }
            row[i] = (unsigned char)a;
        }
    }
}

typedef struct { double z; int f, s; float lo, hi; } Best;
typedef struct { unsigned char *bins; int n; const float *w; const signed char *y; Best best[64]; } SearchCtx;
static void search_job(int lo, int hi, int tid, void *c) {
    SearchCtx *S = c;
    Best b = {DBL_MAX, -1, 0, 0, 0};
    double eps = 1.0 / S->n;
    for (int f = lo; f < hi; f++) {
        double h[NBINS][2]; memset(h, 0, sizeof h);
        const unsigned char *row = S->bins + (size_t)f * S->n;
        for (int i = 0; i < S->n; i++) h[row[i]][S->y[i] > 0] += S->w[i];
        double Tn = 0, Tp = 0;
        for (int k = 0; k < NBINS; k++) { Tn += h[k][0]; Tp += h[k][1]; }
        double Ln = 0, Lp = 0;
        for (int s = 1; s < NBINS; s++) {
            Ln += h[s - 1][0]; Lp += h[s - 1][1];
            double Rn = Tn - Ln, Rp = Tp - Lp;
            double z = sqrt(Lp * Ln) + sqrt(Rp * Rn);
            if (z < b.z) {
                b.z = z; b.f = f; b.s = s;
                b.lo = (float)(0.5 * log((Lp + eps) / (Ln + eps)));
                b.hi = (float)(0.5 * log((Rp + eps) / (Rn + eps)));
            }
        }
    }
    S->best[tid] = b;
}

static int cmpf(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

/* Train one stage. pos/neg integrals; calib positives for threshold (may be NULL for final). */
static Stage train_stage(Integ *pos, int np, Integ *neg, int nn, Integ *cal, int nc,
                         int maxw, float fp_target, int final) {
    int n = np + nn;
    Integ *I = malloc(sizeof(Integ) * n);
    memcpy(I, pos, sizeof(Integ) * np); memcpy(I + np, neg, sizeof(Integ) * nn);
    signed char *y = malloc(n);
    float *w = malloc(sizeof(float) * n), *m = calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) { y[i] = i < np ? 1 : -1; w[i] = i < np ? 0.5f / np : 0.5f / nn; }
    BinCtx B = {I, n, malloc((size_t)nfeat * n)};
    pfor(nfeat, bin_job, &B);
    float *mc = calloc(nc ? nc : 1, sizeof(float)), *tmp = malloc(sizeof(float) * (nc ? nc : 1));
    Stage S = {0, 0, 1, malloc(sizeof(Weak) * maxw)};
    for (int t = 0; t < maxw; t++) {
        SearchCtx C = {B.bins, n, w, y};
        pfor(nfeat, search_job, &C);
        Best b = C.best[0];
        for (int k = 1; k < nthreads; k++) if (C.best[k].z < b.z) b = C.best[k];
        Weak *wk = &S.w[S.n++];
        wk->f = feats[b.f]; wk->thr = edges[(size_t)b.f * (NBINS - 1) + b.s - 1]; wk->lo = b.lo; wk->hi = b.hi;
        const unsigned char *row = B.bins + (size_t)b.f * n;
        double sum = 0;
        for (int i = 0; i < n; i++) {
            float h = row[i] < b.s ? b.lo : b.hi;
            m[i] += h; w[i] *= expf(-y[i] * h); sum += w[i];
        }
        for (int i = 0; i < n; i++) w[i] /= sum;
        for (int i = 0; i < nc; i++) mc[i] += feat_val(&wk->f, &cal[i]) < wk->thr ? wk->lo : wk->hi;
        if (final) continue;
        memcpy(tmp, mc, sizeof(float) * nc); qsort(tmp, nc, sizeof(float), cmpf);
        int k = (int)floor((1.0 - STAGE_DET) * nc);
        double cm = 0, cv = 0;
        for (int i = 0; i < nc; i++) cm += mc[i];
        cm /= nc;
        for (int i = 0; i < nc; i++) cv += (mc[i] - cm) * (mc[i] - cm);
        S.thr = tmp[k] - 1e-4f - (float)(SAFETY * sqrt(cv / nc));
        int fp = 0, tp = 0;
        for (int i = np; i < n; i++) fp += m[i] >= S.thr;
        for (int i = 0; i < np; i++) tp += m[i] >= S.thr;
        fprintf(stderr, "  weak %d: feat nr=%d thr=%.3f calib-det=%.4f train-det=%.4f fp=%.4f\n",
                S.n, wk->f.nr, S.thr, 1.0 - (double)k / nc, (double)tp / np, (double)fp / nn);
        if ((double)fp / nn <= fp_target) break;
    }
    /* score scale: spread of training margins */
    double mu = 0, v = 0;
    for (int i = 0; i < n; i++) mu += m[i];
    mu /= n;
    for (int i = 0; i < n; i++) v += (m[i] - mu) * (m[i] - mu);
    S.scale = (float)sqrt(v / n) * 0.5f;
    if (S.scale < 1e-3f) S.scale = 1;
    if (final) {
        S.thr = 0;
        int fp = 0, tp = 0;
        for (int i = 0; i < n; i++) { if (y[i] > 0) tp += m[i] >= 0; else fp += m[i] >= 0; }
        fprintf(stderr, "  final stage: %d weak, train det=%.4f fp=%.4f\n", S.n, (double)tp / np, (double)fp / nn);
    }
    free(B.bins); free(I); free(y); free(w); free(m); free(mc); free(tmp);
    return S;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: train <images_dir> <labels.txt> <model_out>\n"); return 1; }
    long nc = sysconf(_SC_NPROCESSORS_ONLN);
    nthreads = nc < 1 ? 1 : nc > 64 ? 64 : (int)nc;
    FILE *lf = fopen(argv[2], "r");
    if (!lf) { perror(argv[2]); return 1; }
    int cap = 1024, nf = 0, nnf = 0;
    Img *faces = malloc(cap * sizeof(Img)), *nonf = malloc(cap * sizeof(Img));
    int capf = cap, capn = cap;
    char id[512], path[4096]; int lab;
    while (fscanf(lf, "%511s %d", id, &lab) == 2) {
        Img im;
        snprintf(path, sizeof path, "%s/%s.pgm", argv[1], id);
        if (read_pgm(path, im.p)) { fprintf(stderr, "skip %s\n", path); continue; }
        if (lab == 1) { if (nf == capf) faces = realloc(faces, (capf *= 2) * sizeof(Img)); faces[nf++] = im; }
        else { if (nnf == capn) nonf = realloc(nonf, (capn *= 2) * sizeof(Img)); nonf[nnf++] = im; }
    }
    fclose(lf);
    fprintf(stderr, "loaded %d faces, %d non-faces, %d threads\n", nf, nnf, nthreads);

    /* faces: split train/calib, augment with horizontal mirror */
    for (int i = nf - 1; i > 0; i--) { int j = rnd() % (i + 1); Img t = faces[i]; faces[i] = faces[j]; faces[j] = t; }
    int ncal = (int)(nf * CALIB_FRAC), ntr = nf - ncal;
    Integ *pos = malloc(sizeof(Integ) * 2 * nf); /* [0,2*ntr) train, then calib */
    for (int i = 0; i < nf; i++) {
        unsigned char mir[NPIX];
        transform(faces[i].p, mir, 1, 0, 0);
        int base = i < ntr ? 2 * i : 2 * ntr + 2 * (i - ntr);
        make_integ(faces[i].p, &pos[base]); make_integ(mir, &pos[base + 1]);
    }
    int np = 2 * ntr, npc = 2 * ncal, npall = 2 * nf;

    /* negative pool: 8 dihedral transforms x 5 one-pixel shifts of every training non-face */
    static const int sh[5][2] = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    int npool = nnf * 40;
    Img *pool = malloc(sizeof(Img) * npool);
    for (int i = 0, k = 0; i < nnf; i++)
        for (int d = 0; d < 8; d++) for (int s = 0; s < 5; s++) transform(nonf[i].p, pool[k++].p, d, sh[s][0], sh[s][1]);
    char *alive = malloc(npool); memset(alive, 1, npool);
    fprintf(stderr, "negative pool: %d patches\n", npool);

    gen_features();
    /* quantile bin edges from a sample of faces and pool negatives */
    {
        int ns = 4000; Integ *smp = malloc(sizeof(Integ) * ns);
        for (int i = 0; i < ns; i++) {
            if (i < ns / 2) smp[i] = pos[rnd() % npall];
            else make_integ(pool[rnd() % npool].p, &smp[i]);
        }
        edges = malloc(sizeof(float) * nfeat * (NBINS - 1));
        float *v = malloc(sizeof(float) * ns);
        for (int f = 0; f < nfeat; f++) {
            for (int i = 0; i < ns; i++) v[i] = feat_val(&feats[f], &smp[i]);
            qsort(v, ns, sizeof(float), cmpf);
            for (int k = 1; k < NBINS; k++) {
                int a = (int)((long)k * ns / NBINS);
                edges[(size_t)f * (NBINS - 1) + k - 1] = 0.5f * (v[a - 1] + v[a]);
            }
        }
        free(v); free(smp);
    }

    Model M = {0, calloc(MAX_EARLY + 1, sizeof(Stage))};
    Integ *neg = malloc(sizeof(Integ) * NEG_CAP);
    int *idx = malloc(sizeof(int) * npool);
    for (int st = 0; st <= MAX_EARLY; st++) {
        int na = 0;
        for (int i = 0; i < npool; i++) if (alive[i]) idx[na++] = i;
        int final = (st == MAX_EARLY) || (na < MIN_NEG);
        for (int i = na - 1; i > 0; i--) { int j = rnd() % (i + 1); int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
        int nn = na < NEG_CAP ? na : NEG_CAP;
        for (int i = 0; i < nn; i++) make_integ(pool[idx[i]].p, &neg[i]);
        fprintf(stderr, "stage %d: %d surviving pool negatives, training on %d%s\n", st + 1, na, nn, final ? " (final)" : "");
        Stage S;
        if (final) S = train_stage(pos, npall, neg, nn, NULL, 0, FINAL_WEAK, 0, 1); /* all faces */
        else S = train_stage(pos, np, neg, nn, pos + np, npc, early_max[st], early_fp[st], 0);
        M.s[M.ns++] = S;
        if (final) break;
        int surv = 0;
        for (int i = 0; i < npool; i++) if (alive[i]) {
            Integ I; make_integ(pool[i].p, &I);
            if (stage_margin(&S, &I) < S.thr) alive[i] = 0; else surv++;
        }
        fprintf(stderr, "  stage %d keeps %d/%d pool negatives\n", st + 1, surv, na);
    }
    save_model(argv[3], &M);
    fprintf(stderr, "saved %d stages\n", M.ns);
    return 0;
}
