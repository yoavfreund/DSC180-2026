/* WaldBoost cascade: Real AdaBoost binned-stump weak learners on Haar features,
 * one rejection threshold per weak classifier (each weak = a stage). */
#include "feat.h"
#include <time.h>

#define NF 5000
#define T 100
#define NNEG 7000
#define REJ 0.003
static unsigned long long rs = 88172645463325252ULL;
static unsigned rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 11); }
static double urnd(void) { return rnd() / 4294967296.0; }
static int cmpd(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }

static Feat feats[NF];
static double edges[NF][NB - 1];
static void make_feat(Feat *f) {
    int t = rnd() % 5, cx = (t == 0 || t == 4) ? 2 : (t == 2 ? 3 : 1), cy = (t == 1 || t == 4) ? 2 : (t == 3 ? 3 : 1);
    int w = 1 + rnd() % (W / cx), h = 1 + rnd() % (H / cy);
    int x = rnd() % (W - w * cx + 1), y = rnd() % (H - h * cy + 1);
    memset(f, 0, sizeof *f);
    if (t == 0) { f->n = 2; int a[2][4] = {{x,y,w,h},{x+w,y,w,h}}; memcpy(f->r, a, sizeof a); f->wt[0] = 1; f->wt[1] = -1; }
    else if (t == 1) { f->n = 2; int a[2][4] = {{x,y,w,h},{x,y+h,w,h}}; memcpy(f->r, a, sizeof a); f->wt[0] = 1; f->wt[1] = -1; }
    else if (t == 2) { f->n = 3; int a[3][4] = {{x,y,w,h},{x+w,y,w,h},{x+2*w,y,w,h}}; memcpy(f->r, a, sizeof a); f->wt[0] = 1; f->wt[1] = -2; f->wt[2] = 1; }
    else if (t == 3) { f->n = 3; int a[3][4] = {{x,y,w,h},{x,y+h,w,h},{x,y+2*h,w,h}}; memcpy(f->r, a, sizeof a); f->wt[0] = 1; f->wt[1] = -2; f->wt[2] = 1; }
    else { /* checker: (TL+BR)-(TR+BL) as TL+BR rect pair of double-weight trick -> use 3 rects: whole 2x2 -> approximate */
        f->n = 3; int a[3][4] = {{x,y,w,h},{x+w,y+h,w,h},{x,y,2*w,2*h}}; memcpy(f->r, a, sizeof a);
        f->wt[0] = 2; f->wt[1] = 2; f->wt[2] = -1; }
}

static unsigned char *bins;  /* [NF][NS] */
static int NS;
static void fill_slot(int s, const unsigned char *u8) {
    double raw[NPIX], ii[(W + 1) * (H + 1)];
    for (int i = 0; i < NPIX; i++) raw[i] = u8[i];
    prep(raw, ii);
    for (int f = 0; f < NF; f++) bins[(size_t)f * NS + s] = bin_of(edges[f], feat_eval(&feats[f], ii));
}

typedef struct { int f; double h[NB]; double thr; } Weak;
static Weak weak[T];
static double eval_weak_u8(const Weak *w, const double *ii) { return w->h[bin_of(edges[w->f], feat_eval(&feats[w->f], ii))]; }

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    int n = 0;
    char **ids = list_ids(argv[1], &n);
    double *px = malloc((size_t)n * NPIX * sizeof *px);
    int *lab = calloc(n, sizeof *lab);
    if (!ids || load_images(argv[1], ids, n, px) || read_labels(argv[2], ids, n, lab) != n) return 1;
    unsigned char *img = malloc((size_t)n * NPIX);
    for (int i = 0; i < n * NPIX; i++) img[i] = (unsigned char)px[i];

    /* split: 20% of faces held out (threshold calibration); 100% of non-faces train */
    int *hold = calloc(n, sizeof *hold);
    for (int i = 0; i < n; i++) if (lab[i] && urnd() < 0.2) hold[i] = 1;
    int nh = 0, np = 0, nn = 0;
    for (int i = 0; i < n; i++) { if (lab[i] && hold[i]) nh++; else if (lab[i]) np++; else nn++; }

    /* slots: train faces + flips, holdout faces, negatives */
    int NP = 2 * np, NH = nh;
    NS = NP + NH + NNEG;
    unsigned char *simg = malloc((size_t)NS * NPIX);
    int *y = calloc(NS, sizeof *y), *alive = calloc(NS, sizeof *alive);
    double *F = calloc(NS, sizeof *F);
    int k = 0;
    for (int i = 0; i < n; i++) if (lab[i] && !hold[i]) {
        memcpy(simg + (size_t)k * NPIX, img + (size_t)i * NPIX, NPIX); k++;
        for (int yy = 0; yy < H; yy++) for (int xx = 0; xx < W; xx++)
            simg[(size_t)k * NPIX + yy * W + xx] = img[(size_t)i * NPIX + yy * W + (W - 1 - xx)];
        k++;
    }
    for (int i = 0; i < n; i++) if (lab[i] && hold[i]) { memcpy(simg + (size_t)k * NPIX, img + (size_t)i * NPIX, NPIX); k++; }
    for (int i = 0; i < NP + NH; i++) { y[i] = 1; alive[i] = 1; }

    /* negative pool: originals, flips, then shifted variants (shuffled) */
    int nvar = 50;
    size_t PS = (size_t)nn * nvar;
    unsigned char *pool = malloc(PS * NPIX);
    size_t pk = 0;
    int *nid = malloc(nn * sizeof *nid), m = 0;
    for (int i = 0; i < n; i++) if (!lab[i]) nid[m++] = i;
    for (int v = 0; v < nvar; v++) {
        int fl = v % 2, dd = v / 2, dx = dd % 5 - 2, dy = dd / 5 - 2;
        for (int j = 0; j < nn; j++) {
            const unsigned char *s = img + (size_t)nid[j] * NPIX;
            unsigned char *d = pool + pk * NPIX; pk++;
            for (int yy = 0; yy < H; yy++) for (int xx = 0; xx < W; xx++) {
                int sx = xx + dx, sy = yy + dy;
                sx = sx < 0 ? 0 : sx >= W ? W - 1 : sx; sy = sy < 0 ? 0 : sy >= H ? H - 1 : sy;
                if (fl) sx = W - 1 - sx;
                d[yy * W + xx] = s[sy * W + sx];
            }
        }
    }
    /* variants 0 (orig) and 1 (flip) stay first; shuffle the rest */
    size_t first = (size_t)2 * nn;
    for (size_t i = PS - 1; i > first; i--) {
        size_t j = first + rnd() % (i - first + 1);
        unsigned char tmp[NPIX]; memcpy(tmp, pool + i * NPIX, NPIX); memcpy(pool + i * NPIX, pool + j * NPIX, NPIX); memcpy(pool + j * NPIX, tmp, NPIX);
    }
    size_t pptr = 0;

    /* features and bin edges from a random subsample */
    for (int f = 0; f < NF; f++) make_feat(&feats[f]);
    { int ns = 2500; double *v = malloc(ns * sizeof *v * NF); 
      for (int s = 0; s < ns; s++) {
        double raw[NPIX], ii[(W + 1) * (H + 1)];
        const unsigned char *u = (s % 2) ? simg + (size_t)(rnd() % (NP + NH)) * NPIX : pool + (size_t)(rnd() % (2 * nn)) * NPIX;
        for (int i = 0; i < NPIX; i++) raw[i] = u[i];
        prep(raw, ii);
        for (int f = 0; f < NF; f++) v[(size_t)f * ns + s] = feat_eval(&feats[f], ii);
      }
      for (int f = 0; f < NF; f++) {
        double *a = v + (size_t)f * ns; qsort(a, ns, sizeof *a, cmpd);
        for (int b = 0; b < NB - 1; b++) edges[f][b] = a[(int)((b + 1) * (double)ns / NB)];
      }
      free(v); }

    bins = malloc((size_t)NF * NS);
    for (int s = 0; s < NP + NH; s++) fill_slot(s, simg + (size_t)s * NPIX);
    double prior[2];
    prior[1] = 1.0 / (NP + NH); prior[0] = 1.0 / NNEG;
    int nslot_neg_filled = 0;
    #define REFILL() do { \
        for (int s = NP + NH; s < NS; s++) if (!alive[s] && pptr < PS) { \
            while (pptr < PS) { \
                const unsigned char *u = pool + pptr * NPIX; pptr++; \
                double raw[NPIX], ii[(W + 1) * (H + 1)], sum = 0; int ok = 1; \
                for (int i = 0; i < NPIX; i++) raw[i] = u[i]; \
                prep(raw, ii); \
                for (int t = 0; t < nt; t++) { sum += eval_weak_u8(&weak[t], ii); if (sum < weak[t].thr) { ok = 0; break; } } \
                if (ok) { memcpy(simg + (size_t)s * NPIX, u, NPIX); fill_slot(s, u); F[s] = sum; alive[s] = 1; y[s] = -1; break; } \
            } } } while (0)
    int nt = 0;
    REFILL();
    (void)nslot_neg_filled;

    double *w = malloc(NS * sizeof *w);
    int *al = malloc(NS * sizeof *al);
    for (nt = 0; nt < T;) {
        int na = 0, nneg = 0; double tot = 0;
        for (int s = 0; s < NS; s++) if (alive[s] && !(s >= NP && s < NP + NH)) {
            al[na++] = s; if (y[s] < 0) nneg++;
            double e = -y[s] * F[s]; if (e > 30) e = 30;
            w[s] = prior[y[s] > 0] * exp(e); tot += w[s];
        }
        if (nneg < 40) { fprintf(stderr, "negatives exhausted at t=%d\n", nt); break; }
        for (int i = 0; i < na; i++) w[al[i]] /= tot;
        double bestZ = 1e30; int bf = 0; double bh[NB];
        for (int f = 0; f < NF; f++) {
            double wp[NB] = {0}, wn[NB] = {0};
            const unsigned char *b = bins + (size_t)f * NS;
            for (int i = 0; i < na; i++) { int s = al[i]; if (y[s] > 0) wp[b[s]] += w[s]; else wn[b[s]] += w[s]; }
            double Z = 0;
            for (int j = 0; j < NB; j++) Z += sqrt(wp[j] * wn[j]);
            if (Z < bestZ) { bestZ = Z; bf = f; for (int j = 0; j < NB; j++) { double h = 0.5 * log((wp[j] + 1e-5) / (wn[j] + 1e-5)); bh[j] = h > 3 ? 3 : h < -3 ? -3 : h; } }
        }
        Weak *wk = &weak[nt]; wk->f = bf; memcpy(wk->h, bh, sizeof bh);
        for (int s = 0; s < NS; s++) if (alive[s]) F[s] += bh[bins[(size_t)bf * NS + s]];
        /* threshold from held-out faces */
        double hv[NH > 0 ? NH : 1]; int nhv = 0;
        for (int s = NP; s < NP + NH; s++) if (alive[s]) hv[nhv++] = F[s];
        qsort(hv, nhv, sizeof *hv, cmpd);
        wk->thr = hv[(int)(REJ * nhv)];
        int dn = 0, dp = 0;
        for (int s = 0; s < NS; s++) if (alive[s] && F[s] < wk->thr) { alive[s] = 0; if (y[s] < 0) dn++; else dp++; }
        nt++;
        REFILL();
        if (nt % 10 == 0) fprintf(stderr, "t=%d Z=%.3f negalive=%d rejneg=%d rejpos=%d\n", nt, bestZ, nneg, dn, dp);
    }

    FILE *fo = fopen(argv[3], "w");
    if (!fo) return 1;
    fprintf(fo, "waldboost %d\n", nt);
    for (int t = 0; t < nt; t++) {
        const Feat *ft = &feats[weak[t].f];
        fprintf(fo, "%d", ft->n);
        for (int r = 0; r < ft->n; r++) fprintf(fo, " %d %d %d %d %g", ft->r[r][0], ft->r[r][1], ft->r[r][2], ft->r[r][3], ft->wt[r]);
        fprintf(fo, "\n");
        for (int b = 0; b < NB - 1; b++) fprintf(fo, "%.17g ", edges[weak[t].f][b]);
        fprintf(fo, "\n");
        for (int b = 0; b < NB; b++) fprintf(fo, "%.17g ", weak[t].h[b]);
        fprintf(fo, "\n%.17g\n", weak[t].thr);
    }
    fclose(fo);
    return 0;
}
