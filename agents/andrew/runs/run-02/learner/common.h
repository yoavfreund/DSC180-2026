/* Shared definitions: images, integral images, Haar features, model I/O. */
#ifndef COMMON_H
#define COMMON_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define W 19
#define NPIX (W * W)
#define IW (W + 1)

typedef struct { signed char x, y, w, h; float wt; } Rect;
typedef struct { int nr; Rect r[3]; } Feature;

/* integral image with per-window std normalisation */
typedef struct { int ii[IW * IW]; float inv_std; } Integ;

static int read_pgm(const char *path, unsigned char *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[3] = {0};
    int w, h, mx;
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P5")) { fclose(f); return -1; }
    /* skip comments */
    int c;
    int vals[3], nv = 0;
    while (nv < 3) {
        c = fgetc(f);
        if (c == EOF) { fclose(f); return -1; }
        if (c == '#') { while ((c = fgetc(f)) != '\n' && c != EOF); continue; }
        if (c >= '0' && c <= '9') { ungetc(c, f); if (fscanf(f, "%d", &vals[nv]) != 1) { fclose(f); return -1; } nv++; }
    }
    w = vals[0]; h = vals[1]; mx = vals[2];
    fgetc(f); /* single whitespace */
    if (w != W || h != W || mx > 255) { fclose(f); return -1; }
    size_t n = fread(out, 1, NPIX, f);
    fclose(f);
    return n == NPIX ? 0 : -1;
}

static void make_integ(const unsigned char *img, Integ *I) {
    double s = 0, s2 = 0;
    memset(I->ii, 0, sizeof(I->ii));
    for (int y = 0; y < W; y++) {
        int row = 0;
        for (int x = 0; x < W; x++) {
            int v = img[y * W + x];
            row += v; s += v; s2 += (double)v * v;
            I->ii[(y + 1) * IW + x + 1] = I->ii[y * IW + x + 1] + row;
        }
    }
    double m = s / NPIX, var = s2 / NPIX - m * m;
    double sd = var > 1.0 ? sqrt(var) : 1.0; /* flat patches: avoid /0 */
    I->inv_std = (float)(1.0 / sd);
}

static inline int rect_sum(const int *ii, int x, int y, int w, int h) {
    return ii[(y + h) * IW + x + w] - ii[y * IW + x + w] - ii[(y + h) * IW + x] + ii[y * IW + x];
}

/* zero-sum Haar features => mean cancels; divide by std for contrast invariance */
static inline float feat_val(const Feature *f, const Integ *I) {
    float v = 0;
    for (int k = 0; k < f->nr; k++)
        v += f->r[k].wt * rect_sum(I->ii, f->r[k].x, f->r[k].y, f->r[k].w, f->r[k].h);
    return v * I->inv_std;
}

typedef struct { Feature f; float thr, lo, hi; } Weak; /* value<thr -> lo else hi */
typedef struct { int n; float thr, scale; Weak *w; } Stage;
typedef struct { int ns; Stage *s; } Model;

static void save_model(const char *path, const Model *M) {
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(1); }
    fprintf(f, "CASCADE %d\n", M->ns);
    for (int s = 0; s < M->ns; s++) {
        Stage *S = &M->s[s];
        fprintf(f, "STAGE %d %.9g %.9g\n", S->n, S->thr, S->scale);
        for (int i = 0; i < S->n; i++) {
            Weak *w = &S->w[i];
            fprintf(f, "%d", w->f.nr);
            for (int k = 0; k < w->f.nr; k++)
                fprintf(f, " %d %d %d %d %g", w->f.r[k].x, w->f.r[k].y, w->f.r[k].w, w->f.r[k].h, w->f.r[k].wt);
            fprintf(f, " %.9g %.9g %.9g\n", w->thr, w->lo, w->hi);
        }
    }
    fclose(f);
}

static int load_model(const char *path, Model *M) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (fscanf(f, " CASCADE %d", &M->ns) != 1) { fclose(f); return -1; }
    M->s = calloc(M->ns, sizeof(Stage));
    for (int s = 0; s < M->ns; s++) {
        Stage *S = &M->s[s];
        if (fscanf(f, " STAGE %d %f %f", &S->n, &S->thr, &S->scale) != 3) { fclose(f); return -1; }
        S->w = calloc(S->n, sizeof(Weak));
        for (int i = 0; i < S->n; i++) {
            Weak *w = &S->w[i];
            if (fscanf(f, "%d", &w->f.nr) != 1) { fclose(f); return -1; }
            for (int k = 0; k < w->f.nr; k++) {
                int x, y, ww, h; float wt;
                if (fscanf(f, "%d %d %d %d %f", &x, &y, &ww, &h, &wt) != 5) { fclose(f); return -1; }
                w->f.r[k].x = x; w->f.r[k].y = y; w->f.r[k].w = ww; w->f.r[k].h = h; w->f.r[k].wt = wt;
            }
            if (fscanf(f, "%f %f %f", &w->thr, &w->lo, &w->hi) != 3) { fclose(f); return -1; }
        }
    }
    fclose(f);
    return 0;
}

static inline float stage_margin(const Stage *S, const Integ *I) {
    float m = 0;
    for (int i = 0; i < S->n; i++) {
        const Weak *w = &S->w[i];
        m += feat_val(&w->f, I) < w->thr ? w->lo : w->hi;
    }
    return m;
}

/* Returns score; *stop = 1-based stage where the sample stopped. */
static double cascade_score(const Model *M, const Integ *I, int *stop) {
    for (int s = 0; s < M->ns; s++) {
        const Stage *S = &M->s[s];
        float m = stage_margin(S, I);
        if (s == M->ns - 1 || m < S->thr) {
            if (stop) *stop = s + 1;
            double z = (m - S->thr) / S->scale;
            return s + 1.0 / (1.0 + exp(-z));
        }
    }
    return 0;
}
#endif
