#ifndef COMMON_H
#define COMMON_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define S 19
#define NPIX (S * S)
#define II (S + 1)

/* feature: type, x, y, cell width, cell height */
typedef struct { int type, x, y, w, h; } Feat;

/* read binary PGM (P5) of size 19x19; returns 0 on success */
static int read_pgm(const char *path, unsigned char *px) {
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
        if (c >= '0' && c <= '9') {
            int v = c - '0';
            while ((c = fgetc(f)) >= '0' && c <= '9') v = v * 10 + (c - '0');
            vals[nv++] = v;
        }
    }
    w = vals[0]; h = vals[1]; mx = vals[2];
    (void)mx;
    if (w != S || h != S) { fclose(f); return -1; }
    size_t r = fread(px, 1, NPIX, f);
    fclose(f);
    if (r != NPIX) { memset(px + r, 0, NPIX - r); }
    return 0;
}

/* variance-normalized integral image (double) */
static void make_integral(const unsigned char *px, int flip, double *ii) {
    double m = 0, s2 = 0;
    for (int i = 0; i < NPIX; i++) { m += px[i]; s2 += (double)px[i] * px[i]; }
    m /= NPIX;
    double var = s2 / NPIX - m * m;
    double sd = var > 1e-6 ? sqrt(var) : 1.0;
    for (int x = 0; x < II; x++) ii[x] = 0;
    for (int y = 0; y < S; y++) {
        double row = 0;
        ii[(y + 1) * II] = 0;
        for (int x = 0; x < S; x++) {
            int sx = flip ? (S - 1 - x) : x;
            row += (px[y * S + sx] - m) / sd;
            ii[(y + 1) * II + x + 1] = ii[y * II + x + 1] + row;
        }
    }
}

static inline double rect(const double *ii, int x, int y, int w, int h) {
    return ii[(y + h) * II + x + w] - ii[y * II + x + w] - ii[(y + h) * II + x] + ii[y * II + x];
}

static inline double feat_val(const Feat *f, const double *ii) {
    int x = f->x, y = f->y, w = f->w, h = f->h;
    switch (f->type) {
    case 0: return rect(ii, x, y, w, h) - rect(ii, x + w, y, w, h);
    case 1: return rect(ii, x, y, w, h) - rect(ii, x, y + h, w, h);
    case 2: return rect(ii, x, y, w, h) + rect(ii, x + 2 * w, y, w, h) - 2 * rect(ii, x + w, y, w, h);
    case 3: return rect(ii, x, y, w, h) + rect(ii, x, y + 2 * h, w, h) - 2 * rect(ii, x, y + h, w, h);
    default: return rect(ii, x, y, w, h) + rect(ii, x + w, y + h, w, h)
                  - rect(ii, x + w, y, w, h) - rect(ii, x, y + h, w, h);
    }
}

static int enum_features(Feat *out) {
    static const int mw[5] = {2, 1, 3, 1, 2}, mh[5] = {1, 2, 1, 3, 2};
    int n = 0;
    for (int t = 0; t < 5; t++)
        for (int w = 1; w * mw[t] <= S; w++)
            for (int h = 1; h * mh[t] <= S; h++)
                for (int y = 0; y + h * mh[t] <= S; y++)
                    for (int x = 0; x + w * mw[t] <= S; x++) {
                        if (out) { out[n].type = t; out[n].x = x; out[n].y = y; out[n].w = w; out[n].h = h; }
                        n++;
                    }
    return n;
}
#endif
