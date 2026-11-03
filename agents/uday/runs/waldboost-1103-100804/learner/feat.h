#ifndef FEAT_H
#define FEAT_H
#include "common.h"
#define NB 16
typedef struct { int n; int r[3][4]; double wt[3]; } Feat;  /* up to 3 rects: x,y,w,h */
static double feat_eval(const Feat *f, const double *ii) {
    double s = 0;
    for (int k = 0; k < f->n; k++) s += f->wt[k] * rect_sum(ii, f->r[k][0], f->r[k][1], f->r[k][2], f->r[k][3]);
    return s;
}
static void prep(const double *raw, double *ii) { double x[NPIX]; normalize(raw, x); integral(x, ii); }
static int bin_of(const double *edges, double v) { int b = 0; while (b < NB - 1 && v >= edges[b]) b++; return b; }
#endif
