/* Shared helpers for the test learners: read a 19x19 PGM, normalize it. */
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPIX 361

/* Reads pixels into x (zero-mean, unit-variance; constant images become all 0). */
static int read_pgm(const char *path, double *x) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int w, h, maxv;
    if (fscanf(f, "P5 %d %d %d", &w, &h, &maxv) != 3 || w * h != NPIX) { fclose(f); return -1; }
    fgetc(f);
    unsigned char buf[NPIX];
    size_t got = fread(buf, 1, NPIX, f);
    fclose(f);
    if (got != NPIX) return -1;
    double mean = 0, var = 0;
    for (int i = 0; i < NPIX; i++) mean += buf[i];
    mean /= NPIX;
    for (int i = 0; i < NPIX; i++) var += (buf[i] - mean) * (buf[i] - mean);
    double sd = sqrt(var / NPIX);
    for (int i = 0; i < NPIX; i++) x[i] = sd > 1e-9 ? (buf[i] - mean) / sd : 0.0;
    return 0;
}
