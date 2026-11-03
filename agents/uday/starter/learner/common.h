/* common.h - helpers shared by train.c and predict.c.
 *
 * Algorithm-agnostic plumbing only: PGM reading, file listing, labels, patch
 * normalization, the cascade score convention, and an optional integral image.
 * Edit or delete anything here; none of it is required. */
#ifndef COMMON_H
#define COMMON_H
#define _POSIX_C_SOURCE 200809L
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 19
#define H 19
#define NPIX (W * H)

/* Read a 19x19 binary PGM (header "P5\n19 19\n255\n") into px[NPIX] as 0..255. */
static int read_pgm(const char *path, double *px) {
    unsigned char buf[13 + NPIX];
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    if (n != sizeof buf || memcmp(buf, "P5\n19 19\n255\n", 13) != 0) return -1;
    for (int i = 0; i < NPIX; i++) px[i] = buf[13 + i];
    return 0;
}

static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

/* Sorted list of ids (filename without ".pgm") for every .pgm in dir. */
static char **list_ids(const char *dir, int *count) {
    DIR *d = opendir(dir);
    if (!d) return NULL;
    int cap = 1024, n = 0;
    char **ids = malloc(cap * sizeof *ids);
    struct dirent *e;
    while ((e = readdir(d))) {
        size_t len = strlen(e->d_name);
        if (len < 5 || strcmp(e->d_name + len - 4, ".pgm")) continue;
        if (n == cap) ids = realloc(ids, (cap *= 2) * sizeof *ids);
        ids[n] = strndup(e->d_name, len - 4);
        n++;
    }
    closedir(d);
    qsort(ids, n, sizeof *ids, cmp_str);
    *count = n;
    return ids;
}

/* Load every image in ids[] into out[n*NPIX]. Returns 0 on success. */
static int load_images(const char *dir, char **ids, int n, double *out) {
    char path[4096];
    for (int i = 0; i < n; i++) {
        snprintf(path, sizeof path, "%s/%s.pgm", dir, ids[i]);
        if (read_pgm(path, out + (size_t)i * NPIX)) return -1;
    }
    return 0;
}

/* Fill labels[i] (1 = face, 0 = non-face) for each id in ids[] (must be sorted).
 * Returns the number of ids matched; it should equal n. */
static int read_labels(const char *path, char **ids, int n, int *labels) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char id[64];
    int lab, matched = 0;
    while (fscanf(f, "%63s %d", id, &lab) == 2) {
        char *key = id;
        char **hit = bsearch(&key, ids, n, sizeof *ids, cmp_str);
        if (hit) { labels[hit - ids] = lab; matched++; }
    }
    fclose(f);
    return matched;
}

/* Zero-mean, unit-variance copy of a patch. Flat patches (std = 0) come out
 * as all zeros instead of dividing by zero. */
static void normalize(const double *in, double *out) {
    double mean = 0, var = 0;
    for (int i = 0; i < NPIX; i++) mean += in[i];
    mean /= NPIX;
    for (int i = 0; i < NPIX; i++) var += (in[i] - mean) * (in[i] - mean);
    double sd = sqrt(var / NPIX);
    double inv = sd > 1e-6 ? 1.0 / sd : 0.0;
    for (int i = 0; i < NPIX; i++) out[i] = (in[i] - mean) * inv;
}

/* Cascade score convention. stage is 0-based: the stage where the patch was
 * rejected, or the last stage if it was never rejected. The result lies in
 * [stage, stage + 1), so the harness can read the exit stage as floor(score),
 * and a patch rejected earlier always scores below one that got further.
 * margin: bigger means more face-like at that stage. Squash it to a modest
 * range first (e.g. divide by a typical scale) so it does not saturate. */
static double cascade_score(int stage, double margin) {
    double s = 1.0 / (1.0 + exp(-margin));
    if (s < 1e-6) s = 1e-6;
    if (s > 1.0 - 1e-6) s = 1.0 - 1e-6;
    return stage + s;
}

/* Optional: integral image, ii[(H+1)*(W+1)]; rect_sum() = sum over rows y..y+h-1, cols x..x+w-1. */
static void integral(const double *px, double *ii) {
    for (int x = 0; x <= W; x++) ii[x] = 0;
    for (int y = 1; y <= H; y++) {
        double row = 0;
        ii[y * (W + 1)] = 0;
        for (int x = 1; x <= W; x++) {
            row += px[(y - 1) * W + (x - 1)];
            ii[y * (W + 1) + x] = ii[(y - 1) * (W + 1) + x] + row;
        }
    }
}
static double rect_sum(const double *ii, int x, int y, int w, int h) {
    return ii[(y + h) * (W + 1) + x + w] - ii[y * (W + 1) + x + w] - ii[(y + h) * (W + 1) + x] + ii[y * (W + 1) + x];
}
#endif