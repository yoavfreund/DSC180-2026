/*
 * runner: loads 19x19 images, builds a vj_image for each, and calls the
 * agent's weak_score(). Linked with one learner object per candidate.
 *
 *   runner score IMAGES N OUT    write N doubles (native byte order) to OUT
 *   runner time  IMAGES N REPS   print the median nanoseconds per weak_score() call
 *   runner scan  IMAGES N PARAMS write K*N doubles to PARAMS.out: one row of N scores
 *                                for each line "p0 p1 p2 p3" in PARAMS (templates only)
 *
 * IMAGES holds N * 361 bytes, one image after another, row-major (px[y][x]).
 * The vj_image build (integral images, mean, std) is outside the timed loop:
 * in a detector it is shared by every rule.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "vj_learner.h"

/* Template parameters for scan mode. A submitted rule cannot use them: the build
   rejects any reference to vj_p outside a scan template. */
int vj_p[4];

static void die(const char *msg) {
    fprintf(stderr, "runner: %s\n", msg);
    exit(2);
}

static void build_image(vj_image *im, const unsigned char *src) {
    double sum = 0.0, sum2 = 0.0;
    memcpy(im->px, src, VJ_SIZE * VJ_SIZE);
    for (int x = 0; x <= VJ_SIZE; x++) im->ii[0][x] = im->ii2[0][x] = 0;
    for (int y = 1; y <= VJ_SIZE; y++) {
        int row = 0, row2 = 0;
        im->ii[y][0] = im->ii2[y][0] = 0;
        for (int x = 1; x <= VJ_SIZE; x++) {
            int p = im->px[y - 1][x - 1];
            row += p;
            row2 += p * p;
            im->ii[y][x] = im->ii[y - 1][x] + row;
            im->ii2[y][x] = im->ii2[y - 1][x] + row2;
            sum += p;
            sum2 += (double)p * p;
        }
    }
    double n = VJ_SIZE * VJ_SIZE;
    im->mean = sum / n;
    double var = sum2 / n - im->mean * im->mean;
    im->std = var > 1.0 ? sqrt(var) : 1.0;
}

static vj_image *load(const char *path, long n) {
    if (n <= 0) die("N must be positive");
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open IMAGES");
    unsigned char *raw = malloc((size_t)n * VJ_SIZE * VJ_SIZE);
    vj_image *ims = malloc((size_t)n * sizeof(vj_image));
    if (!raw || !ims) die("out of memory");
    if (fread(raw, VJ_SIZE * VJ_SIZE, (size_t)n, f) != (size_t)n) die("IMAGES is shorter than N images");
    fclose(f);
    for (long i = 0; i < n; i++) build_image(&ims[i], raw + i * VJ_SIZE * VJ_SIZE);
    free(raw);
    return ims;
}

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc != 5) die("usage: runner score IMAGES N OUT | time IMAGES N REPS | scan IMAGES N PARAMS");
    long n = strtol(argv[3], NULL, 10);
    vj_image *ims = load(argv[2], n);

    if (strcmp(argv[1], "score") == 0) {
        double *out = malloc((size_t)n * sizeof(double));
        if (!out) die("out of memory");
        for (long i = 0; i < n; i++) out[i] = weak_score(&ims[i]);
        FILE *f = fopen(argv[4], "wb");
        if (!f || fwrite(out, sizeof(double), (size_t)n, f) != (size_t)n) die("cannot write OUT");
        fclose(f);
        return 0;
    }
    if (strcmp(argv[1], "time") == 0) {
        long reps = strtol(argv[4], NULL, 10);
        if (reps < 1 || reps > 1000) die("REPS must be 1..1000");
        double *per = malloc((size_t)reps * sizeof(double));
        volatile double sink = 0.0;
        /* Warm-up pass, which also sizes the timed passes: each rep lasts about
           2 ms, so a coarse clock (1 us on macOS) adds well under 1% error. */
        double t0 = now_ns();
        for (long i = 0; i < n; i++) sink += weak_score(&ims[i]);
        double one_pass = now_ns() - t0;
        long passes = one_pass > 0 ? (long)(2e6 / one_pass) : 1000;
        if (passes < 1) passes = 1;
        if (passes > 1000) passes = 1000;
        for (long r = 0; r < reps; r++) {
            t0 = now_ns();
            for (long p = 0; p < passes; p++)
                for (long i = 0; i < n; i++) sink += weak_score(&ims[i]);
            per[r] = (now_ns() - t0) / ((double)n * passes);
        }
        qsort(per, (size_t)reps, sizeof(double), cmp_double);
        printf("%.4f\n", per[reps / 2]);
        return 0;
    }
    if (strcmp(argv[1], "scan") == 0) {
        FILE *pf = fopen(argv[4], "r");
        if (!pf) die("cannot open PARAMS");
        char out_path[4096];
        snprintf(out_path, sizeof out_path, "%s.out", argv[4]);
        FILE *of = fopen(out_path, "wb");
        if (!of) die("cannot write PARAMS.out");
        double *row = malloc((size_t)n * sizeof(double));
        if (!row) die("out of memory");
        while (fscanf(pf, "%d %d %d %d", &vj_p[0], &vj_p[1], &vj_p[2], &vj_p[3]) == 4) {
            for (long i = 0; i < n; i++) row[i] = weak_score(&ims[i]);
            if (fwrite(row, sizeof(double), (size_t)n, of) != (size_t)n) die("cannot write PARAMS.out");
        }
        fclose(pf);
        fclose(of);
        return 0;
    }
    die("first argument must be score, time or scan");
    return 2;
}
