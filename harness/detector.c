/* Score raw 19x19 patches with a linked, frozen detector. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "vj_image.h"

typedef double (*detector)(const vj_image *, int *);
double score_full(const vj_image *, int *);
double score_cascade(const vj_image *, int *);

static void die(const char *message) {
    fprintf(stderr, "detector: %s\n", message);
    exit(2);
}

static double prediction(const unsigned char *patch, detector score, int *used) {
    vj_image image;
    build_image(&image, patch);
    return score(&image, used);
}

static double now_ns(void) {
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0) die("cannot read clock");
    return time.tv_sec * 1e9 + time.tv_nsec;
}

static volatile double benchmark_sink;

static double time_batch(const unsigned char *raw, long n, detector score) {
    double total = 0;
    int used;
    double start = now_ns();
    for (long i = 0; i < n; ++i)
        total += prediction(raw + i * VJ_SIZE * VJ_SIZE, score, &used);
    double elapsed = now_ns() - start;
    benchmark_sink = total;
    if (!isfinite(total)) die("non-finite prediction");
    return elapsed / n;
}

int main(int argc, char **argv) {
    if (argc != 6) die("usage: detector score IMAGES N full|cascade OUT | time IMAGES N both REPS");
    long n = strtol(argv[3], NULL, 10);
    if (n <= 0) die("N must be positive");
    FILE *in = fopen(argv[2], "rb");
    unsigned char *raw = malloc((size_t)n * VJ_SIZE * VJ_SIZE);
    if (!in || !raw || fread(raw, VJ_SIZE * VJ_SIZE, (size_t)n, in) != (size_t)n)
        die("cannot load image batch");
    fclose(in);

    if (strcmp(argv[1], "score") == 0) {
        detector score = score_full;
        if (strcmp(argv[4], "cascade") == 0) score = score_cascade;
        else if (strcmp(argv[4], "full") != 0) die("unknown score mode");
        FILE *out = fopen(argv[5], "wb");
        if (!out) die("cannot open predictions output");
        for (long i = 0; i < n; ++i) {
            int used;
            double value = prediction(raw + i * VJ_SIZE * VJ_SIZE, score, &used);
            double result[2] = {value, used};
            if (!isfinite(value) || fwrite(result, sizeof(double), 2, out) != 2)
                die("cannot write prediction");
        }
        if (fclose(out) != 0) die("cannot finish predictions output");
    } else if (strcmp(argv[1], "time") == 0 && strcmp(argv[4], "both") == 0) {
        long reps = strtol(argv[5], NULL, 10);
        if (reps < 1 || reps > 1000) die("REPS must be 1..1000");
        time_batch(raw, n, score_full);
        time_batch(raw, n, score_cascade);
        for (long r = 0; r < reps; ++r) {
            double full, cascade;
            if (r % 2 == 0) {
                full = time_batch(raw, n, score_full);
                cascade = time_batch(raw, n, score_cascade);
            } else {
                cascade = time_batch(raw, n, score_cascade);
                full = time_batch(raw, n, score_full);
            }
            printf("%.6f %.6f\n", full, cascade);
        }
    } else {
        die("unknown operation");
    }
    free(raw);
    return 0;
}
