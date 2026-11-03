/* train.c - PLACEHOLDER 2-stage cascade so the whole pipeline runs end to end.
 *
 * Stage 1 rejects patches far from the mean face; stage 2 compares distance to
 * the mean face and the mean non-face. It is a template matcher, NOT boosted.
 * Replace main()'s model-building with your own boosted cascade; keep the
 * command line and the cascade score convention (see common.h). The model
 * file format is yours as long as predict.c reads it. */
#include "common.h"

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: train IMAGES_DIR LABELS MODEL_OUT\n"); return 2; }
    int n = 0;
    char **ids = list_ids(argv[1], &n);
    if (!ids || n == 0) { fprintf(stderr, "no images in %s\n", argv[1]); return 1; }
    double *px = malloc((size_t)n * NPIX * sizeof *px);
    int *lab = calloc(n, sizeof *lab);
    if (load_images(argv[1], ids, n, px)) { fprintf(stderr, "bad image\n"); return 1; }
    if (read_labels(argv[2], ids, n, lab) != n) { fprintf(stderr, "labels do not match images\n"); return 1; }

    double mean[2][NPIX] = {{0}}, x[NPIX];
    int cnt[2] = {0, 0};
    for (int i = 0; i < n; i++) {
        normalize(px + (size_t)i * NPIX, x);
        for (int j = 0; j < NPIX; j++) mean[lab[i]][j] += x[j];
        cnt[lab[i]]++;
    }
    for (int c = 0; c < 2; c++)
        for (int j = 0; j < NPIX; j++) mean[c][j] /= cnt[c] ? cnt[c] : 1;

    /* Stage-1 threshold: keep 99% of training faces. */
    double *d1 = malloc(cnt[1] * sizeof *d1);
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (!lab[i]) continue;
        normalize(px + (size_t)i * NPIX, x);
        double d = 0;
        for (int j = 0; j < NPIX; j++) d += (x[j] - mean[1][j]) * (x[j] - mean[1][j]);
        d1[k++] = d;
    }
    qsort(d1, k, sizeof *d1, cmp_double);
    double thr = d1[(int)(0.99 * (k - 1))];

    FILE *f = fopen(argv[3], "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    fprintf(f, "placeholder 2\nthr1 %.17g\n", thr);
    for (int c = 1; c >= 0; c--) {
        for (int j = 0; j < NPIX; j++) fprintf(f, "%.17g ", mean[c][j]);
        fprintf(f, "\n");
    }
    fclose(f);
    return 0;
}