/* Baseline: store the mean normalized face and non-face. */
#include "pgm.h"
int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: train IMAGES LABELS MODEL\n"); return 2; }
    FILE *lf = fopen(argv[2], "r");
    if (!lf) return 1;
    double mean[2][NPIX] = {{0}}, x[NPIX];
    long count[2] = {0, 0};
    char id[64], path[4096];
    int label;
    while (fscanf(lf, "%63s %d", id, &label) == 2) {
        snprintf(path, sizeof path, "%s/%s.pgm", argv[1], id);
        if (read_pgm(path, x)) { fprintf(stderr, "cannot read %s\n", path); return 1; }
        for (int i = 0; i < NPIX; i++) mean[label][i] += x[i];
        count[label]++;
    }
    fclose(lf);
    FILE *mf = fopen(argv[3], "w");
    for (int c = 0; c < 2; c++)
        for (int i = 0; i < NPIX; i++) fprintf(mf, "%.17g\n", mean[c][i] / count[c]);
    fclose(mf);
    fprintf(stderr, "trained on %ld faces, %ld non-faces\n", count[1], count[0]);
    return 0;
}
