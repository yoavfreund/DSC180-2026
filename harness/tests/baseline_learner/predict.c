/* Baseline: score = distance to mean non-face minus distance to mean face. */
#include "pgm.h"
#ifdef TWO_STAGE
#define DESCRIBE "stage=1 weak=1\nstage=2 weak=1\n"
#else
#define DESCRIBE "stage=1 weak=2\n" /* one stage: the two mean templates */
#endif

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--describe")) { fputs(DESCRIBE, stdout); return 0; }
    if (argc != 3) { fprintf(stderr, "usage: predict MODEL IMAGES\n"); return 2; }
    double mean[2][NPIX], x[NPIX];
    FILE *mf = fopen(argv[1], "r");
    if (!mf) return 1;
    for (int c = 0; c < 2; c++)
        for (int i = 0; i < NPIX; i++)
            if (fscanf(mf, "%lf", &mean[c][i]) != 1) return 1;
    fclose(mf);
    DIR *d = opendir(argv[2]);
    if (!d) return 1;
    struct dirent *e;
    char path[4096];
    while ((e = readdir(d))) {
        size_t n = strlen(e->d_name);
        if (n < 5 || strcmp(e->d_name + n - 4, ".pgm")) continue;
        snprintf(path, sizeof path, "%s/%s", argv[2], e->d_name);
        if (read_pgm(path, x)) return 1;
        double d0 = 0, d1 = 0;
        for (int i = 0; i < NPIX; i++) {
            d0 += (x[i] - mean[0][i]) * (x[i] - mean[0][i]);
            d1 += (x[i] - mean[1][i]) * (x[i] - mean[1][i]);
        }
        printf("%.*s %.17g\n", (int)(n - 4), e->d_name, d0 - d1);
    }
    closedir(d);
    return 0;
}
