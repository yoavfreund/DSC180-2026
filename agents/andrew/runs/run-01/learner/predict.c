#include "common.h"
#include <dirent.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: predict <model> <images_dir>\n"); return 1; }
    FILE *mf = fopen(argv[1], "r");
    if (!mf) { perror("model"); return 1; }
    int T;
    if (fscanf(mf, "%d", &T) != 1) return 1;
    Feat *fs = malloc(sizeof(Feat) * (T + 1));
    double *thr = malloc(sizeof(double) * (T + 1)), *al = malloc(sizeof(double) * (T + 1));
    int *pol = malloc(sizeof(int) * (T + 1));
    int n = 0;
    for (int r = 0; r < T; r++) {
        if (fscanf(mf, "%d %d %d %d %d %lf %d %lf", &fs[n].type, &fs[n].x, &fs[n].y, &fs[n].w, &fs[n].h,
                   &thr[n], &pol[n], &al[n]) != 8) break;
        n++;
    }
    fclose(mf);
    DIR *d = opendir(argv[2]);
    if (!d) { perror("dir"); return 1; }
    struct dirent *e;
    char path[4096];
    unsigned char px[NPIX];
    double ii[II * II];
    while ((e = readdir(d))) {
        size_t L = strlen(e->d_name);
        if (L < 5 || strcmp(e->d_name + L - 4, ".pgm")) continue;
        snprintf(path, sizeof path, "%s/%s", argv[2], e->d_name);
        double score = 0;
        if (read_pgm(path, px) == 0) {
            make_integral(px, 0, ii);
            for (int r = 0; r < n; r++) {
                float v = (float)feat_val(&fs[r], ii);
                int h = pol[r] > 0 ? (v < (float)thr[r]) : (v >= (float)thr[r]);
                score += h ? al[r] : -al[r];
            }
        }
        if (!isfinite(score)) score = 0;
        printf("%.*s %.9g\n", (int)(L - 4), e->d_name, score);
    }
    closedir(d);
    return 0;
}
