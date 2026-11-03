/* predict.c - reads the placeholder model written by train.c.
 *   ./predict MODEL IMAGES_DIR   -> "<id> <score>" per image
 *   ./predict --describe MODEL   -> "stage=<k> weak=<n>" per stage (reads only the model file)
 * Replace the scoring with your cascade; keep the output format and the
 * score convention. */
#include "common.h"

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: predict MODEL IMAGES_DIR | predict --describe MODEL\n"); return 2; }
    int describe = !strcmp(argv[1], "--describe");
    FILE *f = fopen(describe ? argv[2] : argv[1], "r");
    if (!f) { fprintf(stderr, "cannot open model\n"); return 1; }

    int stages;
    double thr, mean[2][NPIX];  /* mean[0] = face mean, mean[1] = non-face mean */
    if (fscanf(f, "placeholder %d thr1 %lf", &stages, &thr) != 2) return 1;
    for (int c = 0; c < 2; c++)
        for (int j = 0; j < NPIX; j++)
            if (fscanf(f, "%lf", &mean[c][j]) != 1) return 1;
    fclose(f);

    if (describe) {
        for (int s = 1; s <= stages; s++) printf("stage=%d weak=1\n", s);
        return 0;
    }

    int n = 0;
    char **ids = list_ids(argv[2], &n);
    if (!ids) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    char path[4096];
    double raw[NPIX], x[NPIX];
    for (int i = 0; i < n; i++) {
        snprintf(path, sizeof path, "%s/%s.pgm", argv[2], ids[i]);
        if (read_pgm(path, raw)) return 1;
        normalize(raw, x);
        double df = 0, dn = 0;
        for (int j = 0; j < NPIX; j++) {
            df += (x[j] - mean[0][j]) * (x[j] - mean[0][j]);
            dn += (x[j] - mean[1][j]) * (x[j] - mean[1][j]);
        }
        double score;
        if (df > thr) score = cascade_score(0, 4.0 * (thr - df) / thr);  /* rejected at stage 1 */
        else          score = cascade_score(1, (dn - df) / NPIX);        /* reached the last stage */
        printf("%s %.17g\n", ids[i], score);
    }
    return 0;
}