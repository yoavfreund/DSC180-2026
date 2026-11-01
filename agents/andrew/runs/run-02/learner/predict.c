#include "common.h"
#include <dirent.h>

int main(int argc, char **argv) {
    Model M;
    if (argc == 3 && !strcmp(argv[1], "--describe")) {
        if (load_model(argv[2], &M)) { fprintf(stderr, "bad model\n"); return 1; }
        for (int s = 0; s < M.ns; s++) printf("stage=%d weak=%d\n", s + 1, M.s[s].n);
        return 0;
    }
    if (argc != 3) { fprintf(stderr, "usage: predict <model> <images_dir> | --describe <model>\n"); return 1; }
    if (load_model(argv[1], &M)) { fprintf(stderr, "bad model\n"); return 1; }
    /* last stage has no rejection threshold: its threshold only centres the score */
    DIR *d = opendir(argv[2]);
    if (!d) { perror(argv[2]); return 1; }
    struct dirent *e;
    char path[4096];
    unsigned char img[NPIX];
    Integ I;
    while ((e = readdir(d))) {
        size_t L = strlen(e->d_name);
        if (L < 5 || strcmp(e->d_name + L - 4, ".pgm")) continue;
        snprintf(path, sizeof path, "%s/%s", argv[2], e->d_name);
        double sc;
        if (read_pgm(path, img)) sc = 0; /* unreadable: lowest score */
        else { make_integ(img, &I); sc = cascade_score(&M, &I, NULL); }
        printf("%.*s %.17g\n", (int)(L - 4), e->d_name, sc);
    }
    closedir(d);
    return 0;
}
