#ifndef VJ_IMAGE_H
#define VJ_IMAGE_H

#include <string.h>
#include "vj_learner.h"

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

#endif
