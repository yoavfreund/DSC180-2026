/*
 * vj_learner.h: the contract between the coordinator and an agent's weak rule.
 *
 * An agent writes ONE function:
 *
 *     double weak_score(const vj_image *im);
 *
 * The coordinator fits the threshold and the polarity on the weighted training
 * images, so a high score may mean "face" or "non-face". The function must be
 * pure: the same image always gives the same number, and no state is kept
 * between calls. No #include, no I/O, no memory allocation. Only <math.h>
 * functions may be called (this header includes it).
 *
 * The harness builds each vj_image (pixels, integral images, mean, std) once.
 * In a detector that cost is shared by every rule, so it is NOT part of a
 * rule's timed cost.
 *
 * Coordinates: x is the column (0..18), y is the row (0..18), (0, 0) is the
 * top-left pixel. A rectangle (x, y, w, h) covers columns x..x+w-1 and rows
 * y..y+h-1. The helpers clamp rectangles to the image, so out-of-range values
 * do not crash; a rectangle clamped to nothing has sum 0.
 */
#ifndef VJ_LEARNER_H
#define VJ_LEARNER_H

#include <math.h>

#define VJ_SIZE 19

typedef struct {
    unsigned char px[VJ_SIZE][VJ_SIZE]; /* px[y][x], 0..255 */
    int ii[VJ_SIZE + 1][VJ_SIZE + 1];   /* ii[y][x] = sum of px over rows < y and columns < x */
    int ii2[VJ_SIZE + 1][VJ_SIZE + 1];  /* the same, for squared pixels */
    double mean;                        /* mean of the 361 pixels */
    double std;                         /* standard deviation of the 361 pixels, never below 1 */
} vj_image;

double weak_score(const vj_image *im);

static inline int vj_clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* Sum of the raw pixels in rectangle (x, y, w, h). */
static inline double vj_rect(const vj_image *im, int x, int y, int w, int h) {
    int x0 = vj_clamp(x, 0, VJ_SIZE), y0 = vj_clamp(y, 0, VJ_SIZE);
    int x1 = vj_clamp(x + w, 0, VJ_SIZE), y1 = vj_clamp(y + h, 0, VJ_SIZE);
    if (x1 <= x0 || y1 <= y0) return 0.0;
    return (double)(im->ii[y1][x1] - im->ii[y0][x1] - im->ii[y1][x0] + im->ii[y0][x0]);
}

/* Mean of rectangle (x, y, w, h) after the window is normalized to mean 0, std 1. */
static inline double vj_rect_mean(const vj_image *im, int x, int y, int w, int h) {
    int x0 = vj_clamp(x, 0, VJ_SIZE), y0 = vj_clamp(y, 0, VJ_SIZE);
    int x1 = vj_clamp(x + w, 0, VJ_SIZE), y1 = vj_clamp(y + h, 0, VJ_SIZE);
    if (x1 <= x0 || y1 <= y0) return 0.0;
    return (vj_rect(im, x, y, w, h) / ((x1 - x0) * (y1 - y0)) - im->mean) / im->std;
}

/* Standard deviation of the raw pixels inside rectangle (x, y, w, h). */
static inline double vj_rect_std(const vj_image *im, int x, int y, int w, int h) {
    int x0 = vj_clamp(x, 0, VJ_SIZE), y0 = vj_clamp(y, 0, VJ_SIZE);
    int x1 = vj_clamp(x + w, 0, VJ_SIZE), y1 = vj_clamp(y + h, 0, VJ_SIZE);
    if (x1 <= x0 || y1 <= y0) return 0.0;
    double n = (double)((x1 - x0) * (y1 - y0));
    double s = vj_rect(im, x, y, w, h);
    double s2 = (double)(im->ii2[y1][x1] - im->ii2[y0][x1] - im->ii2[y1][x0] + im->ii2[y0][x0]);
    double var = s2 / n - (s / n) * (s / n);
    return var > 0.0 ? sqrt(var) : 0.0;
}

/* One pixel, normalized: (px - mean) / std. */
static inline double vj_pixel(const vj_image *im, int x, int y) {
    return (im->px[vj_clamp(y, 0, VJ_SIZE - 1)][vj_clamp(x, 0, VJ_SIZE - 1)] - im->mean) / im->std;
}

/*
 * Viola-Jones Haar features. Every rectangle in a feature is w wide and h high.
 * Each result is divided by the window std (V-J variance normalization), and
 * each is 0 on a flat image.
 *
 *   vj_haar2_lr: left (x, y) minus right (x+w, y)                    [covers 2w x h]
 *   vj_haar2_tb: top (x, y) minus bottom (x, y+h)                    [covers w x 2h]
 *   vj_haar3_lr: 2 * middle (x+w, y) minus left (x, y) and right (x+2w, y)   [3w x h]
 *   vj_haar3_tb: 2 * middle (x, y+h) minus top (x, y) and bottom (x, y+2h)   [w x 3h]
 *   vj_haar4:    top-left + bottom-right minus top-right + bottom-left       [2w x 2h]
 */
static inline double vj_haar2_lr(const vj_image *im, int x, int y, int w, int h) {
    return (vj_rect(im, x, y, w, h) - vj_rect(im, x + w, y, w, h)) / im->std;
}
static inline double vj_haar2_tb(const vj_image *im, int x, int y, int w, int h) {
    return (vj_rect(im, x, y, w, h) - vj_rect(im, x, y + h, w, h)) / im->std;
}
static inline double vj_haar3_lr(const vj_image *im, int x, int y, int w, int h) {
    return (2.0 * vj_rect(im, x + w, y, w, h) - vj_rect(im, x, y, w, h) - vj_rect(im, x + 2 * w, y, w, h)) / im->std;
}
static inline double vj_haar3_tb(const vj_image *im, int x, int y, int w, int h) {
    return (2.0 * vj_rect(im, x, y + h, w, h) - vj_rect(im, x, y, w, h) - vj_rect(im, x, y + 2 * h, w, h)) / im->std;
}
static inline double vj_haar4(const vj_image *im, int x, int y, int w, int h) {
    return (vj_rect(im, x, y, w, h) + vj_rect(im, x + w, y + h, w, h)
            - vj_rect(im, x + w, y, w, h) - vj_rect(im, x, y + h, w, h)) / im->std;
}

/* 8-neighbor local binary pattern code (0..255) at (x, y), clamped to the interior. */
static inline int vj_lbp(const vj_image *im, int x, int y) {
    static const int dx[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
    static const int dy[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
    x = vj_clamp(x, 1, VJ_SIZE - 2);
    y = vj_clamp(y, 1, VJ_SIZE - 2);
    int c = im->px[y][x], code = 0;
    for (int k = 0; k < 8; k++) code |= (im->px[y + dy[k]][x + dx[k]] >= c) << k;
    return code;
}

#endif
