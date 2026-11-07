#include "image_features.h"
#include <stdint.h>
#include <string.h>

static int valid_size(size_t w, size_t h) {
    return w && h && w < SIZE_MAX && h < SIZE_MAX &&
           h <= SIZE_MAX / w && (h + 1) <= SIZE_MAX / (w + 1) &&
           (h + 1) * (w + 1) <= SIZE_MAX / sizeof(uint64_t) &&
           w * h <= UINT64_MAX / 255;
}

int integral_image_u8(const uint8_t *pixels, size_t w, size_t h, uint64_t *out) {
    if (!pixels || !out || !valid_size(w, h)) return -1;
    size_t stride = w + 1;
    memset(out, 0, stride * sizeof(*out));
    for (size_t y = 0; y < h; ++y) {
        uint64_t row = 0;
        out[(y + 1) * stride] = 0;
        for (size_t x = 0; x < w; ++x) {
            row += pixels[y * w + x];
            out[(y + 1) * stride + x + 1] = out[y * stride + x + 1] + row;
        }
    }
    return 0;
}

int integral_rect_sum(const uint64_t *a, size_t w, size_t h,
                      size_t x0, size_t y0, size_t x1, size_t y1, uint64_t *sum) {
    if (!a || !sum || !valid_size(w, h) || x0 > x1 || y0 > y1 || x1 > w || y1 > h) return -1;
    size_t s = w + 1;
    *sum = (a[y1*s+x1] - a[y0*s+x1]) - (a[y1*s+x0] - a[y0*s+x0]);
    return 0;
}

int lbp_u8(const uint8_t *p, size_t w, size_t h, uint8_t *out) {
    if (!p || !out || !valid_size(w, h)) return -1;
    memset(out, 0, w * h);
    for (size_t y = 1; y + 1 < h; ++y) for (size_t x = 1; x + 1 < w; ++x) {
        uint8_t c = p[y*w+x], v = 0;
        v |= (p[(y-1)*w+x-1] >= c) << 0;
        v |= (p[(y-1)*w+x]   >= c) << 1;
        v |= (p[(y-1)*w+x+1] >= c) << 2;
        v |= (p[y*w+x+1]     >= c) << 3;
        v |= (p[(y+1)*w+x+1] >= c) << 4;
        v |= (p[(y+1)*w+x]   >= c) << 5;
        v |= (p[(y+1)*w+x-1] >= c) << 6;
        v |= (p[y*w+x-1]     >= c) << 7;
        out[y*w+x] = v;
    }
    return 0;
}
