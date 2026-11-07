#ifndef IMAGE_FEATURES_H
#define IMAGE_FEATURES_H

#include <stdint.h>
#include <stddef.h>

/* Row-major 8-bit pixels. Output is (height+1) x (width+1), with a zero
 * top row and left column. The caller allocates the output buffers. */
int integral_image_u8(const uint8_t *pixels, size_t width, size_t height,
                      uint64_t *out);
/* Half-open rectangle [x0,x1) x [y0,y1). */
int integral_rect_sum(const uint64_t *integral, size_t width, size_t height,
                      size_t x0, size_t y0, size_t x1, size_t y1,
                      uint64_t *sum);
/* 8-neighbor LBP, clockwise from top-left, bit 0 first; neighbor >= center
 * sets a bit. Border pixels are zero. Output is width x height. */
int lbp_u8(const uint8_t *pixels, size_t width, size_t height, uint8_t *out);

#endif
