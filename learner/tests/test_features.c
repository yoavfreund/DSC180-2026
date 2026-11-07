#include "image_features.h"
#include "weak.h"
#include <assert.h>
#include <math.h>

int main(void) {
    uint8_t pixels[] = {1,2,3,4,5,6,7,8,9}, lbp[9];
    uint64_t integral[16], sum;
    assert(integral_image_u8(pixels, 3, 3, integral) == 0);
    assert(integral_rect_sum(integral, 3, 3, 1, 1, 3, 3, &sum) == 0 && sum == 28);
    assert(integral_rect_sum(integral, 3, 3, 0, 0, 3, 3, &sum) == 0 && sum == 45);
    assert(integral_rect_sum(integral, 3, 3, 2, 0, 1, 3, &sum) == -1);
    assert(lbp_u8(pixels, 3, 3, lbp) == 0 && lbp[4] == 120 && lbp[0] == 0);
    int y[] = {1, -1, 1}, p[] = {-1, -1, 1}, idx[] = {0,1,2};
    double w[] = {2, 3, 5};
    EvalResult e = evaluate(3, p, y, w, 3, idx);
    assert(fabs(e.err - .2) < 1e-12 && fabs(e.fnr - 2.0/7) < 1e-12);
    return 0;
}
