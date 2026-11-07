#include "weak.h"
#include <math.h>

EvalResult evaluate(int n, const int *pred, const int *y, const double *w,
                    int count, const int *idx) {
    EvalResult bad = {NAN, NAN, NAN, NAN};
    double total = 0, pos = 0, neg = 0, errors = 0, fp = 0, fn = 0;
    if (n <= 0 || count <= 0 || !pred || !y || !w || !idx) return bad;
    for (int i = 0; i < count; ++i) {
        int r = idx[i];
        if (r < 0 || r >= n || !isfinite(w[r]) || w[r] < 0 ||
            (y[r] != 1 && y[r] != -1) || (pred[r] != 1 && pred[r] != -1)) return bad;
        total += w[r];
        if (y[r] == 1) pos += w[r]; else neg += w[r];
        if (pred[r] != y[r]) {
            errors += w[r];
            if (y[r] == 1) fn += w[r]; else fp += w[r];
        }
    }
    if (!(total > 0) || !isfinite(total)) return bad;
    EvalResult result = {errors / total, neg ? fp / neg : 0,
                         pos ? fn / pos : 0, pos / total};
    return result;
}
