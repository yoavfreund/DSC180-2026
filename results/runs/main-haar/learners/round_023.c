/* rule: Haar haar2_lr at x=15, y=18, each rectangle 2x1 (brute-force search, training error 0.397) */
double weak_score(const vj_image *im) { return vj_haar2_lr(im, 15, 18, 2, 1); }
