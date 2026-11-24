/* rule: Haar haar2_lr at x=10, y=8, each rectangle 4x8 (brute-force search, training error 0.388) */
double weak_score(const vj_image *im) { return vj_haar2_lr(im, 10, 8, 4, 8); }
