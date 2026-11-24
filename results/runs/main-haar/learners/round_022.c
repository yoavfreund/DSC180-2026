/* rule: Haar haar2_lr at x=1, y=9, each rectangle 3x8 (brute-force search, training error 0.392) */
double weak_score(const vj_image *im) { return vj_haar2_lr(im, 1, 9, 3, 8); }
