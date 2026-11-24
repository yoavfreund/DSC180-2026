/* rule: Haar haar2_lr at x=0, y=0, each rectangle 4x1 (brute-force search, training error 0.321) */
double weak_score(const vj_image *im) { return vj_haar2_lr(im, 0, 0, 4, 1); }
