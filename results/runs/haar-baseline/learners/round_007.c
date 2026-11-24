/* rule: Haar haar3_lr at x=8, y=9, each rectangle 3x1 (brute-force search, training error 0.343) */
double weak_score(const vj_image *im) { return vj_haar3_lr(im, 8, 9, 3, 1); }
