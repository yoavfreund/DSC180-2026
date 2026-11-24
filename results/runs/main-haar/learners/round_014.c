/* rule: Haar haar3_lr at x=7, y=8, each rectangle 2x2 (brute-force search, training error 0.376) */
double weak_score(const vj_image *im) { return vj_haar3_lr(im, 7, 8, 2, 2); }
