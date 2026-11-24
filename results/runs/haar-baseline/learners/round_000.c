/* rule: Haar haar3_lr at x=8, y=3, each rectangle 1x6 (brute-force search, training error 0.125) */
double weak_score(const vj_image *im) { return vj_haar3_lr(im, 8, 3, 1, 6); }
