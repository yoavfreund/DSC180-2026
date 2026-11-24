/* rule: Haar haar3_lr at x=8, y=4, each rectangle 1x6 (brute-force search, training error 0.359) */
double weak_score(const vj_image *im) { return vj_haar3_lr(im, 8, 4, 1, 6); }
