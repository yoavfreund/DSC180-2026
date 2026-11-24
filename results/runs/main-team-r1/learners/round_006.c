/* rule: Haar haar3_lr at x=1, y=0, each rectangle 6x1 (brute-force search, training error 0.332) */
double weak_score(const vj_image *im) { return vj_haar3_lr(im, 1, 0, 6, 1); }
