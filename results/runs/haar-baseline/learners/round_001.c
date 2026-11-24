/* rule: Haar haar3_lr at x=3, y=0, each rectangle 4x6 (brute-force search, training error 0.230) */
double weak_score(const vj_image *im) { return vj_haar3_lr(im, 3, 0, 4, 6); }
