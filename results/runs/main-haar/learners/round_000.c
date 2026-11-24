/* rule: Haar haar3_lr at x=5, y=0, each rectangle 3x8 (brute-force search, training error 0.161) */
double weak_score(const vj_image *im) { return vj_haar3_lr(im, 5, 0, 3, 8); }
