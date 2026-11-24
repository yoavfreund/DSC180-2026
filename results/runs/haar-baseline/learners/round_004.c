/* rule: Haar haar4 at x=8, y=12, each rectangle 4x2 (brute-force search, training error 0.324) */
double weak_score(const vj_image *im) { return vj_haar4(im, 8, 12, 4, 2); }
