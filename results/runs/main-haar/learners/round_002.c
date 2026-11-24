/* rule: Haar haar4 at x=13, y=15, each rectangle 3x2 (brute-force search, training error 0.305) */
double weak_score(const vj_image *im) { return vj_haar4(im, 13, 15, 3, 2); }
