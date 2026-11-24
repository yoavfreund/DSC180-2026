/* rule: Haar haar4 at x=11, y=17, each rectangle 4x1 (brute-force search, training error 0.283) */
double weak_score(const vj_image *im) { return vj_haar4(im, 11, 17, 4, 1); }
