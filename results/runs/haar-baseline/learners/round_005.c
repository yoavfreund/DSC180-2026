/* rule: Haar haar2_tb at x=2, y=3, each rectangle 1x3 (brute-force search, training error 0.325) */
double weak_score(const vj_image *im) { return vj_haar2_tb(im, 2, 3, 1, 3); }
