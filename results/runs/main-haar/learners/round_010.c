/* rule: Haar haar2_tb at x=1, y=8, each rectangle 1x3 (brute-force search, training error 0.371) */
double weak_score(const vj_image *im) { return vj_haar2_tb(im, 1, 8, 1, 3); }
