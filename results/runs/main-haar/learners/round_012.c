/* rule: Haar haar2_tb at x=8, y=5, each rectangle 2x1 (brute-force search, training error 0.359) */
double weak_score(const vj_image *im) { return vj_haar2_tb(im, 8, 5, 2, 1); }
