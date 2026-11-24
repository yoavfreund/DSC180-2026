/* rule: Haar haar2_tb at x=8, y=7, each rectangle 3x1 (brute-force search, training error 0.381) */
double weak_score(const vj_image *im) { return vj_haar2_tb(im, 8, 7, 3, 1); }
