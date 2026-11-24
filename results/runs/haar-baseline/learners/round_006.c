/* rule: Haar haar2_tb at x=12, y=3, each rectangle 6x1 (brute-force search, training error 0.342) */
double weak_score(const vj_image *im) { return vj_haar2_tb(im, 12, 3, 6, 1); }
