/* rule: Haar haar2_tb at x=0, y=0, each rectangle 1x6 (brute-force search, training error 0.387) */
double weak_score(const vj_image *im) { return vj_haar2_tb(im, 0, 0, 1, 6); }
