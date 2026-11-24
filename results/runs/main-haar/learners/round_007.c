/* rule: Haar haar3_tb at x=3, y=1, each rectangle 2x6 (brute-force search, training error 0.353) */
double weak_score(const vj_image *im) { return vj_haar3_tb(im, 3, 1, 2, 6); }
