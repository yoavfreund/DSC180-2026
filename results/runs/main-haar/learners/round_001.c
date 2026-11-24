/* rule: Haar haar3_tb at x=14, y=3, each rectangle 3x4 (brute-force search, training error 0.267) */
double weak_score(const vj_image *im) { return vj_haar3_tb(im, 14, 3, 3, 4); }
