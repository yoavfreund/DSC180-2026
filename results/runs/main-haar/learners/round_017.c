/* rule: Haar haar3_tb at x=0, y=13, each rectangle 2x2 (brute-force search, training error 0.379) */
double weak_score(const vj_image *im) { return vj_haar3_tb(im, 0, 13, 2, 2); }
