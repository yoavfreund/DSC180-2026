/* rule: Haar haar3_tb at x=6, y=13, each rectangle 8x1 (brute-force search, training error 0.378) */
double weak_score(const vj_image *im) { return vj_haar3_tb(im, 6, 13, 8, 1); }
