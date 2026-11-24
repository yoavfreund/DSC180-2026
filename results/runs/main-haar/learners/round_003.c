/* rule: Haar haar3_tb at x=8, y=9, each rectangle 3x2 (brute-force search, training error 0.330) */
double weak_score(const vj_image *im) { return vj_haar3_tb(im, 8, 9, 3, 2); }
