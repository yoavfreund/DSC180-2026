/* rule: Haar haar3_tb at x=8, y=13, each rectangle 3x1 (brute-force search, training error 0.382) */
double weak_score(const vj_image *im) { return vj_haar3_tb(im, 8, 13, 3, 1); }
