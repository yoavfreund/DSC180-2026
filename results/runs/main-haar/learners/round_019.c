/* rule: Haar haar3_tb at x=13, y=2, each rectangle 6x3 (brute-force search, training error 0.382) */
double weak_score(const vj_image *im) { return vj_haar3_tb(im, 13, 2, 6, 3); }
