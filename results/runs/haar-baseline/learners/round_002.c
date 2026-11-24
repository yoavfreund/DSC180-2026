/* rule: Haar haar3_tb at x=9, y=4, each rectangle 8x3 (brute-force search, training error 0.253) */
double weak_score(const vj_image *im) { return vj_haar3_tb(im, 9, 4, 8, 3); }
