/* rule: Haar haar3_tb at x=5, y=0, each rectangle 1x2 (brute-force search, training error 0.377) */
double weak_score(const vj_image *im) { return vj_haar3_tb(im, 5, 0, 1, 2); }
