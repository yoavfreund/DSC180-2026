/* rule: Haar haar3_tb at x=0, y=6, each rectangle 1x2 (brute-force search, training error 0.368) */
double weak_score(const vj_image *im) { return vj_haar3_tb(im, 0, 6, 1, 2); }
