/* rule: Haar haar3_tb at x=7, y=11, each rectangle 6x1 (brute-force search, training error 0.358) */
double weak_score(const vj_image *im) { return vj_haar3_tb(im, 7, 11, 6, 1); }
