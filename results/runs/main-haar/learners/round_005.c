/* rule: Haar haar2_tb at x=3, y=4, each rectangle 8x2 (brute-force search, training error 0.335) */
double weak_score(const vj_image *im) { return vj_haar2_tb(im, 3, 4, 8, 2); }
