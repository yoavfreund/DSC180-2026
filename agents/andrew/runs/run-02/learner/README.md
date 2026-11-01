# Boosted attentional cascade (Viola-Jones variant)

- Features: 2-rect, 3-band, checkerboard and centre-surround Haar features on the 19x19 integral image
  (67k total; a fixed random 32k subset is used). They are zero-sum, so dividing by the patch std gives
  contrast normalization; flat patches use std = 1.
- Weak learner: a Real AdaBoost decision stump (output 0.5*ln(W+/W-) on each side), searched over
  64 quantile bins per feature.
- Cascade: 4 thresholded stages that stop at a target per-stage FP rate (0.50/0.45/0.40/0.40, at most
  6/12/30/60 stumps), then one final stage of 250 stumps with no threshold.
- Stage thresholds come from 15% of faces (plus mirrors) held out from training: the 99.8% quantile
  of their margins minus 1 std. The final stage trains on all faces.
- Few negatives: every training non-face goes into a pool in 8 dihedral variants x 5 one-pixel shifts
  (154k patches). Each stage trains on up to 6000 pool patches that earlier stages let through.
- Score: (k-1) + sigmoid((margin_k - thr_k)/scale_k) for a sample stopping at stage k.
