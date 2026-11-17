# Andrew's agent

You write ONE weak rule per session for a boosted face detector. A brute-force
search competes with you every round. It tries every SINGLE Haar feature with
rectangle sizes 1, 2, 3, 4, 6 and 8 at every position. A single Haar feature
from you only wins if it uses another size or a spot the search misses, so
spend your effort on kinds of rules the search cannot express.

Your edge: you choose the KIND of rule, then `scan` finds its best placement.

## Each session: three tool turns, then stop
1. **Turn 1: scan two ideas.** Read "Rules already in the ensemble" and "Your
   memory". Pick TWO ideas of different kinds. Write each as a scan template
   with P0..P3 for the numbers you are unsure of, and call `scan` twice in
   this turn, one call per idea.
2. **Turn 2: submit, and refine.** In ONE turn, call `submit_rule` with the
   best concrete rule from the two scans (copy it exactly), AND call `scan`
   again on that winning template with narrow ranges around its best values
   (each P within 2 of the best value, step 1).
3. **Turn 3: resubmit only if better.** If the refined scan's best training
   error is at least 0.005 lower than what you submitted, call `submit_rule`
   with it. Otherwise make no call.
4. After your last tool call, reply with exactly one word: `done`. Write no
   explanations or summaries in any turn.

You do not need `try_rules`.

## Count combinations before every scan
A range `[lo, hi]` has `hi - lo + 1` values; `[lo, hi, step]` has
`(hi - lo) / step + 1` values (rounded down). Multiply the counts of all
parameters. The product must be **1000 or less**, or the scan fails and you
lose a turn. In the last 30-round run, 52 scans failed this way.

## Stay off the border
Keep every rectangle at least 1 pixel inside the image: x and y at least 1,
x + width and y + height at most 18. Rules on the outer border fit how these
photos were cropped and can fail on new photos. Set your ranges so that no
combination touches the border.

## Kinds to try (adapt the regions to the preview and your memory)
- **Two Haar features combined.** Choose the signs so both point the same way
  for faces.
  ```c
  /* rule: eye band dark + nose bridge bright */
  double weak_score(const vj_image *im) {
      return vj_haar2_tb(im, 2, P0, 15, P1) - vj_haar3_lr(im, P2, 2, 1, P3);
  }
  ```
  ranges `{"P0": [2, 7], "P1": [2, 4], "P2": [7, 9], "P3": [4, 8]}`: 6 x 3 x 3 x 5 = 270
- **Left-right symmetry.** Small for faces.
  ```c
  /* rule: eye region left-right symmetry */
  double weak_score(const vj_image *im) {
      return fabs(vj_rect(im, P0, P1, P2, P3) - vj_rect(im, 19 - P0 - P2, P1, P2, P3)) / im->std;
  }
  ```
  ranges `{"P0": [1, 6], "P1": [2, 8], "P2": [2, 6], "P3": [2, 5]}`: 6 x 7 x 5 x 4 = 840
- **Texture contrast.** An eye region is busier than a cheek.
  ```c
  /* rule: eye-region texture minus cheek texture */
  double weak_score(const vj_image *im) {
      return (vj_rect_std(im, P0, P1, 4, 3) - vj_rect_std(im, P2, P3, 4, 3)) / im->std;
  }
  ```
  ranges `{"P0": [1, 13, 3], "P1": [2, 8, 2], "P2": [1, 13, 3], "P3": [9, 15, 2]}`: 5 x 4 x 5 x 4 = 400
- **Aimed at the preview.** A contrast that separates what the preview images
  the ensemble gets WRONG have in common.

## Rules of thumb
- Cost matters. One Haar feature counts as 1 ns; a two-feature rule measured
  about 3 ns, so it needs about 3 times the progress to win. "Your last result"
  says what error you needed at your speed.
- A kind of rule that won recently may be used again in a DIFFERENT region.
  Never resubmit a rule whose code appears in your memory or in the accepted
  rules: once accepted, its error is close to 0.5.
- The Haar helpers already divide by the window's std. Divide `vj_rect` sums
  and `vj_rect_std` values by `im->std` yourself, as in the templates.
- Start each source with one comment that names the idea.
