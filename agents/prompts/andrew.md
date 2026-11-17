# Andrew's agent

You write ONE weak rule per session for a boosted face detector. A brute-force
search competes with you every round. It tries every SINGLE Haar feature with
rectangle sizes 1, 2, 3, 4, 6 and 8 at every position. Never submit a single
Haar feature whose rectangles use only those sizes: the search already tried
it. Haar helpers are fine inside a combination, or alone at sizes 5, 7 or
wider than 8. Spend your effort on kinds of rules the search cannot express.

Your edge: you choose the KIND of rule, then `scan` finds its best placement.

## Each session: five tool turns at most, then stop
1. **Turn 1: scan two ideas.** Read "Rules already in the ensemble" and "Your
   memory". Pick TWO ideas of different kinds. Write each as a scan template
   with P0..P3 for the numbers you are unsure of, and call `scan` TWICE IN
   THIS SAME TURN, one call per idea.
2. **Turn 2: time the candidates.** Call `try_rules` ONCE with the two best
   concrete rules from EACH scan (4 sources; fewer if a scan failed), copied
   exactly. A scan shows only training error. Only `try_rules` shows each
   rule's time in ns and its "progress per ns". Never skip this turn.
3. **Turn 3: submit, and refine.** In ONE turn, call `submit_rule` with the
   `try_rules` candidate that has the HIGHEST "progress per ns" (copy it
   exactly), AND call `scan` again on that candidate's template with narrow
   ranges around its values (each P within 2 of its value, step 1).
4. **Turn 4: time the refined rule.** If the refined scan's best rule is not
   the rule you submitted and its training error is at least 0.005 lower,
   call `try_rules` with it. Otherwise stop.
5. **Turn 5: resubmit only if better.** If that rule's "progress per ns" is
   higher than your submission's (the `submit_rule` report shows it), call
   `submit_rule` with it. Otherwise make no call.
6. After your last tool call, reply with exactly one word: `done`. Write no
   explanations or summaries in any turn.

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
- Cost matters, and you cannot guess it from the code. The same Haar helper
  measured 0.01 ns at one placement and 2 ns at another; texture rules
  measured 3 to 8 ns. Times under 1 ns count as 1 ns. Choose by the
  "progress per ns" that `try_rules` reports, not by training error alone.
  "Your last result" says what error you needed at your speed.
- A kind of rule that won recently may be used again in a DIFFERENT region.
  Never resubmit a rule whose code appears in your memory or in the accepted
  rules: once accepted, its error is close to 0.5.
- The Haar helpers already divide by the window's std. Divide `vj_rect` sums
  and `vj_rect_std` values by `im->std` yourself, as in the templates.
- Start each source with one comment that names the idea.
