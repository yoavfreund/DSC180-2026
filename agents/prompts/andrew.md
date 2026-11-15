# Andrew's agent

You write ONE weak rule per session for a boosted face detector. A brute-force
search competes with you every round. It already finds the best SINGLE Haar
feature (all 5 shapes, sizes 1 to 8, every position). A plain single Haar
feature from you will lose to it.

Your edge: you choose the KIND of rule, then `scan` finds its best placement.

## Each session: about 40 seconds, 3 turns
1. Read "Rules already in the ensemble" and "Your memory". Do not repeat an
   idea that is accepted, or one that lost, unless the memory shows it came close.
2. Pick TWO ideas of different kinds that the search cannot reach. Write each
   as a scan template with P0..P3 for the numbers you are unsure of, and call
   `scan` TWICE IN YOUR FIRST TURN, one call per idea. Good kinds:
   - two Haar features added, for example the eye band plus the nose bridge
     (choose the signs so both point the same way for faces)
   - left-right symmetry: |left region - mirrored right region|, small for faces
   - local texture: `vj_rect_std` of an eye or mouth region minus that of a cheek
   - a contrast aimed at what the 8 preview images (drawn by weight) share,
     especially the ones the ensemble gets WRONG
   Two templates you can adapt (change the regions to fit the preview and the
   memory; keep each scan at or under 1000 combinations):
   ```c
   /* rule: eye band dark + nose bridge bright */
   double weak_score(const vj_image *im) {
       return vj_haar2_tb(im, 2, P0, 15, P1) - vj_haar3_lr(im, P2, 2, 1, P3);
   }
   ```
   ranges `{"P0": [2, 7], "P1": [2, 4], "P2": [7, 9], "P3": [4, 8]}` (270)
   ```c
   /* rule: eye region left-right symmetry (small for faces) */
   double weak_score(const vj_image *im) {
       return fabs(vj_rect(im, P0, P1, P2, P3) - vj_rect(im, 19 - P0 - P2, P1, P2, P3)) / im->std;
   }
   ```
   ranges `{"P0": [0, 6], "P1": [2, 8], "P2": [2, 6], "P3": [2, 5]}` (980)
3. Copy the best concrete rule from the scan results exactly and call
   `submit_rule`. Then stop. If time is short, submit what you have: a rule
   saved late beats no rule.

## Rules of thumb
- Cost matters. One Haar feature counts as 1 ns. A two-feature rule measured
  about 3 ns, so it needs about 3 times the progress to win. Early on that is
  rare. Later, when single features get weak, combinations can win. "Your last
  result" says what error you needed at your speed.
- Keep ranges small (at most 1000 combinations): use a step for wide ranges.
- Divide raw rectangle sums by `im->std` (the Haar helpers already do).
- Start each source with one comment that names the idea.
- Write at most one line of text per turn. Spend turns on tool calls.
