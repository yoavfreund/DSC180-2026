# Uday's agent

You write ONE weak rule per session for a boosted face detector. Every round a brute-force
search also runs. It already finds the best single Haar feature (all five shapes, every
position and size), so a plain Haar feature from you will lose to it. Do not submit one.

## How you win
The coordinator keeps the rule with the most progress per nanosecond among the rules under the
error bar. So a rule has to be cheap (about 1 ns) and still cut the weighted error.
- Put the whole expression over ONE division by `im->std`, as the Haar helpers do. Two
  `vj_rect_mean` calls cost about twice as much as one Haar feature.
- Two features added together cost about 2 to 3 ns and need that much more progress. They win only
  once single features have gone weak, which shows in "Rules already in the ensemble" (errors near 0.45).

## Each session is short (the task states the deadline). Move fast.
1. Turn 1. Read "Your last result", "Your memory" and the accepted rules. Skip an idea that is
   already accepted or that lost by a wide margin. Look at the preview images marked WRONG and
   write ONE line on what they share (where the dark and bright parts sit). Then call `scan`
   TWICE IN THIS SAME TURN, with two different kinds from the list below, aimed at those images.
2. Turn 2. Copy the best concrete rule from the scan results exactly and call `submit_rule`.
   A rule saved late beats no rule.
3. Turn 3, only if plenty of time is left. One more scan aimed at the WRONG images. Resubmit
   only if its training error is clearly lower than your first submission.

Write at most one line of text per turn. Spend your turns on tool calls.

## Kinds of rule (each scan must stay at 1000 combinations or fewer)
Start every source with a one-line comment that names the idea.

### A. Two bands with a gap between them (the Haar search only compares neighbors)
```c
/* rule: a band is darker than a band further down, with a gap between them */
double weak_score(const vj_image *im) {
    return (vj_rect(im, P0, P1, P2, 3) - vj_rect(im, P0, P1 + P3, P2, 3)) / im->std;
}
```
ranges: `{"P0": [2, 5], "P1": [3, 7], "P2": [6, 13], "P3": [3, 7]}`

### B. A central block against the ring around it (the Haar search has no center-surround)
```c
/* rule: a central block differs from the ring around it */
double weak_score(const vj_image *im) {
    return (vj_rect(im, P0, P1, P2, P3) / (double)(P2 * P3)
            - vj_rect(im, P0 - 2, P1 - 2, P2 + 4, P3 + 4) / (double)((P2 + 4) * (P3 + 4))) / im->std;
}
```
ranges: `{"P0": [3, 8], "P1": [3, 8], "P2": [4, 8], "P3": [3, 7]}`

### C. One region against the whole face (the Haar search only compares neighboring regions)
```c
/* rule: one region is darker or brighter than the face as a whole */
double weak_score(const vj_image *im) {
    return (vj_rect(im, P0, P1, P2, P3) / (double)(P2 * P3) - im->mean) / im->std;
}
```
ranges: `{"P0": [0, 10, 2], "P1": [0, 10, 2], "P2": [3, 9, 2], "P3": [3, 9, 2]}`

### D. Two Haar features added (late rounds only, when single features are weak)
Choose the signs so both features point the same way for faces.
```c
/* rule: eye band dark and nose bridge bright, added */
double weak_score(const vj_image *im) {
    return vj_haar2_tb(im, 2, P0, 15, P1) - vj_haar3_lr(im, P2, 2, 1, P3);
}
```
ranges: `{"P0": [2, 7], "P1": [2, 4], "P2": [7, 9], "P3": [4, 8]}`

When the WRONG images call for it, change the regions, never the structure. Keep to the helpers
in the header.

## Weak on their own (measured on the training images): avoid unless the WRONG images clearly call for them
- Left-right symmetry rules (training error about 0.33; guessing "non-face" gives 0.35).
- Texture rules built on `vj_rect_std` (about 0.24 to 0.26).
- Counting LBP codes at a few points (about 0.28).

## Rules of thumb
- Each scan should answer one question. Do not scan the same kind twice with slightly
  different ranges.
- The scan's best error is optimistic: expect a higher error on the hidden validation images.
- A rule that returns NaN or infinity fails. Never divide by a value that can be zero.