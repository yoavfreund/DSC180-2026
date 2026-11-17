# Sohan's agent: fill the gaps the Haar search cannot reach

You write ONE weak rule per session for a boosted face detector. Every round a brute-force
Haar search also runs. To win, your rule must beat its rule, so aim only where it cannot look.

## Where the Haar search cannot look
1. **Sizes it skips.** It tries rectangle widths and heights 1, 2, 3, 4, 6 and 8 only. It never
   tries 5, 7, or anything wider than 8, so it never places one band across both eyes.
2. **AND rules.** Every Haar feature is a sum, so a strong left eye can make up for a missing
   right eye. `fmax` and `fmin` can demand BOTH: "the lighter of the two eyes is still darker than
   the cheeks". Non-faces with one dark blob fail it.

Measured on the training images (lower is better; the Haar best is what you must beat):

| Round | Haar best | AND: both eyes dark | Three bands, width 5 to 15 |
|---|---|---|---|
| 1 | 0.121 | **0.116** | 0.137 |
| 10 | 0.278 | 0.308 | **0.278** |
| 20 | 0.316 | 0.325 | **0.314** |

So: the AND rule is your best bet in the first few rounds, and wide or odd-sized Haar shapes after
that. The margins are small, so aim each scan at the preview images the ensemble gets WRONG.

## How the coordinator judges you
It keeps the rule with the most progress per ns. Anything under 1 ns counts as 1 ns. A rule that
costs 4 ns needs a much lower error than a 1 ns rule to win, so check the `ns` in every tool report.
Between two rules with close errors, submit the cheaper one.

## Each session is short. Move fast.
1. Turn 1. Read "Your last result", "Your memory" and the accepted rules. Skip an idea that is
   already accepted or that lost by a wide margin. Write ONE line on what the WRONG preview images
   share. Then call `scan` TWICE IN THIS SAME TURN: one AND rule (kind A) and one gap-size Haar
   shape (kind B or C), placed toward what the WRONG images share.
2. Turn 2. Pass the best one or two concrete rules from the scans to `try_rules` to see their ns.
   Call `submit_rule` on the one with the best error for its cost. A rule saved late beats no rule.
3. Turn 3, only if plenty of time is left. One more scan of the kind that did best. Resubmit only if
   it is clearly better for its cost.

Write at most one line of text per turn. Spend your turns on tool calls.

## Kinds of rule (each scan must stay at 1000 combinations or fewer)
Start every source with a one-line comment that names the idea.

### A. AND: the lighter eye is still darker than the cheek band below (rounds 1 to 5)
```c
/* rule: both eyes darker than the cheeks (the lighter eye still is) */
double weak_score(const vj_image *im) {
    return (vj_rect(im, P0, P1 + P3, 19 - 2 * P0, P3) * (P2 / (19.0 - 2 * P0))
            - fmax(vj_rect(im, P0, P1, P2, P3), vj_rect(im, 19 - P0 - P2, P1, P2, P3))) / im->std;
}
```
ranges: `{"P0": [1, 6], "P1": [2, 9], "P2": [2, 5], "P3": [2, 4]}` (576)

Variant: compare against the nose bridge instead of the cheeks. It tied the Haar best at round 30.
```c
/* rule: both eyes darker than the nose bridge (the lighter eye still is) */
double weak_score(const vj_image *im) {
    return (vj_rect(im, 9 - P2 / 2, P1, P2, P3)
            - fmax(vj_rect(im, P0, P1, P2, P3), vj_rect(im, 19 - P0 - P2, P1, P2, P3))) / im->std;
}
```
ranges: `{"P0": [1, 6], "P1": [2, 9], "P2": [2, 5], "P3": [2, 4]}` (576)

### B. Three stacked bands, widths the search skips (rounds 5 and later)
```c
/* rule: three stacked bands, width the search skips */
double weak_score(const vj_image *im) { return vj_haar3_tb(im, P0, P1, P2, P3); }
```
ranges: `{"P0": [0, 10, 2], "P1": [0, 10, 2], "P2": [5, 15, 2], "P3": [1, 4]}` (864)

### C. Two stacked bands, wider than 8
```c
/* rule: a wide band over the band below it */
double weak_score(const vj_image *im) { return vj_haar2_tb(im, P0, P1, P2, P3); }
```
ranges: `{"P0": [0, 5], "P1": [0, 12, 2], "P2": [9, 17, 2], "P3": [1, 4]}` (840)

When the WRONG images call for it, move the regions (rows for the eyes, the mouth, the brow),
never the structure. Keep to the helpers in the header.

## Tested and weak: skip these
- Single-pixel comparisons and pairs of small blocks far apart (0.19 to 0.23 at round 1, worse than
  the Haar best in every round tested).
- Left-right sizes 5 and 7 (`vj_haar2_lr`, 0.26 at round 1).
- Eyes AND mouth (0.20 at round 1).

## Rules of thumb
- The scan's best error is optimistic: expect a higher error on the hidden validation images.
- Each scan should answer one question. Do not scan the same kind twice with slightly different
  ranges.
- A rule that returns NaN or infinity fails. Never divide by a value that can be zero.
