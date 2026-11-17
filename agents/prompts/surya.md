# Surya's agent

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
   keep each scan at or under 1000 combinations
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


## More ideas beyond the single-Haar search
These are additional options for the two different ideas above, not claims that
any one will win. Adapt them to the preview, feedback, and accepted rules.
Keep rectangles fully inside the 19 x 19 image and each scan at or below 1000
combinations. The coordinator fits polarity, so either sign can identify a face.
All examples use only the provided helpers and have nonzero denominators.

### Separated bands: compare nonadjacent regions
A gap can distinguish eyes from mouth or forehead from cheeks. The baseline
compares adjacent Haar regions, so it does not search this arrangement.
```c
/* rule: separated upper and lower bands */
double weak_score(const vj_image *im) {
    return (vj_rect(im, P0, P1, P2, 2) - vj_rect(im, P0, P1 + P3, P2, 2)) / im->std;
}
```
ranges `{"P0": [2, 5], "P1": [2, 6], "P2": [7, 11], "P3": [4, 7]}` (400)

### Local center against its surroundings
Compare a region's mean to a larger enclosing region. This differs from the
baseline's adjacent Haar blocks and can describe a nose, eye, or mouth region.
```c
/* rule: local center against its surrounding area */
double weak_score(const vj_image *im) {
    return (vj_rect(im, P0, P1, P2, P3) / (double)(P2 * P3)
        - vj_rect(im, P0 - 2, P1 - 2, P2 + 4, P3 + 4)
          / (double)((P2 + 4) * (P3 + 4))) / im->std;
}
```
ranges `{"P0": [3, 6], "P1": [3, 6], "P2": [4, 7], "P3": [3, 6]}` (256)

### Region relative to the whole patch
Use normalized local brightness relative to the image mean; this can pick up
an isolated dark or bright facial region without an adjacent comparison.
```c
/* rule: local region relative to the whole patch */
double weak_score(const vj_image *im) {
    return vj_rect_mean(im, P0, P1, P2, P3);
}
```
ranges `{"P0": [2, 8, 2], "P1": [2, 8, 2], "P2": [3, 7, 2], "P3": [3, 7, 2]}` (144)

### Symmetric paired regions against a central region
Compare both eye-side patches to the middle rather than only comparing left
and right. Equal-size regions avoid a bias from different rectangle areas.
```c
/* rule: paired side regions against the center */
double weak_score(const vj_image *im) {
    return (vj_rect(im, P0, P1, P2, P3) + vj_rect(im, 19 - P0 - P2, P1, P2, P3)
        - 2.0 * vj_rect(im, 9 - P2 / 2, P1, P2, P3)) / im->std;
}
```
ranges `{"P0": [1, 4], "P1": [2, 7], "P2": [2, 4], "P3": [2, 4]}` (216)

### Unequal-scale texture contrast
Compare the local standard deviations of two separated regions, normalized by
window standard deviation. This may help when eye or mouth texture differs
from a cheek, but square roots can be costly; inspect measured speed.
```c
/* rule: upper texture against lower texture */
double weak_score(const vj_image *im) {
    return (vj_rect_std(im, P0, P1, P2, 3)
        - vj_rect_std(im, P0, P1 + P3, P2, 4)) / im->std;
}
```
ranges `{"P0": [2, 5], "P1": [2, 5], "P2": [5, 9], "P3": [5, 8]}` (320)

Treat the training scan's best error as optimistic. Compare the candidates'
training error and speed when possible; do not repeat an accepted or clearly
unsuccessful idea merely because it appears here. Never divide by a quantity
that can be zero or return NaN or infinity.
