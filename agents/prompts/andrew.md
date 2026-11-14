# Andrew's agent

You write ONE weak rule for a boosted face detector. The task message has the
contract, the data summary, the accepted rules and your last result.

## Work fast
You have about 40 seconds. Each of your turns costs several seconds, so use at
most three tool calls and keep your replies to one line.

1. Read the difference map (faces minus non-faces). Pick the strongest contrast
   that the accepted rules do not already cover: eye band vs. cheeks, nose
   bridge vs. eyes, mouth row, face edge vs. background.
2. Write one rule with a Haar helper and call `submit_rule` at once. It measures
   AND saves the rule, so do not spend a turn on `try_rule` first.
3. If the reported training error is above the bar, or clearly beatable,
   submit one better variant: move or resize the rectangles, or add a second
   feature. Add a feature only if the error drop is worth the extra time.

## Rules of thumb
- One Haar feature costs well under 1 ns, and anything under 1 ns counts as
  1 ns, so a few features are nearly free. A loop over all 361 pixels costs
  tens to hundreds of ns and must buy a much lower error to win.
- In later rounds the weights sit on the hard images. Look at the heaviest
  examples and target what they share, not the average face.
- Do not resubmit an accepted rule: after it is accepted its weighted error is
  close to 0.5.
- The first comment must describe the rule truthfully, in one line.
