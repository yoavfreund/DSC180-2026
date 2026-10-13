---
marp: true
size: 16:9
paginate: true
theme: default
style: |
  section { font-family: Helvetica, Arial, sans-serif; font-size: 28px; padding: 56px 70px; }
  h1 { font-size: 46px; }
  h2 { font-size: 38px; margin: 0 0 24px 0; line-height: 1.2; }
  table { font-size: 24px; }
  th { background: #eef3f9; }
  .rep { background: #fff3c4; }
  .good { color: #2e8b57; font-weight: bold; }
  .small { font-size: 20px; color: #555; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
  .big { font-size: 40px; text-align: center; margin: 30px 0; }
  pre { font-size: 23px; }
  blockquote { border-left: 5px solid #1f5fa8; padding-left: 18px; font-style: italic; }
---

# Bagging Predictors

### Leo Breiman, *Machine Learning* 1996

Sohan Shingade · DSC 180A · Week 3

<!--
Bagging is a way to make a shaky learner reliable by training it many times and letting the copies vote. I'll build it up from one dataset and one decision tree.
-->

---

## 1. The task: predict diabetes from 8 numbers

| Patient | Glucose | BMI | Age | … | Diabetic? |
|---|---|---|---|---|---|
| A | 148 | 33.6 | 50 | | yes |
| B | 85 | 26.6 | 31 | | no |
| C | 183 | 23.3 | 32 | | yes |
| D | 89 | 28.1 | 21 | | no |
| E | 137 | 43.1 | 33 | | yes |
| F | 116 | 25.6 | 30 | | no |

768 patients like this (Pima Indians Diabetes, Kaggle). Breiman used this same dataset.

<!--
I want to predict whether a patient gets diabetes from 8 measurements. 768 patients. Breiman used this exact dataset in the paper, so I can compare my numbers to his later.
-->

---

## 2. One decision tree is a shaky guess

<div class="cols">
<div>

**Tree trained on A–F**

```
glucose > 127 ?
  yes → diabetic
  no  → not diabetic
```

</div>
<div>

**Swap out two patients, retrain**

```
BMI > 30 ?
  yes → age > 40 ? …
  no  → not diabetic
```

</div>
</div>

<br>

Change a little data and the whole tree changes. Breiman calls this **unstable**.

The single tree in the paper got 23% of diabetes patients wrong.

<!--
A decision tree asks yes/no questions it picks from the data. The problem: change a few patients and it picks completely different questions. Breiman's word is "unstable". So any one tree is a noisy guess.
-->

---

## 3. Noisy guesses average out, but I only have one dataset

If I had 50 **independent** datasets, I could train 50 trees and let them vote. Their individual mistakes would cancel.

<div class="big">one dataset → one tree → one noisy guess</div>

I can't go collect 50 more sets of 768 patients.

**So: can I fake 50 datasets from the one I have?**

<!--
If I had 50 independent datasets I could train 50 trees and average them. But I have one dataset. So the question is whether I can fake the other 49.
-->

---

## 4. Bootstrap: resample your own data, with replacement

Draw 6 patients from {A, B, C, D, E, F} at random. Put each one back before the next draw.

| Copy | Draws | Repeated | Left out |
|---|---|---|---|
| 1 | <span class="rep">A A</span> C D <span class="rep">E E</span> | A, E | B, F |
| 2 | <span class="rep">B B</span> <span class="rep">C C</span> D F | B, C | A, E |
| 3 | A B D E <span class="rep">F F</span> | F | C |

Each copy is the same size as the original, but some rows repeat and some are missing.

<span class="small">"Bootstrap": Efron, 1979. Breiman's idea was to feed these copies to a learner.</span>

<!--
Bootstrap: draw 768 patients from my 768 at random, putting each one back after drawing. So some patients appear twice, some not at all. Each copy is the same size as the original but a different mix. Do it 50 times, 50 copies.
-->

---

## 5. Why a resample is a fair fake of a new dataset

<div class="cols">
<div>

**What actually happened**

The world has millions of people.
My 768 patients are a **random sample** of them.
Another study would get a *different* 768.

</div>
<div>

**What bootstrap imitates**

Treat my 768 as if they *were* the world.
A bootstrap copy is a **random sample** of them.
Another copy is a *different* 768.

</div>
</div>

<br>

The amount a tree **wobbles** between bootstrap copies is close to how much it would wobble between real studies. That is the whole bet.

It works because the 768 are a decent stand-in for the population. With 6 patients it would not.

<!--
Why is resampling my own data a fair fake? My 768 patients are a random sample from the world. A bootstrap copy is a random sample from my 768. The same relationship one level down. So the way a tree wobbles between bootstrap copies mimics how it would wobble between real studies. It only works because 768 is big enough to stand in for the population.
-->

---

## 6. Each copy sees about 63% of the patients

Chance a given patient is **never** drawn in 768 draws: (1 − 1/768)⁷⁶⁸ ≈ 1/e ≈ **37%**.

So each bootstrap copy contains about **63%** of the distinct patients, with the rest filled by repeats.

<div class="big">different 63% → different tree</div>

That is exactly what we want: 50 copies give 50 genuinely different trees.

<!--
Each copy misses about 37% of the patients, so it has about 63% of them plus repeats. That's why the 50 trees come out different. If the learner were stable this wouldn't matter; for a tree it matters a lot.
-->

---

## 7. Bagging = bootstrap + aggregate

```
for b in 1 … 50:
    copy_b = bootstrap sample of the data
    tree_b = train a tree on copy_b

to predict a new patient:
    ask all 50 trees
    label  →  majority vote
    number →  average
```

Nothing inside the tree learner changes. Breiman: all you add is "a loop in front … and a back end that does the aggregation."

<!--
Bagging is just that: 50 bootstrap copies, 50 trees, majority vote. For predicting numbers, average instead. The learner itself is untouched. That's the entire algorithm.
-->

---

## 8. Three trees vote on a new patient

New patient: glucose 130, BMI 35, age 45.

| Tree | Trained on | Its first question | Says |
|---|---|---|---|
| 1 | copy 1 | glucose > 127? | diabetic |
| 2 | copy 2 | BMI > 30? | diabetic |
| 3 | copy 3 | age > 48? | not diabetic |

<div class="big">2 – 1 → diabetic</div>

A 2–1 vote also says something a single tree never can: **this one isn't certain.**

<!--
Three trees from three copies. Each latched onto a different pattern because each saw different patients. Two say diabetic, one says not. Majority wins. And the 2-1 split itself tells us the prediction is shaky. A single tree can't tell you that.
-->

---

## 9. Why it works: averaging cancels noise

<div class="cols">
<div>

**Unstable learner (tree)**

50 copies → 50 different trees
Averaging their answers removes the wobble

<span class="good">big gain</span>

</div>
<div>

**Stable learner (nearest neighbour)**

50 copies → 50 nearly identical answers
Nothing to average away

**no gain**

</div>
</div>

<br>

Breiman's proof: the squared error of the average is never above the average squared error. The gap is the learner's variance. No variance, no gap.

> "Bagging unstable classifiers usually improves them. Bagging stable classifiers is not a good idea."

<!--
Why does it work? Averaging cancels noise. If the learner wobbles a lot between copies, averaging removes the wobble. If it barely wobbles, like nearest neighbour, there's nothing to remove. Breiman proves the averaged error is never worse, and the gain is the learner's variance.
-->

---

## 10. Breiman's results: better on every dataset

| Dataset | One tree | 50 bagged | Decrease |
|---|---|---|---|
| waveform | 29.0 | 19.4 | 33% |
| heart (UCSD Medical Center) | 10.0 | 5.3 | 47% |
| breast cancer | 6.0 | 4.2 | 30% |
| ionosphere | 11.2 | 8.6 | 23% |
| **diabetes** | **23.4** | **18.8** | **20%** |
| glass | 32.0 | 24.9 | 22% |
| soybean | 14.5 | 10.6 | 27% |

Test error %, averaged over 100 random splits. Nearest neighbour: **no change** on any dataset.

How many copies? 10 gets most of the gain, 25 nearly all. *"More than 25 is love's labor lost."*

<!--
Seven datasets, error down 20 to 47 percent on every one. The heart data is from UCSD Medical Center. Nearest neighbour, the stable learner, didn't move at all. And you don't need 50 copies: 10 gets most of it.
-->

---

## 11. My run: same story on the diabetes data

50 random 80/20 splits, sklearn trees. Error is mean ± 95% CI.

| Method | Test error % | Train (ms) | Predict (µs / patient) |
|---|---|---|---|
| one tree | 30.1 ± 1.0 | 1.1 | 0.2 |
| **50 bagged trees** | <span class="good">23.9 ± 0.8</span> | 45 | 9.6 |
| nearest neighbour | 29.5 ± 0.8 | 0.2 | 2.9 |
| 50 bagged NN | 29.4 ± 0.8 | 18 | 106 |

- **Accuracy:** 21% less error (Breiman: 20%). Bagging won on 49 of 50 splits.
- **Confidence:** when the 50-tree vote is lopsided (>45–5) it is right **92%** of the time; when it is close (<30–20) only **57%**.
- **Speed:** ~40× slower to train and predict, but still 10 µs per patient.

<!--
I ran it on the diabetes data. One tree: 30% error. 50 bagged trees: 24%. That's a 21% drop; Breiman got 20% on the same data. Nearest neighbour didn't move, as predicted. For confidence, I looked at the vote split: lopsided votes are right 92% of the time, close votes are a coin flip. Cost is about 40x, but that's 10 microseconds per patient.
-->

---

## 12. Takeaways

1. **Bootstrap** fakes new datasets by resampling the one you have. It is a fair fake because your sample stands in for the population.
2. **Bagging** = bootstrap copies + vote. The learner is untouched.
3. It fixes **unstable** learners (trees: −20% to −47%). Stable ones gain nothing.
4. The **vote split** is a free confidence estimate.
5. Each copy trains with **zero communication** to the others. Breiman: "almost a dream procedure for parallel computing." That is why it maps onto a swarm of learner agents and one coordinator that votes.

<span class="small">Code: `slides/bagging_1996/bagging_pima_diabetes.ipynb`</span>

<!--
Takeaways. Bootstrap fakes datasets. Bagging votes over them. It works on unstable learners, not stable ones. The vote split gives confidence for free. And each copy trains alone, which is exactly why it fits our agents project: one coordinator, many independent learners.
-->
