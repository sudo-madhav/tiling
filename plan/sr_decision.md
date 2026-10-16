# Tile consistency as the CVPR paper — decision

Written 2026-10-15 on branch `sr`. Raw idea, then a decision. Literature review and
setup wait until this document is agreed. The defocus-deblurring plan in
`plan/gap.md` is paused, not deleted.

## 0. The raw idea

Tiled inference of recent super-resolvers (VOSR, CASR, SeedVR, and others)
produces inconsistencies between tiles. First define a metric, reference-based
or no-reference, that scores tile consistency. Validate it by taking clean
images, cutting them into tiles, injecting inconsistencies, and checking that
the score ranks the injections. Then change the super-resolver so that it
scores better on this metric than existing methods. That comparison is the paper.

## 1. Decision

**Yes, this can be the CVPR paper. Not in the form above.**

The paper that survives review is a metric for **content disagreement between
tiles**, plus a demonstration that current super-resolvers fail it and that
the usual scores do not see the failure. A new super-resolver is in the paper
only if the obvious fixes leave that disagreement in place.

Two narrower papers should not be written:

- A seam detector. Visible grid lines are a solved engineering problem and a
  published metric. Overlap, Gaussian blending, and drawing the VAE posterior
  mode are what VOSR already ships for "seam-free" tiled inference. SWITi
  (Carrara et al., arXiv:2607.18990, July 2026) defines two no-reference seam
  scores, Fraction of Rejected Tests and Artifact Severity, by testing whether
  gradients on a known seam differ from gradients beside it. They report that
  PSNR and SSIM barely see those seams. A second seam score loses to that paper.
- A super-resolver whose claim is "higher on our new score." If the score is
  fit or tuned on the same runs, the comparison is circular. Reviewers treat
  that as training to the test.

**Situation.** Same technical problem, different data. Seam statistics are
situation 1: JPEG blockiness, then SWITi, already score a line at a known
coordinate. Content disagreement is situation 3. CASR (Guo et al., CVPR 2026,
arXiv:2602.22159) states it and shows it: Figure 2 is the same repeated object
reconstructed with different textures in different patches. They reduce it
with a self-similarity module and MultiDiffusion merging, then score PSNR,
LPIPS, MUSIQ, NIQE, and PI. They never publish a number whose only job is
that disagreement. SyncDiffusion's Intra-LPIPS is a pairwise distance between
windows of a panorama, not a calibrated score for super-resolution tiles.
That is enough open ground for a paper, and not enough to skip the controls
in section 4.

## 2. Is the motivation good enough?

**Yes, once the failure is named precisely. The grid line is not the failure.**

Four different things get called "tiling" in the papers and in the notes:

| Kind | What a person sees | Who already handles it |
|---|---|---|
| Seam | A line on the tile boundary | Overlap and blending. VOSR uses posterior mode plus Gaussian blend and calls the result seam-free. SeedVR issues (ByteDance-Seed/SeedVR #54) are mostly VAE tile lines and conv splits, fixed by overlap and by not splitting the convolution. SWITi measures this. |
| Tone | One tile brighter, warmer, or flatter than the next, with no hard edge | Global color fix, wavelet color fix, histogram match. Often removed without touching texture. |
| Statistics | Grain, sharpness, or noise level changes across a tile with no edge | VAE GroupNorm and per-tile latent statistics. ScaleCrafter discusses this for high resolution. Easy to confuse with a real change in the scene. |
| Content | The same material is a different texture, glyph, or pattern in the next tile, and the boundary can look smooth | CASR Figure 2. This is the one standard metrics miss, because each tile is a plausible sharp image and there is no ground-truth pixel for a generated texture. |

PSNR and DISTS compare with a sharp photograph. They punish invented detail
even when every tile agrees, and they forgive two tiles that invent different
detail if both stay near the low-resolution input. MUSIQ and CLIP-IQA go up
when a tile is sharp, whether or not its neighbor invented a different weave.
A metric that only restates those scores does not change a decision.

The motivation a reviewer will accept is one sentence: **for generative
super-resolution of images that do not fit in memory, each tile is a separate
draw, and no published score tells you whether those draws are the same
image.** CASR had to show that with a picture. SeedVR's users can remove the
line and still argue about waxy, context-starved tiles. That is a real gap.
"Tiling looks bad" is not.

## 3. What in the raw plan does not hold

- **Reference-based and no-reference are not two versions of the same metric.**
  The clean full image is available only in the injection test. The images
  that need tiling do not fit in one forward pass, so a distance to a
  full-frame run is an oracle for the lab, not the metric in the paper.
  Ship a no-reference score. Keep the full-frame distance as a ceiling during
  validation, on images small enough to run both ways.
- **"Contrastive" is a test, not a loss to train the metric on.** A scorer
  trained on your injection procedure learns the procedure. Define the score
  first. Inject second, as a test. A learned scorer is allowed only for the
  inconsistency type the defined score cannot rank, and it is trained on
  injections that the test set does not use.
- **"Once the metric works, consistency will be easy" is the claim that can
  kill the method half.** Overlap, shared noise, and a color match may already
  win on seam and tone. If they also win on content, the paper is the metric
  and a negative result about those fixes. Writing a new network anyway is
  the follow-up a reviewer calls incremental.
- **Beating other super-resolvers on the new metric is the last table, not
  the contribution.** The contribution is that the metric ranks a controlled
  disagreement and that humans agree. The table is the consequence.

## 4. Approach

Independent of the raw plan. One failure, then the smallest score that
isolates it, then a method only if the small fixes fail.

### 4.1 The failure

Content disagreement: two tiles of one photograph, processed by one
super-resolver, describe the same surface differently, and the description
does not jump only on the cut.

Seam, tone, and grain stay in the paper as controls. A score that fires on
them and stays silent on a texture swap has measured the wrong thing. A score
that fires on a real window frame lying on the tile grid has measured an edge,
not an artifact.

### 4.2 The score, before any new network

No-reference. It may use the tile grid, which the runner knows, and must not
use a sharp ground truth.

Working definition, to be replaced if the tests say so: inside each tile,
compare a descriptor of texture (a frozen feature, not pixels) with the
descriptor of the same material in other tiles, after the low-frequency
band of the whole image has been removed. The score is the disagreement among
tiles that the low-resolution image says are the same material, minus the
disagreement the low-resolution image itself already has. Subtracting the
input stops the score from punishing a photograph that really does change
across the frame.

That is a hypothesis, not a result. It is the one to implement first because
it uses the grid and the input, needs no training, and is blind to a clean
high-frequency seam. SWITi remains the baseline for the seam control.

### 4.3 Prove the score before using it on a method

Clean photographs only. Pick a grid. Make five copies:

1. Untouched.
2. A line at the boundary.
3. A per-tile gain and color shift, blended so there is no line.
4. A per-tile change of grain or sharpness, no line.
5. A swap of a repeated texture for a different texture of the same stuff,
   pasted with a blend so a gradient test sees nothing.

Pre-register the pass rule before looking at numbers:

- On copies 1, the score is near the score of a shifted grid on the same
  photo (the grid is in the wrong place).
- On copy 5, the score rises with the fraction of tiles swapped, Spearman
  against that fraction above 0.8 on a held-out set of photos.
- On copies 2 and 3, SWITi or a color statistic moves, and this score stays
  near copy 1. If this score moves as much as on copy 5, it is not a content
  score.
- A real edge laid on the grid (a building facade, a window) does not raise
  the score the way copy 5 does.

Then a small two-alternative forced choice: people pick the more inconsistent
image in pairs drawn from these copies and from real tiled outputs. The score
must agree with the people on the content pairs more often than LPIPS, DISTS,
MUSIQ, and SWITi. Twenty pairs and ten people is enough to keep or kill the
score. It is not a user study for a product.

If copy 5 fails, stop and change the descriptor. Do not train a network to
hide the failure.

### 4.4 The method, only after 4.3 passes

Three pipelines, the first one deliberately dumb. Run all three on the same
tiled super-resolvers. The code to call VOSR is already in this repo. CASR
and SeedVR are the external ones.

1. Overlap, linear or Gaussian blend, one noise field for the image, one
   global color match to the upsampled input.
2. SWITi: average shifted tiles. Record the cost in high-frequency energy.
   Their own paper says averaging can erase the seam by blurring.
3. Only if 1 leaves copy-5-style disagreement on real outputs: give every
   tile the same coarse sample of the whole image, and let the tile add
   detail. The coarse sample is the shared draw. CASR's cached low-resolution
   features are this idea inside their cycle. Use it as a baseline, do not
   reimplement CASR.

Pipeline 3 is the paper's method only when pipeline 1 loses on the frozen
score and does not lose on a fidelity number agreed in advance (DISTS against
the input's structure, or the no-reference score already used by that
super-resolver). A gain on the new score bought by smoothing the image is a
failure.

### 4.5 What the paper then is

If pipeline 1 wins: the paper is the score, the injection test, the human
check, and a negative result on VOSR, CASR, and SeedVR under a matched tile
size. That is a complete paper.

If pipeline 1 loses and pipeline 3 wins on the frozen score without a fidelity
drop: the paper adds that method, with pipeline 1 and SWITi as baselines.
Ablations of pipeline 3 come after it wins, not before.

## 5. Not in this round

No literature review beyond the papers named above. No training run yet.
No change to the defocus benchmark. Section 6 is the metric formulation
under discussion. It is not agreed until the open choices at the end of
that section are closed.

## 6. Metric formulation, 2026-10-16

Raw instinct, restated. Cut a consistent photograph on a grid. Corrupt some
tiles, of several types, at several strengths. The label is how much
corruption was applied, and of which kind. Train a model to predict that
label from the corrupted image alone. The prediction is the consistency
score. The backbone may be dilated convolutions, attention, or anything
else. That choice is not the claim. A binary clean-versus-corrupt head is
the wrong thing to train if the score will also be a loss.

### What is being trained

The input is one image and the grid. The grid is known, because the person
running a tiled super-resolver chose the tile size. The clean original is
not an input. Neither is a mask of which tiles were edited. Giving the
model the mask would leak the label.

The target is a vector, one number per anomaly type, each in \[0, 1\]:

| Type | What is edited inside a tile | The number |
|---|---|---|
| Seam | A line on the boundary | Fraction of boundary length that carries a line, times its contrast |
| Tone | Gain or color, blended so there is no line | Fraction of tiles edited, times the size of the shift |
| Statistics | Grain or sharpness, no line | Fraction of tiles edited, times the size of the change |
| Content | A repeated texture replaced by a different texture of the same stuff, blended across the cut | Fraction of eligible tiles swapped |

A scalar severity is a weighted sum of those four. The weights are chosen
after the model exists, not before, and they are frozen before any
super-resolver is trained against the score. Until then, report the four
numbers. A single scalar hides a model that only learned seams.

Cutting the photograph into tiles is not the anomaly. An untouched tiling
of a real photo is the clean example, and its target vector is zero. The
anomaly is the edit.

### Why the head is not a binary classifier

A binary label says the image contains an anomaly or it does not. Eight
swapped tiles and one swapped tile get the same label. A super-resolver
trained to drive a binary score down gets almost no gradient once the
classifier is sure the image is inconsistent, and no gradient that prefers
a milder error over a worse one. Sigmoid outputs saturate.

Binary classification stays in the paper as a check, not as the metric.
Threshold the content component, or the severity, and measure whether clean
images and corrupted images separate (AUROC). That answers "can it see
anything." The trained output, and the loss, are the four continuous
numbers.

If the score is later added to a super-resolution loss, the term is the
severity, or the content component alone. It is not a cross-entropy.

### How a training pair is made

1. Take a clean photograph.
2. Sample a grid size from the sizes real super-resolvers use.
3. Sample a type, a fraction of tiles, and a strength. Most pairs corrupt
   one type, so the four targets are not entangled. A minority corrupt
   two types, so the model sees combinations.
4. Apply the edit. Record the four numbers from the recipe. That record
   is the label. It does not require a person.
5. Hold out whole photographs, hold out at least one grid size, and hold
   out one way of applying the content swap. A model that only ranks the
   swaps it was trained to paste has learned the paste.

### What "correct" means

On held-out photographs, for each type separately, the predicted number
must rise with the applied number. Spearman against the recipe above 0.8
is the same bar already written in section 4.3. A content edit must not
be explained by the seam head, and a seam edit must not be explained by
the content head. An unedited photograph, including one whose real edge
lies on the grid, must sit near zero on every head.

A learned score that passes only on its own training recipes is not yet
the metric. The same heads are then run on tiled outputs of existing
super-resolvers, where there is no recipe. Agreement with people on those
images is the test that the synthetic label transferred. That human check
is unchanged from section 4.3.

## 7. Definition of consistency, 2026-10-16

Superseded by section 8. The claim that fidelity to the original is a different
quantity was the wrong turn. The metric uses the ground truth.

## 8. Definition, corrected 2026-10-16

Pipeline. Y is the 8K ground truth. Two degradation rounds, each a factor of
2, produce x at about 2K. x is cut into about 16 tiles of 512 with overlap.
VOSR at scale 4 is run on each tile. The stitch is Ŷ, back at 8K. The grid
that matters is those 16 tiles. Overlap is removed before any comparison, so
each tile contributes only its interior.

A region R, such as a hat, is taken from Y. It may cross 7 or 8 tiles. A
tile is kept when its interior still contains enough pixels of R.

For each kept tile i, e_i is the reference error between the interior of Ŷ
and the interior of Y, on pixels in R only. The working choice is masked
MSE of the high-pass images, plus the L2 distance of their Gram matrices
from a fixed filter bank. Both terms use Y. Mean color in a lightness-chroma
space can be a third term. e_i is a vector of those terms.

The region is consistent when two technical conditions hold at once.

1. Level. Every e_i is small. Each piece of the hat in Ŷ matches the same
   piece in Y.
2. Dispersion. The variance of e_i across the kept tiles is small. The tiles
   are not faithful in different ways. One piece matching Y and another
   departing from Y is an inconsistent hat, even if the average error looks
   moderate.

The reported score for R is the pair (mean of e_i, variance of e_i). A
single scalar, if needed, is the mean plus the variance. Both parts use Y.
There is no score for this region that ignores Y.

This is not the spread of the tiles of Ŷ among themselves. Two tiles can
agree with each other and both disagree with Y. With Y required, that hat
is not consistent.

The overlap band is not in e_i. A half-period shift that appears only in
the removed band is outside this definition.


The metric uses the ground-truth 8K image. Consistency is a property of one
region that crosses several tiles, for example a hat cut into 7 or 8
interiors. It is not pixel agreement with the ground truth, and it is not
a judgment that the tiles look similar.

Let R be the set of pixels of that region, taken from the ground truth and
scaled onto the stitch. Overlap bands are excluded. A tile is kept only
when its interior still contains enough pixels of R.

Let phi be a translation-invariant texture statistic of those pixels. The
working choice is the mean in a color space with a separate lightness
channel, plus the Gram matrix of a fixed filter bank on the high-pass
image. Phi does not encode where on the hat the piece sits. Crown and brim
are allowed to differ in shape. They are not allowed to differ in material.

On one image, fit the tile means of phi inside R. The between-tile variance
is the variance of those means. The within-tile variance is the variance of
phi on subwindows inside a single tile, averaged over the kept tiles. The
tile effect is the ratio of between-tile variance to within-tile variance.
This is the F ratio of a one-way layout whose factor is the tile index.

Compute that ratio on the super-resolved image and on the ground truth,
using the same R and the same tiles. Consistency is the difference

tile effect of the super-resolved region, minus tile effect of the
ground-truth region.

Zero means the super-resolver added no tile structure beyond the variation
the hat already has. Positive means the tile index explains more of phi
in the result than it does in the ground truth. Negative is possible and
means the result is flatter across tiles than the real hat.

The null statement is: inside R, the conditional distribution of phi given
the tile index equals the distribution of phi given only R. The ground
truth is the estimate of how far a real hat departs from that null. The
score is the excess departure of the super-resolved hat.

Pixel distance to the ground truth is a different number. A hat that is the
wrong weave in every tile can score zero here and fail that distance. A hat
whose pieces are each sharp, and not the same material, can pass a
perceptual score and fail here.

The overlap band is not in this ratio. A stripe that is the right material
and shifted by half a period is invisible to phi and visible only in the
band that was removed. That join is a second statistic, computed only on
the discarded overlap, and it is not this definition.

### What is deliberately not decided

- The backbone. Dilated convolution and attention are allowed. They are
  chosen by which one ranks the held-out content swaps, not by taste.
- The four weights in the scalar, until the four heads have been looked at.
- Whether a non-learned descriptor from section 4.2 already ranks the
  content column. If it does, this trained model is unnecessary for that
  column. Build the trained model for the column the descriptor misses.
