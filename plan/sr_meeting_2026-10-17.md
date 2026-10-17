# Meeting notes — how we will start, 2026-10-17

Pointers from the discussion. Not a transcript. Where the two views
differ, both are kept. Current written definition remains
`plan/sr_handoff.md`. This note is what we agreed to do next, and what
we refused to pretend is solved.

## Setup

Real high-resolution images. A degradation pipeline makes the
low-resolution input. The original is the ground truth. Final resolution
can be anything, because inference is tiled. Tile size is at most 1024.
No current model we care about emits a tile larger than about 1024
without running out of memory.

We have input and ground-truth pairs. The model output may be
tile-inconsistent. The ground truth is the reference we can use while
designing the score. PSNR on the whole image is not assumed to catch
that. Whether it does is an experiment, not a belief.

## Test set

Curate images where tiling artifacts are common. Start small: on the
order of 10 to 20 images, not a benchmark. Add recent methods that
actually tile and are known to show the problem. One or two methods
first. Diffusion-based methods are the ones that tile in practice.

Many papers report numbers on a single tile. Some show a full image
only qualitatively. Our comparison is on the full stitched image, with
the tile grid known.

## First experiment, agreed

This is the feasible start. Both sides agreed it is worth running
before any new metric and before any new method.

1. Run the chosen methods on the curated set. Keep the full output and
   the ground truth.
2. Score the full image against the ground truth with a pixel metric
   and with LPIPS.
3. Score the same pair tile by tile, on tile interiors, overlap removed.
4. Compare the global number with the per-tile numbers.

What we hope to see: the global number stays flat and hides the
failure, while the per-tile errors peak on the tiles that are visibly
wrong. A consistent recovery should give a flat per-tile error. The
signal is the variation across tiles, or the number of peaks, not the
average.

This is a probe of existing metrics. It is not a new metric and not a
method. If the per-tile LPIPS or pixel error does not peak where the
image is visibly inconsistent, that result is useful too. Then those
metrics are the wrong instrument and we change them from that
observation.

## The disagreement that bounds the problem

Two different targets got mixed together.

**Ground truth as a proxy.** Each tile is compared with its own piece
of the original. Tiles that are restored to different degrees get
different errors. The variation of that error is the inconsistency
signal. This is doable. It is the first experiment above. A tile that
unbokehs and a tile that stays soft shows up here, because one matches
the original and the other does not.

**Cross-tile consistency of hallucinated detail.** Downscaling destroys
fine texture. Upscaling invents it. Two distant tiles can invent
different textures for the same material. Both can be equally far from
the ground truth, so a per-tile distance to the ground truth stays flat
and the failure is invisible. The hat and the water examples are this
case. Catching it means comparing those tiles with each other, not only
with the original. That needs some notion of which tiles should have
matched. A static pixel or feature distance does not know that.
Segmentation and depth were CASR's way of finding regions that should
match. A general vision model judging tile pairs was proposed as
another way. Both are much larger than the first experiment.

Agreed constraint: we do not have to solve the second target to start.
The extent of the problem is ours to set. The first target is the
start. The second is real, and it is not mandatory for the first paper
if we say so. Using the ground truth does not automatically fix
cross-tile hallucination. It only catches inconsistency that also
shows up as uneven fidelity.

## CASR and SyncDiffusion, not to be mixed up

CASR is a CVPR 2026 paper. They state the problem we care about:
blending across overlaps does not fix consistency when the tiles are
far apart, and in particular does not fix cross-style consistency.
They find homogeneous regions with segmentation, call them superpixels,
and use depth so those regions do not break on edges. Their main
tables still report ordinary image quality. Intra-LPIPS is not their
metric. Intra-LPIPS is SyncDiffusion's, and it was not their
contribution. SyncDiffusion is about generated panoramas, not
super-resolution.

## What a convincing score has to survive

If we claim a score catches tiling inconsistency, it has to agree with
people on images where the failure is visible. A mean-opinion or ABX
check is enough to support that claim. It does not have to be a
theoretical metric paper. A score that only renames PSNR or LPIPS is
not a contribution.

Building a learned scorer, or a proof-style metric, can consume the
whole paper. The restoration method is still the part we want time
for. A new score is justified as the way we substantiate the problem,
not as the only result.

Image restoration papers have not treated tile consistency as the
thing to measure. Generation papers have, for seams and for style
across windows. LPIPS, where it is used, is a distance between feature
maps of two patches. It is reference-based. It is not a tile-consistency
metric.

## Data, if we later train a score

Not the first week. Recorded so it is not reinvented.

- Hand-curated real images where tiling inconsistency is visible.
- Injected inconsistencies as extra training or validation data, not as
  the only data.
- Repeated texture placed at known locations, then degraded and
  restored, is a check: the tiles that should match are known. It does
  not replace real images.
- A general vision model can label pairs. For n tiles, the number of
  pairs is n(n-1)/2. Each call gets the ground truth, the two tiles of
  the output, and a yes or no. The image score is the fraction of pairs
  called inconsistent. A set of 100 to 200 images is the scale that was
  discussed.
- That label is one column in a table. It is not the metric. The model
  will be wrong some of the time. It must be checked. A sampled set of
  its reasons, read by people, is a small user study. Using it as the
  only evaluation, with no human check and no other score, is too weak.

Training a network to map one image to that fraction, with no
comparison inside the network, was judged unlikely to generalize. If a
model is trained, its architecture has to compare tiles on purpose.
That modelling is a project of its own. It is not "throw an image and
a score at a conv net."

## Two paper shapes, not chosen yet

**Metric-heavy.** The score is the contribution. The general-model
labels, or a user study, are support. Little or no new restoration
method. This can fill the paper by itself.

**Method-heavy.** The first experiment shows that per-tile error
against the ground truth peaks where restoration is inconsistent.
That variation is the quantitative evidence. A method is then designed
to flatten it. The general-model pair check can be an extra column.
It is not the only column.

The meeting leaned toward starting on the method-heavy path, with the
ground-truth proxy as the constrained problem. Cross-tile hallucination
stays open. It is not dropped. It is not the first build.

## Action

Curate a handful of images. Run one or two recent tiled methods. Save
full outputs and ground truth. Compare one global score with the same
score computed per tile. Look for peaks. Decide from that plot whether
existing reference metrics already carry the signal. Do not design a
network, a prompt, or a new loss before that plot exists.
