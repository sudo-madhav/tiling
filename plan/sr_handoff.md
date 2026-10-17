# Handoff — tile-consistency paper, 2026-10-17

Branch `sr`. This file is the useful residue of the discussion. The trail,
including rejected formulations, is `plan/sr_decision.md`. The defocus
deblurring paper is paused in `plan/gap.md`. It is not deleted.

Current commit series on `sr` ends at the commit that adds this file.
Repo: https://github.com/sudo-madhav/tiling

## 1. What the paper is now

A reference metric for tiled super-resolution, then a method only if the
obvious fixes still fail that metric. Not a new upscaler as the first
deliverable. Not a seam detector.

The failure worth a paper: tiles of one super-resolved image are processed
differently, so a region is restored in one tile and left soft in the next,
or the fine pattern matches the original in one tile and not in another.
Usual scores do not isolate that. PSNR and DISTS punish invented detail
even when every tile agrees, and they can forgive tiles that fail in
different ways if each stays near the input. MUSIQ and CLIP-IQA rise when
a tile is sharp, whether or not the next tile did something else.

Visible grid lines are a bad paper. Overlap, Gaussian blending, and a
shared VAE posterior mode already target them. VOSR ships posterior mode
plus Gaussian blend and calls that seam-free. SWITi (Carrara et al.,
arXiv:2607.18990, July 2026) already publishes two no-reference seam scores,
FRT and ASV. A second seam metric loses to that paper.

A super-resolver whose only claim is a higher score on a metric tuned on
the same runs is circular. The metric is frozen before any method is
trained against it.

## 2. Pipeline we will actually run

`Y` is an 8K ground truth.

Two Real-ESRGAN-style rounds, each a factor of 2 (blur, resize, noise,
JPEG), produce `x` at about 2K. Not three rounds.

Tile `x`, not the 8K image. Tile size 512, with overlap recorded. A 4 by 4
grid is about 16 tiles.

VOSR at scale 4 on each 512 tile. Each output tile is 2048. Overlap scales
by 4: a 32-pixel overlap on `x` is a 128-pixel band in the stitch. The
stitch is 8K again.

The index that the score uses is these 16 tiles. Save the unblended
interiors, not only the stitch.

VOSR's own denoising crop is smaller than 512 on the low-resolution side.
If VOSR cuts each of the 16 tiles again, that inner grid is a second index.
Disable it for the first runs, or score it separately. Do not mix the two
grids.

## 3. Current definition of the metric

This replaces every earlier definition in the discussion. In particular it
replaces "fidelity is a different quantity" and "compare tiles to each
other" and "score one object such as a hat."

Each tile interior, overlap removed, is paired with the same pixels of `Y`.
The target for tile `i` is `Y_i`. Tiles are not required to look like each
other. Sky and foliage differ in chroma, edges, and detail because `Y`
differs. That is content.

Residual `R_i = Ŷ_i - Y_i` on the interior. Content is in both images, so
it cancels. A detailed tile still has a larger raw residual than a flat
tile when the algorithm is equally good. Divide by the scale of that tile
in `Y`:

- restoration, texture, artifacts: high-pass energy of `Y_i`
- tone: chroma spread of `Y_i`
- structure: use one minus the correlation of gradients with `Y_i`, which
  is already scaled

`r_i` is that normalized vector.

| Coordinate | What a person sees | What `r_i` measures |
|---|---|---|
| Restoration | Some tiles unbokeh, deblur, or gain detail; some stay soft | Missed or extra high-pass energy, divided by the high-pass energy of `Y_i` |
| Structure | Edges hold in some tiles and move or vanish in others | One minus gradient correlation with `Y_i` |
| Texture | Fine pattern matches `Y` in some tiles and is a different pattern in others | Gram-matrix distance on a fixed filter bank, divided by the Gram energy of `Y_i` |
| Tone | A tile is the wrong brightness or color for its own content | Mean color difference divided by the chroma spread of `Y_i`. Zero when the tile has the chroma of its own `Y_i` |
| Artifacts | Ringing, false texture, or a leftover grid in some tiles | High-pass energy in `Ŷ_i` that is absent from `Y_i`, divided by the high-pass energy of `Y_i` |

The mean of `r_i` over the 16 tiles is the overall error. The variance of
`r_i` is the inconsistency. A tile that restores and a tile that stays soft
is variance on the restoration coordinate. Two tiles of different color are
consistent when each matches its own `Y` color.

There is no score that ignores `Y`. Without `Y` there is no target for what
that tile's content should be. A no-reference score is later, if this one
works. It is not the current metric.

The overlap band is not inside `r_i`. A half-period shift that exists only
in the removed band is a separate join statistic. It is not this score.

A binary clean-versus-corrupt label is the wrong training target if the
score is ever a loss. Eight bad tiles and one bad tile would share a label,
and a saturated classifier has no gradient toward a milder error. The
number that would enter a super-resolution loss is the variance, or one
coordinate of it. Binary accuracy can be reported later as a check. Do not
train it first.

## 4. What we tried that we are not using

- Scoring tiles against each other. Content is supposed to differ.
- A single-object story (a hat across 7 or 8 tiles) as the definition.
  The same coordinates apply to every tile. The hat was only an example of
  restoration firing in some tiles and not others.
- Excess high-frequency dependence on the tile index, with fidelity set
  aside. The ground truth is in the metric. Tiles that agree with each
  other and disagree with `Y` fail the mean.
- Treating the Real-ESRGAN degradation as irrelevant to the definition.
  It is the way `x` is built. The score is still `Ŷ` against `Y` on the
  interiors.
- A learned scorer trained on injected anomalies, as the first step. The
  first step is the normalized residual on real VOSR outputs. Injections
  come after that table looks right, one edit type at a time, to check that
  only the matching coordinate moves.
- Seam-only metrics (SWITi FRT/ASV, JPEG blockiness) as the paper.

## 5. Papers, and what to take from each

Already studied by us. Do not re-read for the meeting. Take only the point
named here.

**SWITi** (Carrara et al., arXiv:2607.18990). Microscopy, not
super-resolution. Two causes of a seam: the receptive field is larger than
the tile, and neighboring tiles are independent samples. Their stitch
averages shifted windows at the same cost as repeating one grid, for
posterior models only. On a deterministic network the slide costs extra
passes. FRT is the fraction of tiles whose seam gradients differ from the
gradients beside the seam. ASV is how strong that difference is. On their
strongest set, FRT falls from about 0.97 to about 0.50. The line gets
weaker. It does not disappear. A smooth image fools the test, so they
report it next to fidelity. The test needs the seam coordinate and only
sees the one-pixel jump. A different texture with a smooth cut is invisible
to it.

**SyncDiffusion** (Lee et al., NeurIPS 2023). The paper that says overlap
averaging only couples neighbors. During sampling every window is pulled
toward one anchor window by LPIPS or a style loss. Intra-LPIPS averages
LPIPS over all pairs of non-overlapping windows, neighbors and far windows
together. It assumes windows should match. A real scene often should not.
Useful as a far-window score for generation, not as our metric.

**DemoFusion** (Du et al., CVPR 2024). A local denoise path plus a dilated
sample of the whole latent, so a corner crop sees a coarse view of the
opposite corner. Their complaint about SyncDiffusion is repetition and bent
structure, not a leftover seam.

**ScaleCrafter** (He et al., ICLR 2024). Tiled VAE decoding gives each tile
its own GroupNorm mean and variance, so distant tiles shift color. One
mean and variance over all tiles removes that. This is the tone coordinate,
not texture.

**CASR** (Guo et al., CVPR 2026, arXiv:2602.22159). Figure 2 is the same
repeated object with a different texture in different patches. They run
self-attention over every patch bottleneck and still publish LPIPS, MUSIQ,
NIQE, and PI. They name the failure and do not publish a number whose only
job is that failure.

**PatchFusion** (Li, Bhat, Wonka, CVPR 2024). Depth, not super-resolution.
A coarse full-image prediction is fused with finer tiles. Consistency-aware
training is an L2 on overlapping tiles. We already studied it. The idea we
keep is the coarse image as the carrier of global structure.

Read, and what we kept.

**ElasticDiffusion** (Haji-Ali, Balakrishnan, Ordonez, CVPR 2024). One idea.
Patches carry detail. One reference image at the training resolution owns
the layout. Far patches do not talk to each other. They talk to that
reference. Same split as PatchFusion. Nothing else to mine.

**AccDiffusion** (Lin et al., ECCV 2024). Worth using. The same caption on
every patch makes every patch grow the same object. No caption, and the
detail collapses. They cut the caption with the low-resolution attention
map, so a word is given only to patches where it actually appeared. For
super-resolution the copied condition is not a caption. It is the tile's
own crop. The analogue is a descriptor from the whole low-resolution image,
masked to that tile, saying this region is the same surface as that one.
Separately, their dilated samples were denoised independently and the
layout broke. They exchange noise across those samples. Two jobs: tell each
tile what it contains, and do not update the global samples in isolation.

**DiffuseHigh** (Kim, Hwang, Zhang, Park). The clean reference for the
coarse image as a hard constraint. The finished low-resolution image owns
structure. The high-resolution pass must not redraw it. A wavelet split
keeps the high-resolution step on the detail. In our pipeline that
low-resolution image already exists. It does not have to be generated.

Skim only if a later method still repeats after the coarse image is locked.

**FouriScale** (Huang et al., ECCV 2024). Dilation enlarges the field and
still leaves repetition, because high-resolution features alias. A low-pass
in front of the dilated convolution is their fix.

**FreeScale** (Qiu et al., ICCV 2025). Extra high-frequency energy above the
training resolution becomes repeated pattern. They fuse scales by keeping
one frequency band from each. Neither paper tells two patches they are the
same surface.

**HiDiffusion** (Zhang et al., ECCV 2024). Not read in detail. Diagnosis
only: repetition sits in the deepest U-Net blocks, whose attention behaves
locally at high resolution. Skip until an experiment looks like that.

No rectified-flow paper we checked scores or fixes disagreement between
tiles. FlowSR's consistency is agreement along the sampling trajectory.

## 6. Work now

In order. Do not train a scorer. Do not modify VOSR except to disable an
inner tile grid if one is on.

1. One 8K image. Two factor-of-2 Real-ESRGAN degradations to about 2K.
   Sixteen tiles of 512, overlap recorded. VOSR scale 4. Save the 16
   unblended 2048 interiors and the stitch. Confirm there is no second grid
   inside each tile.
2. Compute `r_i` on the interiors. Print the 16 by 5 table.
3. Two checks. An image where some tiles restore and some stay soft must
   show variance on the restoration coordinate. An image where tiles differ
   only because the scene differs, such as sky beside foliage, must not
   show variance for that reason.
4. If step 3 holds, repeat on about 20 images. The score of an image is
   the mean and the variance of `r_i`.
5. Controlled edits on `Y`, one type at a time, on this same grid, only
   after the real VOSR tables look right. The matching coordinate moves.
   The others stay put.

## 7. Explicitly not in scope yet

A new super-resolution architecture. Training a network to predict
consistency. A no-reference metric. Ablations of a method. The defocus
benchmark, native Restormer finetune, and ReFIR/iRAG tile study. Those
remain written in `plan/gap.md` and `plan/secure_runbook.md` and are paused.
