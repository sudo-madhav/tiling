# Review of `sr_directions.md` — 2026-10-17

Branch `sr`. Read for this review: `plan/sr_directions.md`, `plan/sr_handoff.md`,
`plan/sr_meeting_2026-10-17.md`. The code in the repo was not used. Web sources
checked are listed at the end.

Notation: `Y` ground truth, `x` low-resolution input, `Ŷ` stitched output,
`Up(x)` bicubic-upsampled input. A region `p` is a patch or segment inside a
tile interior.

## 0. Bottom line

- Keep the two answers in `sr_directions.md`. The score reads only the output
  (plus a reference and the tile grid), and the ground truth is in it.
- Several supporting claims are wrong or unsupported (section 2). Fix them
  before anyone quotes them.
- Main gap: it treats "consistency with GT" and "consistency across all tile
  pairs" as two things to combine. They are one quantity. Comparing two tiles
  relative to GT is comparing their GT residuals (section 3).
- That also handles the meeting's hard case: two tiles equally far from GT,
  with different invented texture. A scalar per-tile error cannot see it. A
  residual vector in a texture space can. No VLM is needed to measure it.
- Zero-shot VLMs are a weak label source here. On generative SR preference,
  GPT-5.2 and Gemini 3 Pro agree with people 52.5% and 58.8% of the time,
  LPIPS 62.5%, people with each other 84.7% (RefReward-SR, Table 1). Use a VLM
  as an audited extra column, not as training labels.
- The proposed method (coarse anchor plus affinity-routed key/value sharing)
  is close to published work: DemoFusion and DiffuseHigh for the anchor,
  DC-VSR's Spatial Attention Propagation and CASR for the sharing. It is a
  contribution only if our score shows those fail.
- Recommendation: measurement first (closed-form, GT-anchored, with controls
  and a small human test), then a finding on current methods, then a method
  only if a measured gap remains. Either paper shape needs the score, so it is
  not wasted if we go method-heavy.

## 1. What `sr_directions.md` gets right

- The score reads only the output, the reference and the grid. Pipeline
  interventions are for diagnosis.
- The ground truth is in the score, and it decides which tile pairs should
  agree.
- A VLM must not be asked about unrelated pairs (sky against jacket).
- A VLM must see native-pixel crops, not the 8K frame.
- A full image mapped to one scalar, trained on 150 images, will not learn to
  compare tiles.
- If anything is trained, supervise pairs, not images.
- A consistency number alone can be won by blurring every tile. Report mean
  error next to it.
- Shared noise, global normalization, blending and colour fix come before any
  new method.

## 2. Corrections to `sr_directions.md`

A claim-to-evidence audit (paper-self-review skill). Each item gives the
claim, why it fails, and the fix.

1. **"Material similarity `w_ij` on `Up(x)` matches `Y` with >0.95
   correlation, as shown in RefReward-SR."**
   RefReward-SR has no such number. It is an MLLM reward model that uses the
   LR image as a semantic anchor.
   Fix: delete. Measure it on the probe set as the overlap of the
   same-material pair sets built from `Up(x)` and from `Y`.

2. **"As demonstrated in PandaBench, VLM encoders downsample to 448 or 768."**
   PandaBench shows MLLMs fail at region-level comparison even when given
   region names and boxes, and that some flip with input order. It does not
   study resizing. Resizing is an API property. GPT-4o "high" detail fits the
   image in 2048 px, then scales the short side to 768 px. Claude's standard
   tier caps the long edge at 1568 px, and its token budget limits a square
   crop to about 1,090 px. The high-resolution tier allows a 2576 px edge and
   about a 1,930 px square.
   Qwen2.5-VL uses a configurable `max_pixels`.
   Fix: cite each source for what it shows. Check the effective resolution per
   model before any labeling run.

3. **"Mathematical proof: a linear average of per-tile scores is blind to
   inter-tile variance."**
   Overstated, and it does not support the conclusion drawn from it. The
   narrow part holds: a plain average of per-tile scores is separable, so the
   gradient for one tile ignores the others. But a model that mean-pools
   per-tile features and applies a nonlinear head computes variance exactly
   (pool `s` and `s²`, as in DeepSets). Pooling is not blind to spread.
   The real limit is practical. Take a pairwise term gated by which tiles
   should match: "tile i and tile j disagree, and the GT says they are the
   same material". Sum pooling can represent it only with a latent size at
   least the number of tiles (Wagstaff et al., ICML 2019). Learning that from
   about 150 images is unlikely.
   Fix: give the model pairwise inputs (section 6.3).

4. **Pair term `w_ij · max(0, d(Ŷi,Ŷj) − d(Yi,Yj))`.**
   A difference of two scalar distances, the same form as TecoGAN's tLP. It is
   zero whenever the two outputs are as far apart as the two GT tiles, even if
   they are wrong in different directions. The hinge also ignores outputs that
   are more alike than the GT (one texture pasted over a real change).
   Fix: the residual difference, section 3.

5. **Whole tiles as the unit of comparison.**
   A tile at 8K holds several materials (sky, roof, tree). "Are tiles i and j
   consistent?" has no single answer.
   Fix: regions inside tile interiors. A pair is two regions in different
   tiles.

6. **"15 to 35 comparable pairs per image", "80 to 90 of 120 pairs are
   unrelated", "targeting >85% human–VLM agreement", "16 tiles, 120 pairs".**
   The first two are estimates with no measurement behind them. The 85% bar
   is arbitrary and probably unreachable: on SR preference, people agree with
   each other 84.7% (RefReward-SR). 16 tiles and 120 pairs is right for the
   handoff's 4×4 grid, but that grid has 2048 px output tiles. The meeting
   caps output tiles at about 1024 px. At 8K with 1/8 overlap that is about
   45 to 80 tiles, depending on frame shape and exact tile size (1,000 to
   3,200 pairs). 512 px output tiles give about 170 tiles (14,000 pairs).
   Fix: count on the probe set once the geometry is fixed (section 4). Report
   VLM–human agreement next to human–human agreement on the same items.

7. **VLM 0/1 labels as the training target, and "4,500 to 18,000 pair
   labels ... will converge cleanly without overfitting".**
   Zero-shot MLLMs agree with people 32 to 59% of the time on generative SR
   preference, against 84.7% between people, and often give ties (section
   6.2). Distilling them copies their bias and makes validation circular.
   The label count also overstates the data. Pairs from one image share
   tiles, content and method, so they are strongly correlated. The number of
   independent samples is closer to the number of images than to the number
   of pairs.
   Fix: the VLM is an audited column. Training labels come from the
   closed-form GT score and from interventions (section 6.3). Split train and
   test by image, never by pair.

8. **"Unary consistency = 1 − 4·Var(u_i)" on binary per-tile labels.**
   Uniform failure scores as perfectly consistent, and severity is lost.
   Fix: drop it. Uniform failure belongs in the mean error, as the handoff
   already says.

9. **Global coarse anchor: "one globally coherent restore decision".**
   The cost figure is fine: one extra pass, 1/16 of a 4×4 run. The decision
   claim is not. Downsampling `x` by 4 and running ×4 SR gives an image at the
   resolution of `x`, not 8K. In its pixels the blur is four times smaller
   than what the tiles see, and noise and JPEG blocks are averaged away. Its
   restore-or-keep decision is made on a different degradation.
   Fix: treat it as a hypothesis. Test whether the anchor's restore/keep map
   agrees with the sharpness map of `Y`.

10. **Affinity-routed exemplar key/value sharing as the method.**
    DC-VSR (arXiv 2502.03502) injects uniformly subsampled keys and values
    from all tiles into every tile's self-attention (Spatial Attention
    Propagation). Its example is two distant brick regions that mismatch
    without it. CASR caches global bottleneck features. StyleAligned shares
    attention for style. Routing by affinity instead of uniform sampling is a
    design choice.
    Fix: keep it only if the score shows uniform sharing fails where routing
    succeeds.

11. **"IQA reviewers demand ... (typically 500-1,000+ images, 20+ human
    raters) to validate a learned metric."**
    No source. It also sets the bar for a learned IQA model, which is not the
    first deliverable. The meeting judged a mean-opinion or ABX check enough
    for the claim we make: the score agrees with people where the failure is
    visible. Diffusion-4K (section 7.1) shipped closed-form metrics at CVPR
    with a human correlation check and a pairwise study.
    Fix: size the human test from the claim it supports (section 8.3, step
    4).

12. **Wording.** "Crush", "Undisputed", a leftover "Wait:", a "48-Hour Probe"
    heading, and "(Expected)" on an untested outcome.
    Fix: remove. Mark hypotheses as hypotheses.

## 3. One quantity: consistency with GT is consistency across pairs

This is the main point of the review.

### 3.1 Three identities

Residual of region `p` in a descriptor space `φ`:

`r_p = φ(Ŷ_p) − φ(Y_p)`

Identity 1. Judging two output regions against their two GT regions is
comparing residuals:

`(φ(Ŷ_p) − φ(Ŷ_q)) − (φ(Y_p) − φ(Y_q)) = r_p − r_q`

Identity 2. Summing over all pairs with equal weight gives the spread:

`sum over pairs |r_p − r_q|² = n² · var(r)`

Here `var(r)` is the mean squared distance of the residuals to their mean.
With one node per tile this is the handoff's "variance of r_i across the 16
tiles". The handoff's rule "do not score tiles against each other" was right
for raw comparisons. It is not broken here, because the comparison is
relative to `Y`.

Identity 3. Weight each pair by a GT affinity `w_pq`, high for the same
material and near zero otherwise:

`E = sum over pairs w_pq · |r_p − r_q|²`

This is the Dirichlet energy of the residual field on the GT affinity graph.

Reading:

- The mean of `|r_p|` is fidelity: how wrong.
- `E` on same-material pairs is consistency: how differently wrong.
- A method may be wrong. It has to be wrong the same way on the same
  material.

### 3.2 Why this catches the meeting's hard case

Two water regions sit in different tiles, with GT texture `T` in both. The
output invents `A` in one and `B` in the other, each equally far from `T`. The
scalar errors are equal, so the per-tile spread is zero. That is why the
meeting found the per-tile GT distance flat.

The vectors still differ: `r_p − r_q = φ(A) − φ(B)`. A vector residual sees
the disagreement. A scalar does not.

The handoff's texture coordinate is a distance, a magnitude. Its other
coordinates are written as magnitudes or are ambiguous about sign. Magnitudes
drop the direction. The change to make is signed, per-band coordinates.

### 3.3 Descriptors (signed, so the direction survives)

| Coordinate | Signed descriptor per region | Sign tells |
|---|---|---|
| Restoration | log of output/GT high-pass energy, per scale band | softer or sharper than GT |
| Texture | feature mean/variance or Gram difference, per layer | which statistics moved |
| Spectrum | power per orientation and scale band | finer, coarser, rotated |
| Tone | low-pass Lab difference | which way colour moved |
| Structure | gradient correlation with GT | scalar is enough |

Keep the handoff's normalization by the GT scale of each region, per
coordinate.

### 3.4 Definition and three closed-form estimators

Definition. A tiled method is tile-consistent when its residual depends on
content and not on which tile the region fell in. Given the content, the tile
ID tells you nothing about the error.

Estimators:

1. **Cross minus within.** Take same-material region pairs. Compare the pair
   error for pairs in different tiles with pairs inside one tile, matched on
   distance and GT similarity. With no tiling effect the two agree. The
   difference is the score, and a permutation test gives a p-value. This is
   SWITi's test (seam strip against nearby control strips) moved from pixel
   gradients at one seam to content residuals at any distance.
2. **Content-adjusted ICC.** Regress residuals on GT content features plus
   tile fixed effects. The share of residual variance explained by tile
   membership is the score.
3. **Grid scan.** Recompute 1 or 2 for shifted and rescaled candidate grids.
   The true grid should stand out, and a random grid gives the null. This also
   scores outputs whose grid is unknown, the way JPEG blockiness detectors find
   the 8×8 grid.

### 3.5 Limits to write into the paper

- Consistency is measurable only where the same material appears in more than
  one tile, or on both sides of a boundary. A face inside one tile has
  fidelity, not consistency. Curation follows from this: large surfaces and
  repeated structures that cross many tiles (water, foliage, facades, fabric,
  text).
- Within-tile pairs exist only up to the tile size. For longer distances, use
  the untiled run on small images, or fit pair error against distance on
  within-tile pairs and test the excess.
- A consistent method can still have content-dependent error, for example
  sharpening fine texture more than coarse. Weighting by GT similarity keeps
  that small: nearly identical GT content gives nearly identical residuals for
  a consistent method.
- One plot carries the story: excess pair error against pair distance.
  Blending should flatten the short range. Independent tiles should show at
  long range. SyncDiffusion's claim that overlap averaging couples only
  neighbours becomes a curve.

This needs no training, no VLM and no full-frame run. Inputs: `Ŷ`, `Y`, the
grid (or a scan), and a region affinity from `Y`.

### 3.6 Edits this implies for the handoff definition

- Regions inside tiles, not whole tiles.
- Signed or vector coordinates in `r_i`.
- Consistency is the energy on same-material pairs, cross minus within. It is
  not the plain variance over all tiles.
- Fake-grid, tile-count and grid-shift controls (section 4).
- Settle the tile geometry first (section 4).

## 4. Question 1: intervene in the pipeline, or score only the output

The question has two readings.

**Reading A: does the score need access to the method?** No. It takes the
output, the reference and the grid. It must score any method's output.
Repeated runs mix sampling noise with tiling, and multi-pass scoring is
expensive.

**Reading B: is our contribution a change to existing pipelines?** That is
the method question (section 7). It does not compete with an output-only
score. The score is how the change is judged.

Interventions belong in three places, all offline:

1. **Validation with a causal knob.** Same image, same method, only the tiling
   changes.
   - Tile count: untiled (small images), then 4, 16, 64 tiles. Consistency
     should worsen with tile count on most images. Fidelity should move less.
   - Half-tile grid shift: anomalies caused by tiling move with the grid.
     Anomalies caused by content stay.
   - Fake grid on an untiled output: the score should sit at the null.

   These checks are free, run on real outputs, and need no human. MEt3R (CVPR
   2025) validated its multi-view consistency metric the same way: a near-zero
   lower bound on real videos, and the ordering expected by construction (a
   model that generates one view at a time is least consistent). I found no
   large human study in it.
2. **Labels**, if anything is trained (section 6.3).
3. **Diagnosis** for the paper's motivation section.

A method-level number is also possible: the output change under a half-tile
shift, with noise defined in image coordinates so the shift does not change
the noise. It measures the method, not an image. Report it as a diagnostic
column, not as the score.

The handoff glosses over one point: our own tiler is an intervention.
Wrapping a method in an external tiler changes what each tile is conditioned
on. Methods that derive a prompt or tags from their input (OSEDiff and SeeSR
use DAPE tags, SUPIR uses a captioner) get a different prompt per tile.
Anything computed from the crop (semantic features, colour statistics,
degradation estimates) diverges the same way. A method's own tiling often
computes these once per image. External tiling can therefore create
inconsistency that users of the method never see.

- Report the method's own tiling as the headline and controlled tiling as the
  diagnostic.
- The first cheap fix to test: compute conditioning once on the whole image.
  Sharing one global prompt unchanged has its own failure (AccDiffusion: every
  patch grows the captioned object), so restrict it per tile to what the tile
  contains.

Tile geometry has to be settled before any number is computed. The handoff
runs VOSR on 512 px low-resolution tiles (2048 px out). The same handoff says
VOSR's own denoising crop is smaller than 512 px on the low-resolution side.
The meeting says no model we care about emits more than about 1024 px per
tile. These do not fit together. Either VOSR cuts each 512 tile again, and
the real grid is the inner one. Or the inner grid is disabled, and the model
runs above its native crop size and above the meeting's memory limit. In the
second case, failures from running at an unseen size mix with the tiling
signal. Score on the grid where independent forward passes actually happen.

## 5. Question 2: ground truth in or out

There are three reference levels, not two.

| Level | Reference | Role |
|---|---|---|
| Full | `Y` | benchmark, validation |
| Degraded | `x` | deployment, always available |
| None | nothing | weakest |

"No-reference" for SR should mean "no GT", not "no input". This is
degraded-reference IQA (CKDN, ICCV 2021): the input is the reference, and
pristine images are used only in training. TecoGAN did the same over time.
tLP and tOF compare the output's change between frames with the GT's change,
and without GT they compare with the input's change.

A pure no-reference score is weakest for a specific reason. It has to decide
which regions should match from `Ŷ` itself. If the method made two patches of
one material look different, the affinity says "different material" and the
failure hides itself.

The useful split:

- **Consistency needs a reliable affinity, not the truth.** If the same
  surface is sharpened in one tile and left soft in the next, the gain
  relative to `Up(x)` differs between same-material regions. `x` is enough to
  see that the decision was inconsistent.
- **Correctness needs `Y`.** Only `Y` says whether the surface was sharp or
  bokeh in the original.
- With `x` only, legitimate texture change that `x` cannot resolve
  (perspective, lighting) looks like disagreement at fine scales. The
  within-tile control removes it, because natural change affects within-tile
  and cross-tile pairs alike at matched distance. The LR version is valid only
  with that control.

Answer: build the score GT-anchored as the reference implementation, and the
LR-anchored version beside it for deployment. Report how closely they agree
on the benchmark. Nothing is assumed about that agreement; assuming it is
where the false 0.95 claim came from.

One caution on `Y`: generative outputs can beat the GT in places (A-FINE,
CVPR 2025; RQI, CVPR 2026). That argues for measuring consistency on residual
differences, which ignore a shared bias, and for keeping fidelity a separate
number.

## 6. Path A made concrete: VLM pair labels and a trained predictor

### 6.1 The triplet is right; the unit and the question need changes

Showing two tiles plus the GT is the right idea. It is TecoGAN's idea in VLM
form: judge the change between two outputs against the change between the two
references. Three changes make it measurable.

1. **Region pairs, not tile pairs** (section 2, item 5).
2. **Same-material pairs only, chosen from `Y`, split into cross-tile and
   within-tile at matched distance.** The within-tile pairs are the control.
   Without them, the fraction measures how often a VLM thinks two places look
   different, not tiling. The image score becomes cross minus within, not a
   raw fraction.
3. **A relative question.** Show a 2×2 panel: GT p and GT q above, output p
   and output q below. Ask whether the output pair differs more than the GT
   pair in sharpness, texture or colour. Better still is a two-alternative
   forced choice between two outputs of the same image (two methods, or two
   tile counts): which renders these two marked regions more consistently?
   People and MLLMs compare more reliably than they rate (DepictQA, ECCV 2024;
   Compare2Score, NeurIPS 2024).

### 6.2 Why the VLM cannot be the label source yet

- RefReward-SR, Table 1 (generative SR preference, LR as anchor, in-domain):
  people agree with each other 84.7%, LPIPS 62.5%, MUSIQ 60.9%, Gemini 3 Pro
  58.8%, GPT-5.2 52.5%, Qwen3-VL 8B 31.8%. The authors note that zero-shot
  MLLMs are insensitive to subtle detail and often give identical scores,
  which their protocol counts against them. A reliable judge needed
  fine-tuning on 18,000 human rankings.
- PandaBench: region-level comparison fails even with region names and boxes,
  and some models flip with input order.
- SemanticSRJudge (SIGGRAPH poster): VLM judges miss few-pixel failures at
  full-frame scale, so it sends native 128 px crops to the judge.

Tile consistency is a subtler judgment than those preferences.

Protocol if the column is used:

- Native crops no larger than the model's no-resize size (768 px short side
  for GPT-4o "high"; check per model).
- Both presentation orders, repeated calls, and log-probabilities where the
  API returns them (expected score, as in Q-Align).
- A human-labeled subset of the same items, with VLM–human agreement reported
  next to human–human agreement.
- Never train on the VLM labels unless that agreement is close to
  human–human.

### 6.3 If a predictor is trained: how to make it look at relations between tiles

The question "how do we make the model focus on inter-tiling given the full
image" has a direct answer. Do not give it the full image, and train it on
data where only the tiling changes.

1. **Inputs are relations.** Per region, the residual (GT version) or the gain
   relative to `Up(x)` (LR version). Per pair, the residual difference and the
   reference similarity. A model that only sees differences cannot score
   content. The full image supplies only the grid and the affinity graph.
2. **Labels that cost nothing and do not come from a VLM.**
   - Distillation. The closed-form GT score per pair is the target for an
     LR-anchored student. CKDN distills pristine-image knowledge into a
     degraded-reference model. CrossScore learns an SSIM map for a novel view
     without its ground truth, from other views. Labels are dense: every
     same-material pair in every image.
   - Provenance. Same tile or different tile, for same-material pairs matched
     on distance. If a classifier beats chance, it is measuring tiling. This
     is Huh et al. (ECCV 2018): a Siamese patch-pair model trained on free
     metadata, run on up to 625 patches per image, and pooled by mean shift
     into a consistency map. It is also a classifier two-sample test
     (Lopez-Paz and Oquab, ICLR 2017).
   - Ranking under intervention. Same image and method, more tiles against
     fewer (RankIQA, ICCV 2017). Caveat: smaller tiles also lower quality.
     Match fidelity, or give the model a separate fidelity head, or it learns
     "blurrier means more tiles".
3. **Architecture.** A graph over regions, supervised per edge, with edges
   pooled into the image score. The tile-graph network in `sr_directions.md`
   is fine in spirit. Change its labels and its pair input (residual
   difference, not a difference of distances).
4. **Validation.** Leave one method out, leave one grid size out, and run the
   human test in section 8. If it only works on methods it saw, it learned the
   methods.

Training is not worth it if the closed-form LR score already agrees with the
GT score and with people. A network then adds nothing for the paper.

## 7. Path B made concrete: a GT-referenced score, and a method that improves it

### 7.1 The "notional" metric can be the section 3 score

- **Variation from baseline.** Use a tiling-invariant baseline as the
  content-difficulty reference. Bicubic is deterministic and local, so tiling
  does not change it (nor a regression SR model run with enough overlap).
  Where bicubic fails, the region is hard. Gain over bicubic per region,
  compared across same-material regions in different tiles, is a clean
  restoration coordinate.
- **Number of anomalies.** Count regions whose residual departs from
  same-material regions in other tiles by more than k robust deviations
  (median and MAD of cross-tile pair errors). Report count and severity.
  Emphasizing sparse worst cases has precedent. PSNR99 (arXiv 2509.26398) and
  HalluGen's SHAFE (CVPR 2026) both show that averaging dilutes sparse
  failures, and HalluGen finds LPIPS and DISTS barely above chance at
  detecting hallucinations.
- **Per-tile PSNR/LPIPS spread** (the meeting's first experiment) stays as the
  naive baseline the score has to beat.

A closed-form metric shipped next to a method is accepted at CVPR when it is
checked against people. Diffusion-4K (CVPR 2025) introduced GLCM Score and
Compression Ratio and reported correlation with human evaluation (SRCC 0.75
for GLCM, 0.36 for MUSIQ), plus a pairwise preference study with people and
GPT-4o. That is the bar.

### 7.2 Rules so "maximize the metric" is not training to the test

- Freeze the metric (descriptor, regions, affinity threshold, aggregation)
  before method work. Commit the definition with a date.
- Develop on one image set, report on another.
- Always report fidelity (mean residual, PSNR, LPIPS, DISTS) and a
  no-reference quality score next to consistency. Consistency up with fidelity
  or sharpness down is a loss. TecoGAN made this point: warping-error metrics
  reward blurry bicubic output, so they paired temporal metrics with LPIPS.
- Do not train on the evaluation descriptor. If a consistency loss is used,
  use a different feature space.
- Run a human two-alternative test on the final comparison.

### 7.3 The method: what is taken, what may be open

Taken. By the novelty gate, a generic cross-tile consistency method is case 2:

- Global context or a coarse anchor: DemoFusion, DiffuseHigh,
  ElasticDiffusion, PatchFusion, CASR's cached global features.
- Cross-tile attention: DC-VSR's SAP, CASR, StyleAligned-type shared
  attention, video cross-frame attention.
- Far-window coupling: SyncDiffusion.

Possibly open. Each item is a hypothesis to test with the frozen score,
cheapest first. The first is the deliberately conservative baseline the
research-ideation skill asks for.

1. **Per-tile conditioning divergence** (section 4). If inconsistency drops
   when prompts, tags, features and colour statistics are computed once per
   image, the fix is trivial. That is a finding about external tiling, not a
   method.
2. **Restore-or-keep regime switching** (the handoff's restoration
   coordinate). A soft surface is kept soft in one tile and sharpened in the
   next. The open question is how to make that decision once per region
   instead of once per tile, for example with a decision map computed from the
   whole input and given to every tile. That differs from injecting a coarse
   image.
3. **Long-range texture disagreement on one material** (the meeting's second
   target). DC-VSR's uniform sampling is the baseline. Whether material-routed
   sharing beats it shows in the distance curve of section 3.5.

Pick the method after the probe shows which of the three carries most of the
measured inconsistency on current methods. Picking it now is guessing the
gradient (research-experiment skill).

## 8. Recommendation and order of work

### 8.1 Novelty gate (research-ideation skill)

| Candidate | Closest prior work | Case | Verdict |
|---|---|---|---|
| Seam score | JPEG blockiness, SWITi | 1 | no |
| Cross-tile content score for SR | none found in SR; tLP/tOF (video), MEt3R (views) | 3 | go |
| Generic cross-tile method | DemoFusion, SyncDiffusion, CASR, DC-VSR | 2 | only if the score shows they fail |
| Region-level restore/keep decision | not searched yet | ? | search before building |
| Predictor trained on VLM labels | RefReward-SR, Compare2Score, DepictQA | 2 | not first |

### 8.2 Paper shape

1. Measurement: the residual-difference definition, its GT and LR versions,
   its controls, and a human test showing it agrees with people better than
   per-tile LPIPS spread, Intra-LPIPS, SWITi and global metrics.
2. Finding: how much tiling inconsistency current methods have under their own
   tiling, which coordinate carries it, and whether existing fixes remove it.
3. Method: only if step 2 leaves a measured gap. Otherwise the paper is steps
   1 and 2.

This is neither the pure Path A of `sr_directions.md` (no learned predictor
first, no VLM as truth) nor a "notional" score: the section 3 score is defined
from first principles and validated. Path B needs it anyway.

### 8.3 Order of work, each step with a pass rule

0. Fix the tile geometry per method: the grid where independent passes happen.
   Save tile IDs and interiors.
   Pass: every output has a known grid.
1. The meeting's probe: global against per-tile PSNR and LPIPS on 10 to 20
   images.
   Pass: per-tile spread is higher on images we flag as inconsistent. If not,
   the naive instrument fails, which is also a result.
2. The section 3 score on the same outputs: scalar spread, vector energy,
   cross minus within.
   Pass: the vector score ranks flagged above unflagged images. Either way we
   learn whether the handoff's scalar definition is enough.
3. Controls.
   Pass: the fake grid sits at the null, the score worsens with tile count on
   most images, and peaks move with a half-tile shift.
4. A small human two-alternative test.
   Pass: the score agrees with the majority more often than per-tile LPIPS
   spread and SWITi.
5. Cheap fixes: shared conditioning, larger tiles, shared noise, global colour
   and normalization, and one cross-tile attention baseline.
   Decides Path A against Path B (section 7.3).
6. The VLM column on the same items.
   Reported with its human agreement, and dropped if near chance.

## 9. Open questions for the next meeting

1. Method-native tiling as the headline, controlled tiling as the diagnostic?
2. Regions inside tiles as the unit?
3. Signed or vector coordinates in `r_i`?
4. Which methods? At least one near-deterministic one-step model and one
   multi-step stochastic model, since independent draws matter more in the
   second.
5. Size of the human test, and who rates.
6. Is "no new method" acceptable for CVPR 2027 if step 5 shows existing fixes
   work?

## Sources checked

- RefReward-SR, arXiv 2603.24198. Table 1 numbers above.
- PandaBench (distortion graph), arXiv 2604.11004. Region-level failure, order
  dependence.
- RQI, CVPR 2026. SR metrics against people; relative quality; imperfect GT.
- A-FINE, CVPR 2025. FR-IQA without a perfect reference; SRIQA-Bench.
- CKDN, ICCV 2021. Degraded-reference IQA.
- TecoGAN, ACM TOG 2020. tLP and tOF; input-referenced versions without GT;
  warping error fooled by blur.
- DC-VSR, arXiv 2502.03502. Cross-tile attention (SAP); spatial consistency
  shown only in figures; SAP ablation with MUSIQ and DOVER.
- Diffusion-4K, CVPR 2025. Closed-form detail metrics checked against human
  ratings.
- MEt3R, CVPR 2025. Consistency metric validated with a real-data lower bound
  and expected orderings.
- Huh et al., ECCV 2018. Patch-pair self-consistency and consistency maps.
- Wagstaff et al., ICML 2019. Sum-pooled set models need a latent size at
  least the set size to represent arbitrary set functions.
- DepictQA, ECCV 2024; Compare2Score, NeurIPS 2024. MLLM comparison with and
  without a reference.
- CrossScore, arXiv 2404.14409. A full-reference metric map predicted without
  the aligned reference.
- SemanticSRJudge, SIGGRAPH poster (ACM DL 10.1145/3799825.3818778). Native
  crops for VLM judges.
- HalluGen, CVPR 2026; PSNR99, arXiv 2509.26398. Averaging dilutes sparse
  failures.
- SWITi, arXiv 2607.18990. Seam test against local controls (FRT, ASV).
- OpenAI, Anthropic and Qwen2.5-VL documentation on image resizing.

Searches found no published number for cross-tile content disagreement in SR.
CASR and DC-VSR both show it only in figures. The searches are not
exhaustive; repeat them before submission.
