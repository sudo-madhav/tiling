# WCSI-2

`wcsi2.py` replaces WCSI v1 (`wcsi.py`, no reference) and GMCI (`with_ref_genet/gmci.py`,
with reference) for scoring tiled super-resolution. One per-tile description of the band
the SR model invents is used by both scores. Without a reference it is compared with what
natural images do, learned from ground-truth / LR pairs. With a reference it is compared
with the ground-truth tile.

```bash
python genet/wcsi2.py noref SR.png LR.png --scale 8 --tile-sr 512 [--device cuda:0] [--advice] [--json out.json]
python genet/wcsi2.py ref SR.png GT.png LR.png --scale 8 --tile-sr 512 [--device cuda:0] [--advice]
python genet/wcsi2.py calibrate --gt GT_DIR --lr LR_DIR --scale 8 --tile-sr 512 --out genet/calib/wcsi2_x8_t512.json
python genet/natural_check.py --gt-dir GT --lr-dir LR --scale 8 --tile-sr 256 --first 50 --count 50 --out check.csv
pytest -q genet/test_wcsi2.py
```

```python
from wcsi2 import wcsi2_score, wcsi2_ref_score
d = wcsi2_score(sr, lr, scale=8, tile_sr=512, calib="auto", device="cuda:0", return_details=True)
d = wcsi2_ref_score(sr, gt, lr, scale=8, tile_sr=512, device="cuda:0", return_details=True)
# d["wcsi2"] / d["wcsi2_ref"], d["terms"], d["tiles"][k]["z" | "advice" | "phenotype"], d["notes"]
```

## 1. Why v1 had to change

On DIV8K, v1 gave VOSR 11, the ground truth 20 and CASR 25: a method scored above the
photograph. The held-out check below reproduces this on DIV2K. At 8×, v1 scores bicubic 61.6
and the ground truth 17.3, and the ground truth never beats bicubic. The causes, each
confirmed on those images:

1. **Zero-difference targets between tiles.** `E_cyto`, `E_PCP` and `E_sheaf` penalise any
   difference in invented texture between tiles that look like the same material in the
   LR. Natural photographs have such differences: grass in sun and in shade, near and far
   foliage. An output that invents little, and invents it uniformly, scores best. Bicubic
   invents nothing and is optimal. CASR, smoother and more uniform than the ground truth,
   beats it. VOSR invents a lot and non-uniformly, and loses.
2. **No term for missing detail.** Visibility weights came from the SR itself, so a blurrier
   SR switches its own spectral terms off. `kappa="auto"` divides out a global blur. Nothing
   measured whether the right amount of detail was invented at all.
3. **`E_RG` compared the whole LR band.** It coarse-grained the SR with WCSI's own kernel
   and compared it with an LR made by a different downsampler (successive PIL bicubic ×2).
   The kernel mismatch alone costs the ground truth about 0.2 JND per tile.
4. **Fixed thresholds.** JNDs did not account for how much natural tiles vary, nor for the
   sampling error of a periodogram on a small tile, nor for the texture of the image.
5. **Seam detector.** It took the median of the raw extra step over the rows of a seam. A
   2 px misregistration only shows in textured rows, so the median missed it (the `seam`
   variant scored 96.9 in a first version of v2 with the same detector).

GMCI kept the zero-difference targets (cause 1) and added a per-pixel 14-D state with dense
eigendecompositions: 5–6 minutes per 33 MP image.

## 2. Design

### The per-tile vector

Each image is split into a coarse part `M = D_s(img)` (Gaussian σ = 0.4 s, then s × s block
mean) and a residual `R = img − U_s(M)` (Keys bicubic, a = −0.5). Everything WCSI-2 says
about invented detail comes from `R`. That is the band [1/(2s), 1/2] cycles/px above the
LR Nyquist limit, the only band an SR model is free to invent.

For each analysis tile (the SR model's tile grid), from a Hann-windowed periodogram of the
tile of `R` in CIELAB:

| coordinates | count at 8× (4×) | what |
|---|---|---|
| `lum_o{o}_{a}` | 12 (8) | log luminance energy per octave (o = 0 just above the LR Nyquist limit, up to 1/2) and orientation (0, 45, 90, 135°) |
| `Q_xx`, `Q_xy` | 2 | orientation tensor of the band (from spectrum moments), 0 for isotropic texture, magnitude 1 for one orientation |
| `chroma_o{o}` | 3 (2) | log chroma energy (a\* + b\*) per octave |
| `peak` | 1 | log of max / mean power in the top octave: ringing, checkerboards, aliasing |

Octaves are `round(log2 s)`. Energies get a floor of (0.75 ΔE)² so invisible bands do not
dominate the logs. Each coordinate has a sampling standard error from the number of
effective periodogram ordinates in its band (Hann window ENBW = 1.5 bins per axis). Log
residuals saturate as 4 tanh(e / 4): a tile at 2 % of its target detail and one at 5 %
are both "no detail", not a nat apart.

### No reference

The target for tile i is `t_i = a + B x_i`, where `x_i` is the same vector computed on the LR
tile one scale down (the LR's own residual against its s-times coarse-graining). `(a, B)` is a
ridge regression fitted on ground-truth / LR pairs, NIQE-style. This is the scale
self-similarity of natural images, learned rather than assumed. Tolerance per coordinate:

    tol² = σ_nat² + JND² + se_SR² + B² se_LR²

with `σ_nat²` the natural residual spread of the fit (sampling noise subtracted). Then, with
`z = e / tol` and the residual `e = y − t`:

| term | definition | catches |
|---|---|---|
| `E_detail` | max(0, mean over tiles of mean(z²) − 1) | missing detail (bicubic, blur), excess detail (oversharpening), false colour, periodic artefacts |
| `E_consist` | max(0, Σ_pairs w (mean(Δz²) − 1) / (Σ w + 1)), Δz the difference of two tiles' residuals in units of the natural pair spread | tiles of one material departing *differently* (tile-to-tile hallucination). A departure shared by all tiles is in `E_detail`, not here |
| `E_low` | mean CIEDE2000² between low-passed (σ = 1 LR px) `D_s(SR)` and low-passed LR, / 1.5², minus the ground truth's calibrated floor | colour and tone drift |
| `E_seam` | per neighbour pair, the largest per-row log ratio of the jump at a seam line to the jump at control lines ±3 and ±6 px, minus the LR's own ratio at that line, squared in units of 0.25, minus the calibrated floor | seams, misregistration, per-tile colour offsets |

Pair weights `w` are material affinity, from the LR alone: Bures distance of the tiles' Lab
Gaussians (σ = 5) and the distance of their LR band energies (τ = 0.69). The SR cannot
change the graph it is judged by. Every term counts only the excess over what natural
images score, so the ground truth is the optimum in expectation.
`score = 100 / (1 + Σ w_k E_k / Σ w_k)`: 100 when every term is zero, 50 at one unit on
average.

### With reference

Same vector and tile graph, target = the ground-truth tile's vector, tolerance =
JND² + se_SR² + se_GT². Affinity comes from the ground truth. Terms:

| term | definition |
|---|---|
| `E_tone` | low-band colour of `D_s(SR)` against `D_s(GT)` (CIEDE2000², / 1.5²) |
| `E_colour` | sliced Wasserstein-2² (16 directions) between the tile's SR and GT CIELAB pixel clouds |
| `E_struct` | Σ γ \|∇S − ∇G\|² / Σ γ (\|∇S\|² + \|∇G\|² + c), / 0.05; γ = phase congruency of the GT (log-Gabor, 4 scales × 4 orientations, median-floored per tile), computed with 16 px of context. Position is enforced only where the GT is phase-locked |
| `E_detail` | mean over tiles of mean(z²) (no subtraction: zero for the GT itself) |
| `E_consist` | affinity-weighted mean of mean(Δz²) |
| `E_seam` | seam statistic of the SR minus that of the GT, clipped at zero |

Also returned: `rho_coherence = 1 − Σ w |z_i − z_j|² / (2 Σ w (|z_i|² + |z_j|²))`. It is 1 for
one shared departure, about 1/2 for independent departures and 0 for opposite ones. Each
tile also gets a label from (master = detail + tone + colour + struct, consist), each
compared with 1:

| label | master | consist |
|---|---|---|
| faithful | < 1 | < 1 |
| faithful among inconsistent peers | < 1 | ≥ 1 |
| consistent departure | ≥ 1 | < 1 |
| inconsistent departure | ≥ 1 | ≥ 1 |

With reference the score is `100 / (1 + F / Σw)` like the no-reference one.

### Advice

Per tile, `d["tiles"][k]["advice"]` names the largest residual in words. Examples: "too
little invented detail in the finest octave (horizontal strokes), 3.1 tolerances", "false
colour / chroma noise", "periodic artefact in the finest octave", "coarse colour drifts
from the target (b\* +1.8)". `--advice` prints it.

## 3. Calibration

`genet/calib/wcsi2_x{4,8}_t{256,512}.json` were fitted on DIV2K validation images 0801–0850
(ground truth cropped to a multiple of s). The LR is successive ×2 PIL bicubic downsamples,
the recipe the DIV8K LR was described with. `calib="auto"` picks the file for the scale
whose tile is closest to the scored tile.

| file | tiles | low floor | seam floor | R² lum octaves (coarse → fine) | R² chroma | R² Q | R² peak |
|---|---|---|---|---|---|---|---|
| x4 t256 | 2790 | 0.139 | 0.230 | 0.89, 0.69 | 0.91, 0.68 | 0.77, 0.71 | 0.54 |
| x4 t512 | 810 | 0.139 | 0.068 | 0.90, 0.71 | 0.92, 0.70 | 0.81, 0.75 | 0.55 |
| x8 t256 | 2790 | 0.226 | 0.242 | 0.88, 0.72, 0.51 | 0.92, 0.74, 0.48 | 0.63, 0.57 | 0.39 |
| x8 t512 | 810 | 0.225 | 0.128 | 0.90, 0.72, 0.50 | 0.94, 0.76, 0.49 | 0.63, 0.57 | 0.43 |

The finest octave at 8× is the least predictable from the LR (R² ≈ 0.5). Its natural spread
is correspondingly wide, so the score tolerates a range of fine-detail amounts there.

**Refit before the paper numbers.** The DIV8K LR's exact downsampler and the analysis tiles
of the experiments (1024, 512, 256 SR px) should be matched. Use about 50 DIV8K training
images that are not scored:

```bash
python genet/wcsi2.py calibrate --gt DIV8K_CAL_GT --lr DIV8K_CAL_LR_x8 --scale 8 --tile-sr 512 \
    --limit 50 --device cuda:0 --note "DIV8K train 50 imgs, LR = 3x PIL bicubic /2" \
    --out genet/calib/wcsi2_x8_t512.json
```

A mismatched calibration is reported in `d["notes"]`. Without any calibration file, the
score falls back to `t = x` with nominal spreads and says so.

## 4. Validation: does the ground truth come first?

`natural_check.py` makes controlled stand-ins for SR outputs from held-out ground truth
(DIV2K 0851–0900, never used for calibration). Analysis tile 256 HR px, overlap tile/16.

| variant | what it is | should rank |
|---|---|---|
| gt | the ground truth | first |
| bicubic | bicubic of the LR | below (no detail) |
| bicubic_usm | bicubic + unsharp mask 1.5, σ 2 px | below (wrong detail) |
| resample | invented band shifted 3 px where the GT is not phase-locked | tie without reference (equally natural); below with reference |
| phase_random | invented band of each tile phase-randomised | below (right spectrum, no structure) |
| tile_gain | invented band × U(0.4, 1.6) per tile | below (tile-to-tile inconsistency) |
| clash | right-half tiles keep 30 % of the invented band | below |
| sharpen | invented band × 1.4 everywhere | below or near (consistent departure) |
| colour_jitter | ±1.75 b\* per tile, checkerboard | below |
| seam | odd tile columns pasted 2 px misregistered | below |

"GT wins / ties" is the share of images where the ground truth scores higher by more than
0.05 / within 0.05.

**8×, 50 images** (scores ↑):

| variant | v1 mean | GT wins / ties | WCSI-2 mean | GT wins / ties | WCSI-2 ref mean | GT wins / ties |
|---|---|---|---|---|---|---|
| gt | 17.3 | - | **94.1** | - | **100.0** | - |
| bicubic | **61.6** | 0% / 0% | 51.0 | 100% / 0% | 12.8 | 100% / 0% |
| bicubic_usm | 55.3 | 0% / 0% | 54.7 | 100% / 0% | 13.6 | 100% / 0% |
| resample | 16.3 | 16% / 6% | 94.3 | 18% / 46% | 87.7 | 100% / 0% |
| phase_random | 13.9 | 30% / 0% | 41.4 | 100% / 0% | 16.9 | 100% / 0% |
| tile_gain | 13.6 | 92% / 0% | 71.5 | 92% / 8% | 41.0 | 100% / 0% |
| clash | 16.0 | 64% / 2% | 67.0 | 94% / 6% | 28.7 | 100% / 0% |
| sharpen | 13.6 | 100% / 0% | 83.8 | 66% / 24% | 61.7 | 100% / 0% |
| colour_jitter | 12.7 | 100% / 0% | 39.9 | 100% / 0% | 41.3 | 100% / 0% |
| seam | 16.9 | 64% / 26% | 83.7 | 90% / 10% | 48.9 | 100% / 0% |

**4×, 50 images** (scores ↑):

| variant | v1 mean | GT wins / ties | WCSI-2 mean | GT wins / ties | WCSI-2 ref mean | GT wins / ties |
|---|---|---|---|---|---|---|
| gt | 21.1 | - | 96.9 | - | **100.0** | - |
| bicubic | **62.7** | 0% / 0% | 51.5 | 98% / 2% | 17.1 | 100% / 0% |
| bicubic_usm | 40.5 | 4% / 0% | 62.7 | 98% / 2% | 21.0 | 100% / 0% |
| resample | 20.9 | 26% / 10% | **97.0** | 18% / 56% | 88.7 | 100% / 0% |
| phase_random | 18.7 | 28% / 0% | 74.5 | 98% / 2% | 27.3 | 100% / 0% |
| tile_gain | 17.9 | 80% / 4% | 76.0 | 80% / 20% | 47.7 | 100% / 0% |
| clash | 20.4 | 54% / 0% | 68.8 | 86% / 14% | 34.8 | 100% / 0% |
| sharpen | 17.8 | 100% / 0% | 86.1 | 66% / 30% | 66.1 | 100% / 0% |
| colour_jitter | 14.8 | 100% / 0% | 40.4 | 100% / 0% | 40.9 | 100% / 0% |
| seam | 19.8 | 84% / 6% | 79.1 | 98% / 2% | 47.4 | 100% / 0% |

Mean seconds per score at 4× (3 MP): v1 1.5, WCSI-2 1.2, WCSI-2 ref 3.7.

What this shows. At both scales v1 ranks bicubic about three times above the ground truth (62
against 17–21): the failure reported on DIV8K. Without reference, WCSI-2 puts the ground truth
above every degraded variant on average. It beats bicubic, phase-randomised texture and colour
jitter on 98–100 % of images, and ties on the rest. The resampled texture ties with the
ground truth on average (94.3 vs 94.1 at 8×, 97.0 vs 96.9 at 4×), by design: nothing without
a reference can tell two natural textures apart (it scores above the ground truth on 26–36 %
of images and below on 18 %). Of the degraded variants, only uniform sharpening ever beats
the ground truth, on 4–10 % of images, where the change stays within the natural spread;
tile-gain, clash and seam variants tie at most. With reference, the ground truth wins on every image against every variant, and the
resampled texture is the closest (88–89), as intended for valid generative texture.

**GMCI against WCSI-2 with reference** (8×, DIV2K 0851–0853, tile 256 HR px; `natural_check.py
--metrics wcsi2_ref,gmci`):

| variant | WCSI-2 ref | GMCI |
|---|---|---|
| gt | 100.0 | 100.0 |
| resample | **85.8** | 0.0 |
| sharpen | **67.4** | 0.0 |
| tile_gain | **56.7** | 0.0 |
| bicubic | **14.0** | 0.0 |

GMCI's `100 exp(−F)` saturates at 0 for anything that is not the ground truth, a valid texture
resample included, so it cannot rank outputs. WCSI-2 orders them as intended. Time per score
on these 3 MP images: GMCI 12.8 s, WCSI-2 ref 2.9 s.

## 5. Speed and GPUs

Measured on 3 MP DIV2K images, one CPU process (numpy): about 1.1 s without and 3.4 s with
reference. v1 took 1.5 s. Cost is linear in pixels, so expect about 11 s and 35 s at 33 MP.
GMCI took 5–6 minutes.

`device="cuda:k"` (CLI `--device cuda:k`) runs the per-pixel work on that GPU with torch:
CIELAB, the coarse / residual split, batched tile spectra (32 tiles per FFT batch), phase
congruency and gradients. Per-tile tables, the seam statistic and the sliced Wasserstein stay
on the CPU (they are small). `device="torch:cpu"` runs the same torch code on the CPU, and
`test_wcsi2.py` checks it against numpy. For many images, `genet_ablation/score_sweep.py
--metric wcsi2_ref --devices cuda:0,cuda:1,...` starts one worker per entry (`cuda:0*2` for
two per GPU when PNG decoding is the bottleneck). The GPU path has not been timed here
(no GPU in the outside environment).

## 6. Defaults

All in `wcsi2.DEFAULTS`, overridable as keyword arguments:

| parameter | value | meaning |
|---|---|---|
| `sigma_ds` | 0.4 | D_s Gaussian σ in units of s HR px |
| `sigma_low` | 1.0 | low-pass before the coarse colour comparison, LR px |
| `floor_de` | 0.75 | band RMS (ΔE) below which a band is invisible |
| `jnd_de` | 1.5 | CIEDE2000 |
| `jnd_loge` | 0.25 | log band energy (about 13 % contrast) |
| `jnd_q` | 2 sin 4° | 4° turn of a fully oriented texture |
| `jnd_peak` | 0.5 | log peak ratio |
| `jnd_struct` | 0.05 | normalised gradient error |
| `jnd_seam` | 0.25 | seam log ratio |
| `sat_loge` | 4.0 | saturation of log-energy residuals, nats |
| `sigma_cyto`, `tau_band` | 5.0, 0.69 | material affinity bandwidths (Lab Bures, log band energy) |
| `prior_pairs` | 1 | pseudo-pairs of zero excess in pair means |
| `neighbour_weight` | 0 | extra weight for touching tiles regardless of material |
| `n_proj`, `colour_samples` | 16, 4096 | sliced Wasserstein (reference) |
| `struct_margin` | 16 | context for rigidity and gradients, HR px |
| `weights` | None | per-term weights; 1 each |

## 7. Open items

- Refit the calibration on DIV8K with the experiments' tiles (section 3) and rerun
  `natural_check.py` on held-out DIV8K images.
- Rerun the DIV8K sweep (`genet_ablation/sh/score_wcsi2.sh`) and check that the ground truth
  is above CASR and VOSR without reference.
- JNDs are nominal; the term weights are 1.
