# with_encoder ablation protocol

Written for whoever runs this on the GPU machine. Do not download weights or data. Do not
resize tiles to fit an encoder. Do not build the graph from the SR image. Every script is
resumable; rerun to fill in errors.

Setup:

```bash
cp with_encoder/configs/encoders.example.json with_encoder/configs/encoders.json   # gitignored
export WITH_ENCODER_WEIGHTS=/path/that/already/holds/the/weights
export WITH_ENCODER_OUT=/path/for/outputs
export GENET_ABLATION_CONFIG=genet_ablation/config.json   # same file as the WCSI ablation
bash with_encoder/sh/check_encoders.sh                     # every encoder loads, shapes printed
CALIB_GT=... CALIB_LR_8=... CALIB_LR_4=... bash with_encoder/sh/calibrate_noref.sh  # held-out split, see below
python -m pytest -q with_encoder                           # CPU tests
```

## Part 0. No-reference calibration

The no-reference score subtracts what natural images score (DESIGN.md D14, D15). Fit the
WCSI-2 pixel model and the feature model on ground-truth / LR pairs that are not scored later
(for example 50 DIV8K training images, LR made exactly like the evaluated LR), at the analysis
tile that will be scored. `sh/calibrate_noref.sh` writes `$WITH_ENCODER_OUT/calib/`; the
ablation and method launchers pick the feature files up from there. Report the per-layer `r2`
and `nu_pair` from the feature JSON.

## Part 1. Controlled corruptions (does each arm see what it should, and only that?)

`sh/make_corruptions.sh` writes, per DIV8K image and scale (8× and 4×, 100 images each,
analysis tile 512 SR px), the ground truth, the LR, a tile layout and eight variants.
Above-Nyquist means frequencies above the LR Nyquist \(1/(2s)\) cycles/px, the only band an SR
model is free to invent. Every corruption that is meant to be LR-consistent changes only
that band.

| kind | what | should the score move? |
|---|---|---|
| clean | ground truth | reference point |
| F1_texture_resample | above-Nyquist band shifted 3 px where GT rigidity < 0.3 | **no** (valid generative texture) |
| F2_colour_drift | smooth b\* ramp +3.5 → −3.5 across the image (CIEDE2000 is smaller) | yes: low band |
| F3_seam_step | odd tile columns pasted 2 px misregistered | yes: seams |
| F4_texture_clash | right-half tiles keep 30 % of the above-Nyquist band | yes: non-local |
| F5_oversmooth | bicubic of the LR | yes: fidelity |
| F6_coherent_sharpen | above-Nyquist band × 1.4 everywhere | small, and coherent: `rho_feature` high |
| F7_tile_colour_jitter | ±1.75 b\* per tile, checkerboard | yes: seams + low band |

`sh/ablate_synthetic.sh` scores every row with:

- the learned stack under all four gates (`pix`, `feat`, `none`, `all`);
- the weight-free Gabor stack (control), gate `pix`;
- baselines MSE and 1−SSIM, and LPIPS / DISTS with `--external-baselines` if they are
  installed and torchvision's VGG-16 checkpoint is already in the torch cache.

Every configuration in `terms.py` comes out of the same rows. `ablate.py summarize` writes
`summary.md` and `ablation.csv`:

- per arm and corruption, the median paired change from clean, divided by the arm's median
  change over the catch set (F2, F3, F4, F5, F7). F1 should stay under 0.25 and each catch
  column over 0.25;
- ISR per arm: per image, mean |catch change| / |F1 change|, then the median;
- median `rho_feature` on F6 (consistent departure), with and without a reference.

Hypotheses to confirm or reject (the proposal's ✅/❌ table, restated as predictions):

| arm | F1 | F2 | F3 | F4 | F5 |
|---|---|---|---|---|---|
| base:A_mse, A_one_minus_ssim | moves (bad) | weak | catches | weak | catches |
| ref:B_pointwise_feature | moves (bad) | weak for DINOv2-like encoders | weak (stride) | weak | catches |
| ref:C_pixel_wcsi2 (= WCSI-2 with reference) | small | catches | catches | catches | catches |
| ref:D_feature_latent | small | weak | weak | weak | catches |
| ref:E_feature_sheaf | small | weak | weak | catches | catches |
| ref:F_hybrid | small | catches | catches | catches | catches |

Each hypothesis is tested twice: once with the learned stack and once with the Gabor control.
Where the control already catches something, the learned encoder is not needed for it.

Sweeps (each with its own `--tag`, 30 images, 8× only):

| knob | values | what it tests |
|---|---|---|
| `tau_anchor` (no reference) | 0.5, 1, 2, ∞ | D7: the anchor kernel. Report ground-truth `nr_feat_consist` and the F4 change |
| `tau_sem` | 0.1, 0.3, 0.5 | graph density |
| `k_pca` / `k_rho` | 8/2, 16/4, 32/8 | basis size |
| `--delta-e` | 1, 2, 3.5, 5 | F2/F7 severity: the response should be monotone (report Spearman) |
| analysis tile | 256, 512, 1024 SR px | same three tiles as the WCSI ablation |

Encoder swaps (one `--tag` each; edit `stack` in a copy of the config):

| slot | default | arms |
|---|---|---|
| texture | VGG-19 relu2_2/3_4/4_4 | ConvNeXt-L stage1/2; Gabor control |
| semantic | DINOv2-L/14 + registers, last layer | DINOv2-g; DINOv2-L layer 12; ConvNeXt-L stage3 (non-DINOv2 cross-check) |
| latent | FLUX.1 VAE | none; SD3.5 VAE |

## Part 2. Real outputs on DIV8K

`sh/score_methods.sh`: bicubic, VOSR, CASR, SeedVR and the ground truth at 8× and 4×,
analysis tile 512, with and without a reference, every term saved per image. Then:

1. Per method: mean score of every arm, median `rho_feature`, and phenotype counts.
2. Rank agreement between arms: Kendall τ of the method ranking per image, `F_hybrid`
   against `C_pixel_wcsi2` and against `B_pointwise_feature`, and `N3_hybrid` against
   `N0_wcsi2` (and against the legacy `N0_wcsi` if run with `--v1`). Where they disagree, name
   the terms that cause it.
3. Conflicts: the result JSON lists `encoder_conflict`. For VOSR, the DINOv2-based terms
   are flagged. Rerun VOSR and one other method with the ConvNeXt-stage3 semantic slot and
   report whether the ranking changes.
4. Ground truth without a reference: report `N3_hybrid` and `N0_wcsi2` of the ground truth
   against each method and the fraction of images on which a method beats the ground truth.
   With calibration the ground truth should be first (WCSI v1 scored it about 20, below CASR).

## Part 3. Human data (only if a study is run)

Public sets cannot test tiling: BAPPS is 64×64 patches and PIPAL patches are small. They can
check only the per-tile fidelity terms (`feat_wadd`, `feat_sem`, `pix_*`) against human
judgements (SRCC / KRCC). The consistency terms need a new 2AFC study at full resolution:
pairs of tiled SR outputs of the same image that differ in tile size, overlap or seed, shown
as full-image crops containing several tiles. Fit non-negative weights per term only on that
study, with 5-fold cross-validation, and report the fitted weights next to the unit weights.

## What to send back

The report follows the handoff format in `CLAUDE.md`, with these tables:

- `summary.md` from Part 1 for 8× and 4× (learned and Gabor), and the sweep tables;
- the Part 2 per-method table and the rank-agreement table;
- wall time per image (encoders and CPU separately) and peak GPU memory.

Name every encoder by checkpoint (for example "DINOv2-L/14 with registers") and never by
internal path.
