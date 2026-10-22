# WCSI ablation protocol (DIV8K, 4x and 8x)

Instruction for a model that runs or extends this directory. The scripts already implement this contract. Extend them. Do not rewrite them from scratch, and do not modify `genet/wcsi.py` or `genet/wcsi2.py` from here.

Three metrics share the harness (`--metric`): `wcsi` (v1, the default, `genet/wcsi.py`), `wcsi2` (WCSI-2 without reference) and `wcsi2_ref` (WCSI-2 against the ground truth), both in `genet/wcsi2.py`. See `genet/WCSI2.md` for why v2 exists and how it is calibrated.

## Hard rules

- Do not download DIV8K or any other dataset. The images are already on disk. Paths live in `genet_ablation/config.json`, copied from `config.example.json` and edited. `config.json` is gitignored.
- Do not run the full sweep unless the user asked to run it. A full pass is on the order of 10^4 to 10^5 scores, about 10 seconds each on a 33-megapixel image.
- Do not run super-resolution inference from this directory. These scripts only score images that already exist. Re-inference ablations are extra folders the user fills in.
- Do not add fidelity metrics (PSNR, SSIM, LPIPS) here. This harness is WCSI (v1 and v2) only.
- Score files are JSON, one image per file, so a killed run resumes. Do not switch the output to a single CSV written at the end.
- WCSI's `tile` and `overlap` are in low-resolution pixels. Every tile size in this experiment is quoted in upscaled (SR) pixels and converted by dividing by the scale. 1024 SR px at 8x is tile 128. At 4x it is tile 256.

## What is being measured

WCSI (`genet/wcsi.py`, `wcsi_score`) scores one super-resolved image against its low-resolution input. It does not look at the ground truth. 100 means every error term is zero. 50 means the terms average one just-noticeable difference.

The five terms, all in JND units:

| Term | What it sums |
| --- | --- |
| E_RG | colour of the blurred SR against the LR, per tile |
| E_GJ | seams between neighbouring tiles |
| E_PCP | texture orientation between neighbours, plus corner defects |
| E_cyto | invented texture of distant tiles that are the same material in the LR |
| E_sheaf | each tile's statistics against what the LR predicts, all pairs |

Natural photographs score low, mostly from E_cyto and the spectral-slope part of E_sheaf, because two distant patches of the same colour do not share one texture. Bicubic scores high because it invents nothing and so cannot disagree with itself. A method with WCSI above the ground truth is more self-consistent than the photograph. It is not closer to the photograph. The summary table reports the difference and the ratio anyway, because that comparison was requested. Rank methods by the term deltas, not by which WCSI is larger.

**WCSI-2** (`genet/wcsi2.py`) fixes the v1 ranking failure (the ground truth scoring below generative methods). Every tile is described by the band it invents above the LR Nyquist limit (log energy per octave and orientation, orientation tensor, chroma energy, top-octave peak ratio). Without reference, the target for that vector is predicted from the LR tile by a model fitted on ground-truth / LR pairs, and every term counts only the excess over what natural images score. Terms: `E_low` (coarse colour against the LR), `E_seam`, `E_detail` (missing or excess invented detail per tile), `E_consist` (tiles of one material departing differently). With reference (`wcsi2_ref`), the target is the ground-truth tile and the terms are `E_tone`, `E_colour`, `E_struct`, `E_detail`, `E_consist`, `E_seam`. The ground truth scores 100 there, so `--targets gt` is skipped.

Call:

```python
details = wcsi.wcsi_score(sr, lr, scale=8, tile=64, overlap=4, return_details=True)
# details["wcsi"], details["terms"][name], details["F_total"], details["kappa"], details["layout"]
```

```python
d = wcsi2.wcsi2_score(sr, lr, scale=8, tile=64, overlap=4, calib="auto", device="cuda:0", return_details=True)
d = wcsi2.wcsi2_ref_score(sr, gt, lr, scale=8, tile=64, overlap=4, device="cuda:0", return_details=True)
# d["wcsi2"] / d["wcsi2_ref"], d["terms"][name], d["F_total"], d["layout"], d["tiles"] (per-tile z and advice)
```

`sr` and `lr` may be paths or HxWx3 arrays. `tile_coords` replaces `tile` and `overlap` when the grid does not start at the origin. Passing both raises.

## Data contract

`config.json`:

- `gt`: directory of original DIV8K images. The user reports about 1400-1500 files. Official DIV8K (Gu et al., ICCVW 2019) is 1500 train / 100 val / 100 test, with validation and test at horizontal width 7680. Score whatever is in the directory. Do not "correct" the count by downloading the official test split.
- `scales.8.lr`: the same images, each side 8x smaller. Recipe the user described: three successive x2 downsamples. Do not regenerate this directory.
- `scales.4.lr`: each side 4x smaller (two successive x2). Same stems.
- `scales.*.methods.<name>`: a directory of 8x or 4x upscaled images, or `{"path": "...", "model_tile_sr": 512}`. Empty path means the run does not exist yet. Empty path on `bicubic` means the scorer synthesizes bicubic with `wcsi.upsample_bicubic` (Keys, a=-0.5). If a bicubic directory is filled in, those files are used instead.
- Stems must match across gt, lr and SR. Extensions may differ. Two files with the same stem in one directory is an error.
- `out`: where JSON results go. Not inside the image directories.

Pairing is by stem only. An SR file whose size is not an integer multiple of the LR is recorded as `status=error` and the sweep continues.

## Methods

Score the folders the user already has. The names below are the config keys.

**bicubic.** Deterministic enlargement of the LR. Consistency ceiling. No seams, no invented grain. Computed on the fly unless a directory is given. Slow in pure Python at 8K; that is expected.

**vosr.** VOSR, a vision-only generative DiT for super-resolution (arXiv:2604.03225, github.com/cswry/VOSR). VOSR 2.0 is a 1.4B one-step model. Multi-step sampling defaults to 25 steps. `--cfg_scale` trades fidelity to the LR against generative freedom (about -0.5 to 2; 0.5 is their practical default; they use -0.5 on RealSR). `--align_method` is `wavelet` (default), `adain`, or `nofix`. `--tile_size` is the DiT tile in pixels, in the same upscaled space as this experiment; 0 disables tiling, and tiling is required once the output exceeds 512 px because the model was trained natively up to 512. Default overlap 32. `--vae_tile_size` is a separate memory tile (example 1024) and is not the DiT tile. Posterior mode on is what they use for seam-free tiles. `-u` is the integer scale. Outputs are PNG.

**casr.** The user's CASR outputs. Two published methods share the name; do not guess which folder this is, and do not download either:

- Yoon et al., CVPRW 2024, real-time x4 cascade CNN (github.com/rlghksdbs/CASR). Feed-forward, not tiled, not generative. Expect WCSI near bicubic and E_GJ flat when the analysis grid slides.
- Chen et al., CasSR, cascaded diffusion (arXiv:2403.11451). Generative. Expect it to move like VOSR.

The paper CNN is x4 only. Leave the 8x path empty if there is no 8x run. The sweep skips empty paths.

**seedvr.** SeedVR, a diffusion-transformer video restorer (Wang et al., CVPR 2025, github.com/ByteDance-Seed/SeedVR), run here on stills. Official large-frame inference is not tiled (sequence parallel on H100s). Community and low-VRAM ports tile the VAE, commonly 1024 px with overlap 128, and that tiling is a memory choice that can introduce seams. One-step distilled checkpoints exist in those ports. Colour fix is often a wavelet injection, the same idea as VOSR's `wavelet`. Record 3B vs 7B and step count in the folder name.

**gt.** Not a method directory. `--targets gt` scores the original image against the LR, once per analysis grid.

## Factors

Primary grid, which `sh/score_div8k.sh` runs:

| Factor | Values | Where it acts |
| --- | --- | --- |
| scale | 4, 8 | which lr and SR folders |
| method | bicubic, vosr, casr, seedvr, and gt via --targets | which image is the "SR" |
| analysis tile | 1024, 512, 256 SR px | WCSI's grid, after dividing by the scale |
| analysis overlap | tile/16 SR px | WCSI default |
| analysis offset | 0 | grid origin |

Crossed ablations, pilot shells only:

| Factor | Values | Why |
| --- | --- | --- |
| offset | 0 and half the tile | If E_GJ is detecting the model's seams, it changes when the analysis grid slides off those seams. Bicubic and a non-tiled CNN should barely move. |
| overlap | 0, 32, 128 SR px | 32 is VOSR's default blend. 128 is the usual SeedVR VAE overlap. tile/16 is WCSI's default (64, 32, 16 SR px at the three tiles). |
| model tile | 256, 512, 1024, separate folders | The tile the model was run with. Score these at a fixed analysis tile (512) and at the matched tile. Scoring a 256-tile run only at 1024 confounds the two. |
| VOSR alignment | none, wavelet, adain | Should move E_RG. Wavelet is their default and copies LR low frequencies back in. |
| VOSR sampler | one-step; 25 steps at cfg -0.5, 0.5, 2 | Should move E_cyto and E_sheaf more than E_RG. |
| VOSR seed | 0, 1, 2 | Variance of a generative score. Pilot images only. |
| SeedVR VAE tile | 256, 512, 1024 | Same confound warning as model tile. |
| SeedVR wavelet | on (default folder) and off | E_RG. |

Config keys for the crossed runs are already in `config.example.json` (`vosr_tile256`, `vosr_align_none`, `seedvr_vae1024`, ...). A key with an empty path is skipped. Do not invent results for empty keys.

## Hypotheses the tables should be able to confirm or refute

1. Bicubic WCSI is above VOSR and SeedVR. A feed-forward CASR lands near bicubic. A diffusion CasSR does not.
2. WCSI of the ground truth is low and dominated by E_cyto. Many SR methods score above it. That gap is not a quality ranking.
3. For a tiled generative run, E_GJ changes between offset 0 and offset half-tile when the analysis tile matches the model tile. For bicubic it does not.
4. Analysis tiles of 256 SR px (32 LR px at 8x, 64 at 4x) are legal (WCSI's minimum is 8 LR px) but noisier spectral estimates. E_cyto's standard deviation across images should be wider at 256 than at 1024.
5. `align_method none` raises E_RG relative to wavelet.
6. Step count and cfg move E_cyto and E_sheaf more than E_RG.
7. Three seeds of one VOSR setting: report the per-image standard deviation. If it is comparable to the gap between methods, do not rank those methods on one seed.
8. WCSI-2 without reference: the ground truth scores at or above every method on most images (`frac_above_gt` well below 0.5). Bicubic loses on `E_detail` (missing detail), not on `E_consist`.
9. WCSI-2 with reference: methods that hallucinate one plausible texture everywhere have high `rho_coherence` and low `E_consist` but high `E_detail`; tile-by-tile hallucination raises `E_consist`.

## Scripts

All commands assume the repository root, `python3`, and `GENET_ABLATION_CONFIG` unset meaning `genet_ablation/config.json`.

| File | Role |
| --- | --- |
| `common.py` | config, stem index, SR-px to LR-px, offset lattice, output paths |
| `inventory.py` | pair stems and optionally print size ratios. Scores nothing. |
| `score_sweep.py` | the sweep. `--metric wcsi\|wcsi2\|wcsi2_ref`. `--dry-run` prints the job count. `--limit N` takes the first N stems. `--shard i/n` splits the sorted stems. `--workers` is a process pool. `--devices cuda:0,cuda:1` (or `cuda:0*2` for two workers on one GPU) runs WCSI-2 on torch with one worker process per entry, instead of `--workers`. `--calib` picks the WCSI-2 natural model (`auto`, a JSON, `none`). `--retry-errors` recomputes JSON with status error or missing. |
| `summarize.py` | `by_condition.csv`, `vs_gt.csv`, `summary.md`; `--metric` as above |
| `sh/*.sh` | the invocations below. They are not run by default. |

```bash
python3 genet_ablation/inventory.py --config genet_ablation/config.json --check-sizes 3
python3 genet_ablation/score_sweep.py --config genet_ablation/config.json --dry-run
python3 genet_ablation/score_sweep.py --config genet_ablation/config.json \
    --scales 8 --methods bicubic,vosr,casr,seedvr --targets sr,gt \
    --tiles 1024,512,256 --overlaps tile/16 --offsets 0 --limit 20 --workers 4
python3 genet_ablation/summarize.py --out /path/to/wcsi_ablation_out --dataset div8k
```

Shells:

- `sh/inventory.sh` — pairing report.
- `sh/score_pilot.sh` — 20 stems, both scales, primary methods, three tiles, gt included.
- `sh/score_div8k.sh` — every paired stem. Edit `--workers` for the machine.
- `sh/score_alignment.sh` — pilot, offsets `0,half`, overlaps `0,32,128`, tiles 512 and 1024. This is the seam probe.
- `sh/score_model_tiles.sh` — the `vosr_tile*` and `seedvr_vae*` keys, tiles 256, 512, 1024, so each model tile is scored on a matched grid and on the other two. Skips keys whose path is empty.
- `sh/score_wcsi2.sh` — WCSI-2 without and with reference on every paired stem, primary grid. `GPUS=cuda:0,...` (default 8 GPUs; empty = numpy on `WORKERS` processes), `CALIB`, `TILES`.
- `sh/summarize.sh` — writes the tables for every metric that has results.

Output path:

```
{out}/{dataset}/x{scale}/{method}/tile{tile}_ov{overlap}_off{offset}/{stem}.json            # wcsi (v1)
{out}/{dataset}/{metric}/x{scale}/{method}/tile{tile}_ov{overlap}_off{offset}/{stem}.json   # wcsi2, wcsi2_ref
```

JSON fields: `dataset, scale, method, image, tile_sr, overlap_sr, offset_sr, metric, status, score, F_total, n_tiles, seconds, sr_w, sr_h, device, notes, error`, the metric's terms, and `wcsi` + `kappa` (v1), `wcsi2` + `calibration` (the natural model file used), or `wcsi2_ref` + `rho_coherence`. v1 JSON written before `metric` existed is read as `wcsi`. `summarize.py --metric wcsi2` writes its tables under `{out}/{dataset}/wcsi2/`.

`status` is `ok`, `missing` (stem absent from that method directory), or `error` (WCSI raised, usually a size mismatch). `summarize.py` uses only `ok`, and lists failures at the bottom.

`vs_gt.csv` joins on scale, tile, overlap, offset and image. `delta_mean` is mean of WCSI(method) - WCSI(gt). `delta_E_RG` and the other term deltas are method minus gt, so positive means the method is worse than the photograph on that term. `frac_above_gt` is the fraction of images where the method's WCSI exceeds the photograph's.

## Runtime

Budget about 10 seconds and a little over 1 GB per 33-megapixel score, from the existing WCSI measurements. WCSI-2 on numpy, measured on 3-megapixel DIV2K images on one CPU process: about 1 s without and 3 s with reference, so roughly 11 s and 35 s at 33 megapixels. With `--devices`, the per-pixel work (CIELAB, coarse / residual split, tile spectra, phase congruency, gradients) runs on the GPU; decoding the PNG and the per-tile tables stay on the CPU, so use two workers per GPU (`cuda:k*2`) when decoding dominates. Primary grid: 2 scales x 3 tiles x (gt + bicubic + 3 methods) x 1500 images = 54,000 scores, on the order of a day on 8 cores if every method folder exists. Bicubic synthesis is slower than reading a file. The pilot shell exists so the full shell is a deliberate second step.

## How to extend

- A new method is a new key under `scales.<s>.methods` and a directory of images with the same stems. No code change.
- A new analysis factor belongs in `score_sweep.py`'s CLI and in `common.result_path`, so old JSON is not overwritten.
- Keep the per-image JSON small. Per-tile vectors and the figures from `genet/wcsi_viz.py` are for a handful of stems (`--limit` plus a stem list), not for 1500 images.
- Tests are `pytest -q genet_ablation/test_ablation.py`. They use tiny synthetic images. Do not point them at DIV8K.
- The ground truth enters a score only through `--metric wcsi2_ref`. Do not fold it into `wcsi_score` or `wcsi2_score`. `with_ref_genet/gmci.py` (GMCI) is superseded by `wcsi2_ref` and is not part of this harness.

## Out of scope

Inference commands, dataset download, PSNR/SSIM/LPIPS, and any metric other than `wcsi2_ref` that reads the ground truth while computing the score. Comparing an already-computed WCSI(gt) with WCSI(sr) is in scope and is what `summarize.py` does.
