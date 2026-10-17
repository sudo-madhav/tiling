# Deep-Dive Analysis: Directions for the Super-Resolution Tile Consistency Paper

Written 2026-10-17 on branch `sr`.
Synthesizes `plan/sr_decision.md`, `plan/sr_handoff.md`, and `plan/sr_meeting_2026-10-17.md`, combined with recent literature on tiled generation/restoration (SWITi 2026, CASR CVPR 2026, SyncDiffusion, DemoFusion, ScaleCrafter, AccDiffusion, DiffuseHigh) and region/pairwise perceptual quality assessment (PandaBench ICLR 2026, RQI CVPR 2026, RefReward-SR 2026, CrossScore 2024, BiAtten-Net 2024).

No LaTeX math blocks are used in this document so it reads cleanly on mobile.

---

## 0. Where the Three Existing Documents Stand (and the Core Tension)

Across `sr_decision.md`, `sr_handoff.md`, and `sr_meeting_2026-10-17.md`, three points are already settled:
1. **Not a seam detector:** Boundary grid lines are handled by overlap blending + VAE mode decoding, and SWITi (arXiv:2607.18990, July 2026) already publishes no-reference seam gradient tests (FRT and ASV).
2. **Setup & pipeline:** Take an 8K ground-truth image `Y`, apply two 2x Real-ESRGAN degradation rounds to get a ~2K input `x`, cut `x` into `n` tiles (e.g., 4x4 = 16 tiles of 512x512 with overlap), run 4x tiled SR (VOSR, CASR, SeedVR) to get 2048x2048 output tiles, and stitch back to 8K `Y_hat`. Keep both the stitched `Y_hat` and the `n` unblended tile interiors.
3. **First concrete experiment:** Before training any neural scorer or modifying any SR backbone, run 1-2 tiled SR methods on 10-20 curated images and compare **global full-image error against `Y`** versus **per-tile interior error against `Y_i`** to see whether tiling inconsistency appears as sharp per-tile error peaks that the global average hides.

The unresolved tension across the three files comes down to **two foundational design choices** and **two competing paper shapes**:
- **Decision 1:** Should the evaluation metric intervene inside the inference pipeline of existing SR methods (measuring changes under pipeline perturbations), or should it be strictly method-agnostic (taking only the output image + grid / reference)?
- **Decision 2:** Should the metric include the ground-truth image `Y` (reference-based) or operate without `Y` (no-reference), and how do we simultaneously capture **tile-to-GT consistency** and **all-pairs cross-tile consistency**?
- **Paper Shape:** Do we make the **Metric primary (Path A)** (with a VLM pairwise `(tile_i, tile_j, Y)` judge and a learned inter-tile metric predictor) or make the **SR Method primary (Path B)** (using a clean, constrained notional GT-based metric to measure variation/peaks and designing our SR method to optimize it)?

Below is a detailed analysis of each question, followed by concrete architectural solutions and a clear recommendation.

---

## 1. Decision 1: Pipeline-Intervention Metric vs. Method-Agnostic (Output-Only) Metric

### What "Introducing Something in the Inference Pipeline" Would Mean
An intervention-based (white-box or multi-pass) evaluation modifies how an existing SR method runs and measures how much the output changes:
1. **Grid-shift sensitivity (Equivariance probe):** Run the SR model on grid `G`, then shift the grid by half a tile `(stride / 2)` and run it again on `G_shifted`. Measure the pixel/feature difference `dist(Y_hat(G), Y_hat(G_shifted))` on the interior of the image.
2. **Context-window ablation probe:** Run tile `i` at 512x512 alone vs. inside a larger 1024x1024 crop, and measure how much tile `i` changes when given surrounding context.
3. **Latent / noise / GroupNorm perturbation:** Compare outputs with independent per-tile VAE statistics vs. synchronized statistics.

### Why Pipeline Intervention Should NOT Be the Primary Metric
1. **Cannot evaluate saved images or external benchmarks:** In SR papers, baselines are often evaluated from released result folders or fixed inference scripts. A metric that requires hooking into each method's tiler or running 2x-4x forward passes per image cannot be used as a standard drop-in IQA metric.
2. **Confuses stochasticity with single-image inconsistency:** Diffusion and flow SR models are stochastic or seed-dependent. If you shift the grid and re-run, two runs can differ in high-frequency phase even when *each individual stitched image* looks internally consistent. Conversely, a deterministic rule that fails on every low-contrast tile will fail identically on two slightly shifted grids inside a large flat region, giving near-zero shift difference while the single output image is visibly broken.
3. **2x to 4x evaluation compute cost:** Running 8K diffusion SR once is already slow; requiring multiple shifted passes just to compute a metric is impractical.

### Verdict on Decision 1
- **The primary metric MUST be 100% method-agnostic.** Its inputs are strictly the stitched output `Y_hat`, the tile grid coordinates `G = {T_1, ..., T_n}` (known from the tile size and overlap), and the reference `Y` (or input `x`).
- **Where pipeline intervention DOES belong:** Use grid-shift (`G` vs. `G_shifted`) and context-size (`512` vs. `1024`) interventions **only in Section 3 (Motivation / Root-Cause Diagnosis)** on 10-20 images to prove *why* tiling inconsistency happens (e.g., proving that VOSR flips from "keep bokeh soft" to "unbokeh sharply" as soon as a tile's crop includes an object boundary).

---

## 2. Decision 2: Reference-Based (with Ground Truth `Y`) vs. No-Reference

### Why Tiling Inconsistency Splits into Two Distinct Physical Modes
The reason `sr_handoff.md` and `sr_meeting_2026-10-17.md` pulled in different directions on ground truth `Y` is that **"tile inconsistency" is actually the union of two different failure modes**:

| Failure Mode | Physical Cause in Tiled SR | Concrete Example | Does per-tile `dist(Y_hat_i, Y_i)` catch it? | Does naive cross-tile `dist(Y_hat_i, Y_hat_j)` catch it? |
|---|---|---|---|---|
| **Mode 1: Uneven Fidelity / Regime Switching** | Local context starvation causes the model to restore/deblur in some tiles and leave adjacent tiles soft, or shift local tone/exposure relative to the true scene. | A hat or background spans 8 tiles: 5 tiles unbokeh into sharp weave matching `Y`, while 3 tiles stay blurry like `x`. | **YES.** The 5 restored tiles have low error against `Y_i`; the 3 blurry tiles have high error against `Y_i`. Variance / peak count across `i=1..n` spikes. | **NO (confounded).** Without `Y`, you cannot tell if the blurry tile is a restoration failure or genuine depth-of-field bokeh in the photograph. |
| **Mode 2: Cross-Tile Hallucination Disagreement** | 4x downscaling destroys exact high-frequency phase. Two tiles of the same surface hallucinate plausible but *different* micro-textures. | Distant tiles of water ripples, grass, or straw weave: Tile `i` invents tight vertical ripples; Tile `j` invents wide horizontal ripples. | **OFTEN NO.** Both `Y_hat_i` and `Y_hat_j` are equally far from exact pixel/feature phase of `Y_i` and `Y_j` (`d(Y_hat_i, Y_i) approx d(Y_hat_j, Y_j)`), so per-tile error against `Y` stays flat! | **ONLY IF YOU KNOW `i` AND `j` ARE THE SAME SURFACE.** Naive all-pairs comparison (SyncDiffusion Intra-LPIPS) fails because sky and grass are supposed to differ. |

### The Key Insight: Ground Truth `Y` Solves BOTH Mode 1 and Mode 2
In `sr_meeting_2026-10-17.md`, Mode 2 (cross-tile hallucination) was viewed as hard because comparing Tile `i` and Tile `j` requires knowing which tiles *should* have matching texture (which forced CASR to run SAM superpixels + Depth Anything on the LR image).

**Wait: when Ground Truth `Y` is included in the metric, finding which tile pairs `(i, j)` should match becomes trivial and exact!**
Why? Because `Y` is the clean 8K ground truth:
- In Mode 1 (Tile-to-GT), `Y_i` tells us the exact target sharpness, structure, and color for tile `i`.
- In Mode 2 (Tile-to-Tile across all `n(n-1)/2` pairs), the ground-truth pair `(Y_i, Y_j)` tells us **whether tiles `i` and `j` are the same surface/material and how similar their texture distributions should be!**

Specifically, for any pair of tiles `(i, j)` in `1..n`:
- Let `d_tex(A, B)` be a phase-insensitive texture/style distance (e.g., Gram matrix distance on shallow VGG/wavelet features, DISTS texture term, or DINOv2 token distribution distance).
- In the ground truth `Y`, if Tile `i` and Tile `j` are both water (or both the same hat), then `d_tex(Y_i, Y_j)` is **near zero**, even though their exact pixel phases differ! Meanwhile, if Tile `i` is sky and Tile `j` is grass, `d_tex(Y_i, Y_j)` is **large**.
- Therefore, the **GT-Referenced Cross-Tile Relational Error** for pair `(i, j)` is:
  `PairError(i, j) = w_ij * max(0, d_tex(Y_hat_i, Y_hat_j) - d_tex(Y_i, Y_j))`
  where `w_ij = exp(-d_sem(Y_i, Y_j) / tau)` is the ground-truth affinity gate (high when `Y_i` and `Y_j` share semantic/material content in `Y`, near zero when `Y_i` and `Y_j` are different objects).

Look at what happens in the water/hat hallucination example under this formula:
1. Both `Y_i` and `Y_j` are the same water surface in `Y` -> `w_ij` is close to `1`, and `d_tex(Y_i, Y_j)` is close to `0` (because `d_tex` compares texture statistics/distributions, not pixel alignment).
2. In `Y_hat`, Tile `i` hallucinates fine ripples and Tile `j` hallucinates waxy coarse ripples -> `d_tex(Y_hat_i, Y_hat_j)` is **large**!
3. Even if `d_tex(Y_hat_i, Y_i)` and `d_tex(Y_hat_j, Y_j)` were similar, `d_tex(Y_hat_i, Y_hat_j) - d_tex(Y_i, Y_j)` spikes sharply!
4. And for sky (`i`) vs. grass (`j`), `w_ij` is near `0` AND `d_tex(Y_hat_i, Y_hat_j) approx d_tex(Y_i, Y_j)`, so `PairError(i, j) = 0`.

### Verdict on Decision 2
- **Include Ground Truth `Y` (Reference-Based Metric) as the primary formulation.** In every standard SR benchmark (DIV8K, LSDIR, UHD4K, Real-ESRGAN synthetic splits), `Y` is available. Including `Y` eliminates false positives on real scene transitions and solves both Mode 1 (unary tile-to-GT consistency) and Mode 2 (all-pairs cross-tile consistency gated by `Y`).
- **How to support No-Reference (wild inputs without `Y`) as an extension:** When `Y` is unavailable at deployment time, replace `Y` in the affinity gate `w_ij` and low-frequency anchor with the bicubic-upsampled LR input `Up(x)` (since semantic material similarity `w_ij` on `Up(x)` matches `Y` with >0.95 correlation, as shown in RefReward-SR, arXiv:2603.24198).

---

## 3. Deep Dive: Path A (Metric-Primary) — VLM Pairwise Judge & Learned Metric Predictor

In your prompt, you outlined the following pipeline for a Metric-focused direction:
1. Develop a metric addressing **both tile consistency with GT** and **tile consistency across all `n(n-1)/2` pairs of tiles**.
2. Support/label this with **general-purpose vision-language models (VLMs/MLLMs)** by giving them a pair of tiles and the GT image in every call, asking if they are consistent (`0` or `1`), and computing the **fraction of agreeing pairs** as the score.
3. With a small dataset (e.g., 100-200 images), **train a metric score predictor** — while solving the core modeling problem: *how do we actually force the predictor to focus on inter-tiling given the full image?*

Here is a rigorous analysis of what is blurry in this pipeline and how to engineer each piece so it actually works.

### 3.1 The VLM All-Pairs (`n(n-1)/2`) Judge: 4 Hidden Traps and How to Fix Them

For a 4x4 grid (`n = 16` tiles), there are `n(n-1)/2 = 120` tile pairs per image. (For an 8x8 grid, `n = 64`, there are `2,016` pairs per image). If we naively send all 120 pairs `(Y_hat_i, Y_hat_j, Y)` to GPT-4o / Gemini / Qwen2.5-VL and ask *"Are these two tiles consistent? (0 or 1)"*, four problems occur:

#### Trap 1: The Unrelated-Pair Denominator Problem
In a natural photo (say, a portrait outdoors), out of 120 tile pairs, roughly 80-90 pairs compare completely unrelated regions (e.g., Tile 1 is blue sky, Tile 15 is a dark jacket).
- If you ask a VLM *"Are Tile 1 and Tile 15 consistent?"*, either:
  - It says `1` automatically because they are different objects and there is no shared texture to compare (which inflates the score so every image gets >0.85 regardless of tiling artifacts), OR
  - It gets confused by "consistent" and says `0` because one is bright blue and one is dark black.
- **Fix — Filter Pairs by the GT Affinity Graph `G_Y` Before Calling the VLM:**
  - Use the ground truth `Y` to automatically select **Comparable Tile Pairs `E_comp` subset of `{1..n} x {1..n}`**:
    1. All spatially adjacent tile pairs `(i, j)` that share a border and whose boundary in `Y` is not a hard semantic occlusion edge.
    2. All non-adjacent (distant) tile pairs `(i, j)` that share similar semantic/material content in `Y` (`CosineSim(DINOv2(Y_i), DINOv2(Y_j)) > 0.75` or low color/Gram distance in `Y`).
  - Typically `|E_comp|` is **15 to 35 pairs per image** instead of 120 (or 2,016). Every pair in `E_comp` now represents two tiles that *genuinely belong to the same object, background, or material in `Y`*.
  - Furthermore, separate the VLM calls into **two clean queries**:
    - **Query Type 1 — Unary Tile-to-GT Fidelity (`n = 16` calls per image):** Show `(Y_i, Y_hat_i)` at 1:1 native resolution. Ask: *"Does the restored tile Y_hat_i match the sharpness regime, structural edges, and tone of ground truth Y_i?"* Record `u_i in {0, 1}` (or continuous logprob `in [0, 1]`).
      - **Unary Consistency Score:** Not the mean of `u_i` (which is just average quality), but `1 - 4 * Var(u_i)` (since a Bernoulli variable has max variance 0.25 when half the tiles pass and half fail!) or `1 - OutlierFraction(u_i)`.
    - **Query Type 2 — Pairwise Cross-Tile Consistency (`|E_comp| approx 25` calls per image):** For each comparable pair `(i, j) in E_comp`, show the **2x2 Panel**:
      - Top-Left: `Y_i` (GT Tile A) | Top-Right: `Y_j` (GT Tile B)
      - Bottom-Left: `Y_hat_i` (SR Tile A') | Bottom-Right: `Y_hat_j` (SR Tile B')
      - Ask: *"In the ground truth (top row), Tile A and Tile B show the same surface/material. In the super-resolved output (bottom row), are Tile A' and Tile B' rendered consistently with each other in (a) sharpness/deblurring degree, (b) texture pattern/grain style, and (c) color/tone, relative to the top row? Output 1 if consistent, 0 if inconsistent."*
      - **Pairwise Consistency Score:** `S_pair = (1 / |E_comp|) * sum_{(i,j) in E_comp} c_ij`.

#### Trap 2: VLM Input Resizing Destroys 8K High-Frequency Detail
As demonstrated in **PandaBench** (ICLR 2026, arXiv:2604.11004), when you pass a full 8K image `Y` (or even two full 2048x2048 tiles side by side) into an MLLM, its vision encoder downsamples the image to 448x448 or 768x768 tokens. That 4x downsampling **erases the exact high-frequency SR hallucinations and blur differences we want the VLM to judge!**
- **Fix:** Never pass the full 8K image as the pixel source for texture judgment. Pass:
  1. A small low-res context thumbnail of `Y` with red/blue boxes marking where Tile `i` and Tile `j` sit in the scene (for semantic context), AND
  2. **1:1 native-pixel crops (e.g., 512x512 matched high-gradient sub-crops inside `Y_i, Y_j, Y_hat_i, Y_hat_j`)** in the 2x2 comparison panel so the VLM's ViT patches see actual native SR pixels without downsampling.

#### Trap 3: Binary `0/1` Quantization Noise
When a pair `(i, j)` has mild-to-moderate inconsistency, hard `0/1` decoding is brittle and flips with prompt phrasing.
- **Fix:** Run the VLM with `temperature = 0` and read the **log-probabilities of the `"1"` and `"0"` output tokens**:
  `p_ij = exp(logprob("1")) / (exp(logprob("1")) + exp(logprob("0"))) in [0, 1]`
  This gives a continuous, calibrated probability `p_ij in [0, 1]` for every pair while still allowing hard thresholding (`p_ij > 0.5`) when reporting the binary agreement fraction.

#### Trap 4: Reviewer Skepticism of Unvalidated VLM Judges
CVPR reviewers will not accept a VLM's `0/1` fraction as ground truth unless you prove the VLM agrees with humans on this specific task.
- **Fix:** Sample 200 tile pairs (100 called `0` by the VLM, 100 called `1` by the VLM) across 20 images, have 3-5 human raters mark them `0/1` using the exact same 2x2 panel, and report **Human-VLM Cohen's Kappa / % Agreement** (targeting >85% agreement) plus inspect the VLM's 1-sentence rationales.

---

### 3.2 Solving the Modeling Blur: How to Get a Learned Metric Predictor to Focus on Inter-Tiling

You raised the exact right technical concern:
> *"With a small dataset we can train a metric score predictor. Here lot of stuff is blur like how do we actually get model to focus on intertiling given the full image. Somethings need to be thought much more carefully."*

Let us be completely direct about why a naive predictor fails and how to design one that actually works.

#### Why "Full Image -> Backbone -> Scalar Score" Fails Completely
If you collect 150-200 stitched 8K images `Y_hat`, label each image with a scalar score `S in [0, 1]` (from the VLM fraction or GT residual), and train a ConvNeXt / ViT / Swin network `f(Y_hat) -> S`:
1. **Only 150 training targets:** A neural network given 150 `(image, scalar)` pairs will memorize which of the 20 base scenes or which SR method (VOSR vs. Real-ESRGAN) produced the image in 3 epochs. It will never learn to compare tiles.
2. **Resolution vs. Receptive Field:** Downsampling 8K to 512x512 destroys micro-texture; keeping 8K in a single dense backbone runs out of GPU memory; running a local patch CNN averages local patch scores (`mean_i f(patch_i)`), which mathematically **cannot measure variance or disagreement between patch `i` and patch `j`**!
   - *Mathematical proof:* Any linear average of per-tile scores `(1/n) sum_i s(Y_hat_i)` gives the exact same derivative whether two tiles agree or disagree; it is blind to inter-tile variance!

#### The Correct Architecture: A Tile-Graph Relational Predictor Supervised on Pairs (`c_ij`), Not Just Image Scalars

To force the network to focus **exclusively on inter-tiling** and train reliably on a small dataset of 100-200 images, we must change **three things**: (1) the input representation, (2) the relational inductive bias inside the layer, and (3) the level of supervision.

```
Stitched Image Y_hat (8K) + Reference Y (or LR x) + Known Grid G (n=16 tiles)
       |
       v  [1. Crop Tile Interiors T_1 ... T_n (remove overlap bands)]
       |
       v  [2. Frozen Multi-Scale Feature Bank per Tile (No backprop into giant ViT)]
       |      - Semantic/Structure tokens: Frozen DINOv2-B14
       |      - Micro-Texture descriptor: Shallow VGG / Wavelet Gram diagonal & cross-bands
       |      - Tone/Sharpness moments: LAB mean/std + Laplacian high-pass energy
       |
       v  [3. Lightweight Learnable Projection Head phi_theta (~0.3M params)]
       |      Produces per-tile embedding z_hat_i and reference embedding z_ref_i
       |
       +----------------------------------------+----------------------------------------+
       |                                                                                 |
       v  [Branch A: Unary Tile-to-Ref Deviation]                                        v  [Branch B: All-Pairs Cross-Tile Graph Comparator]
  1. Tile residual: r_i = (z_hat_i - z_ref_i) / scale(z_ref_i)                      1. Ref Affinity Gate: w_ij = CosineSim(DINO(ref_i), DINO(ref_j))
  2. ZERO-MEAN INTER-TILE LAYER:                                                    2. Pairwise Relational Diff (CrossScore / BiAtten style):
     delta_i = r_i - (1/n) * sum_k r_k                                                 D_ij = |psi(z_hat_i) - psi(z_hat_j)| - |psi(z_ref_i) - psi(z_ref_j)|
  3. Anomaly head: a_i = MLP_unary(|delta_i|) in [0, 1]                             3. Edge Predictor: p_hat_ij = Sigmoid(MLP_pair(D_ij, w_ij)) in [0, 1]
       |                                                                                 |
       v  Supervised by Unary Labels u_i (16 per image)                                  v  Supervised by VLM Pair Labels c_ij (~30-120 per image!)
       |                                                                                 |
       +----------------------------------------+----------------------------------------+
                                                |
                                                v  [4. Graph Pooling -> Image Consistency Score]
                                   S_pred = alpha * mean_i(a_i) + (1 - alpha) * sum_{i<j}(w_ij * p_hat_ij) / sum_{i<j}(w_ij)
```

#### Why This Specific Architecture Solves the Small-Data & Inter-Tiling Problem
1. **Supervision multiplies by 30x to 100x:**
   - Instead of supervising only the final scalar `S_pred` on 150 images (150 labels), we supervise **every tile `a_i` against `u_i`** (`150 * 16 = 2,400` labels) and **every comparable tile pair `p_hat_ij` against the VLM pair label `c_ij`** (`150 * 30 = 4,500` to `18,000` labels!).
   - Training a 0.3M-parameter comparator head `MLP_pair` on 4,500+ pair labels on top of frozen DINOv2 + Gram features will converge cleanly without overfitting.
2. **Hard Architectural Invariance to Global Quality (Cannot Cheat via Global Blur/Sharpness):**
   - Look at Step 2 in Branch A: `delta_i = r_i - mean_k(r_k)`.
   - If an image is uniformly degraded or uniformly restored across all 16 tiles (`r_1 = r_2 = ... = r_16`), then **`delta_i = 0` identically for every tile!**
   - Look at Step 2 in Branch B: `D_ij = |psi(z_hat_i) - psi(z_hat_j)| - |psi(z_ref_i) - psi(z_ref_j)|`.
   - If two tiles differ only because the scene content differs (sky vs. grass), then `|psi(z_hat_i) - psi(z_hat_j)|` equals `|psi(z_ref_i) - psi(z_ref_j)|`, so **`D_ij = 0` identically!**
   - Thus, by mathematical construction, the network **cannot** predict inconsistency from scene content or uniform global blur. It can **only** produce a non-zero output when tiles diverge from each other relative to the reference.
3. **Seamless Bridge from Full-Reference (`Y`) to No-Reference (`x`):**
   - During training, `ref_i` can be either the ground-truth tile `Y_i` (Full-Reference predictor) OR the bicubic-upsampled LR tile `Up(x_i)` (No-Reference / LR-Referenced student predictor, supervised by the exact same GT-backed VLM labels `c_ij`!).
   - That means if you want a predictor that runs at test time *without* `Y` (using only `Y_hat` and `x`), you simply feed `ref_i = Up(x_i)` into the exact same architecture and train it to predict the GT-backed VLM edge labels `c_ij`!

---

## 4. Deep Dive: Path B (Method-Primary, Metric-Secondary) — Notional GT Metric + Consistency-Preserving SR Method

Now consider the second philosophy from your prompt and from `sr_meeting_2026-10-17.md`:
> *"If metrics is secondary then we will have some notional metric (its design will be constrained to referring to ground truth image something like variation from baseline / number of anomalies in some stat compared to ground truth image). Then with our method we will try to maximise that metric."*

Let us examine how to make Path B completely solid for CVPR 2027.

### 4.1 Designing the "Notional GT Metric" (Zero Training, Transparent, Undisputed)

When the SR method is the primary contribution of the paper, **the metric must NOT be a black-box neural network that we trained ourselves**, because reviewers immediately suspect the learned metric was tuned to favor our method ("training to the test").

Instead, the notional metric should be **100% closed-form, reference-based (using `Y`), and built directly on standard per-tile error distributions**.

Given an 8K ground truth `Y`, stitched SR output `Y_hat`, and `n` tile interiors `i = 1..n`:
Let `e_i` be a per-tile error against `Y_i`. We evaluate `e_i` on two levels:
- **Standard accepted perceptual/pixel errors:** `e_i^LPIPS = LPIPS(Y_hat_i, Y_i)`, `e_i^DISTS = DISTS(Y_hat_i, Y_i)`, and `e_i^PSNR = MSE(Y_hat_i, Y_i)`.
- **Content-normalized 5-coordinate residual `r_i`** (from `sr_handoff.md` Section 3): Restoration (`|HF(Y_hat_i) - HF(Y_i)| / HF(Y_i)`), Structure (`1 - GradCorr(Y_hat_i, Y_i)`), Texture (`GramDist(Y_hat_i, Y_i) / GramEnergy(Y_i)`), Tone (`DeltaLab / ChromaSpread(Y_i)`), and Artifacts (`ExtraHF(Y_hat_i, Y_i) / HF(Y_i)`).

From the profile `(e_1, e_2, ..., e_n)` across the `n` tiles of an image, define **three notional inconsistency statistics**:

1. **Tile Error Spread (`TES` — Variation Across Tiles):**
   - `TES = Std_{i=1..n}(e_i)` (or robust interquartile range `IQR(e_i)`).
   - Directly answers: *How unevenly did the SR model restore the tiles of this image?*
2. **Tile Anomaly / Peak Count (`TAC` — Number of Outlier Tiles):**
   - For each image, define the image's baseline tile error as the median across its tiles: `m = median_{k=1..n}(e_k)` (or the 25th percentile `P25(e_k)`, representing the well-restored tiles of that image).
   - A tile `i` is an **anomalous peak** if its error exceeds the image baseline by more than `tau` (e.g., `e_i - m > tau`, or robust z-score `(e_i - m) / MAD(e) > 2.0`).
   - Report **Peak Count / Fraction (`TAC`)** and **Peak Severity (`MaxPeak = max_i (e_i - m)`)**.
   - This directly captures the exact visual intuition from the meeting: *a consistent SR output has a flat per-tile error curve across the 16 tiles (`TAC = 0`), whereas a tile-inconsistent output has sharp spikes on the 3-4 broken tiles (`TAC = 3 or 4`).*
3. **GT-Matched Pair Texture Spread (`MPTS` — Closed-Form All-Pairs Check):**
   - For all tile pairs `(i, j)` whose ground-truth interiors have high similarity in `Y` (`CosineSim(DINOv2(Y_i), DINOv2(Y_j)) > 0.80` and `GramDist(Y_i, Y_j) < tau_G`), compute the excess texture distance `max(0, GramDist(Y_hat_i, Y_hat_j) - GramDist(Y_i, Y_j))`.
   - Plus, include the **VLM Pairwise Consistency Fraction (`S_pair` from Section 3.1)** as an independent external audit column in the main comparison table!

### 4.2 The Crucial Guardrail in Path B: Preventing the "Uniform Blur Cheat"

If a method simply tries to minimize `TES` (variance across tiles) or `TAC` (number of peaks above median), there is a trivial degenerate solution:
- **Make every tile blurry!** Bicubic upsampling has almost zero variance across normalized tile errors (`TES approx 0`, `TAC = 0`) because it fails equally everywhere, raising the median error `m` up to the ceiling.

Therefore, in Path B, our method and tables must enforce a **Two-Part Pareto Criterion**:
1. **Mean Error Floor (`Mean_i e_i`):** Our method must achieve **equal or lower `Mean_i e_i`** (and equal or higher global `LPIPS / DISTS / MUSIQ / MANIQA`) than the baseline tiled SR method (VOSR / CASR / SeedVR).
2. **Peak Suppression (`TES`, `TAC`, `MaxPeak`):** Our method must **pull the anomalous high-error tiles (`P90 / Max` tile error) down toward the well-restored tiles (`P25 / Median` tile error)**.
   - *Visual figure in the paper:* Plot the sorted tile errors `e_(1) <= e_(2) <= ... <= e_(16)` averaged over the test set. Baseline VOSR/SeedVR curves bend sharply upward on tiles `12..16` (the inconsistent outlier tiles). Our method's curve stays pinned to the baseline on tiles `1..10` and **flattens the upward tail on tiles `11..16`**.

### 4.3 How We Design the SR Method to Crush `TES`, `TAC`, and Pair Disagreement

To beat VOSR, SeedVR, and CASR on these consistency metrics without sacrificing sharpness, our method must directly fix the two physical causes of per-tile peaks:

#### Step 0: The Non-Negotiable Trivial Baseline (Pipeline 1 Controls)
Before claiming a new method contribution, we apply the free engineering fixes to the baseline tiler:
- One shared global random noise tensor sliced per tile (instead of independent `randn` draws).
- Global VAE GroupNorm mean/variance synchronization across tiles during decoding (ScaleCrafter fix for tone drift).
- Gaussian overlap blending + wavelet color fix to `Up(x)`.
*Why:* Whatever `TES / TAC` peaks survive Step 0 are **true semantic/restoration regime switches and texture divergences** that trivial blending cannot fix.

#### Step 1: Global Coarse Anchor Pass (Eliminates Mode 1 Restoration Peaks / `TAC`)
- **Why restoration peaks happen:** When VOSR or SeedVR processes a 512x512 tile `x_i` in isolation, a tile sitting entirely inside a blurry/defocused region or flat surface lacks global object boundaries. The model guesses "this is background bokeh -> keep blurry" in Tile `i`, while in adjacent Tile `j` (which contains an edge of the hat or building) the model guesses "this is degraded foreground -> sharpen aggressively."
- **The Fix:** Before running the 16 tiles, downsample the full 2K input `x` by 4x to a single `512x512` global image `x_global` and run the SR model **once** to get a `2048x2048` global coarse restoration `Y_hat_global`.
  - Compute cost: **1/16th of a 4x4 tiled run (+6% overhead).**
  - What `Y_hat_global` contains: Because `x_global` sees the entire scene in one receptive field, the SR model makes **one globally coherent decision** on which objects are restored/unbokehed vs. kept in bokeh, and locks in global structure and tone!
  - How each tile uses `Y_hat_global`: For each tile `i`, upsample its corresponding crop `Y_hat_global_i` and use it in two ways:
    1. **Mid-Frequency Structural & Restoration-Gain Anchor:** Following DiffuseHigh / DemoFusion, inject the low-to-mid frequency band of `Y_hat_global_i` into tile `i`'s trajectory (or constrain tile `i`'s local high-pass restoration gain ratio to match `Y_hat_global_i`'s gain ratio). Now Tile `i` *cannot* stay soft when Tile `j` on the same object restored sharply!

#### Step 2: Sparse Affinity-Routed Exemplar KV Sharing (Eliminates Mode 2 Cross-Tile Texture Drift)
- **Why cross-tile texture disagreement happens:** Even when two tiles `i` and `j` both restore sharply, `Y_hat_global` only has 2K resolution (it lacks 8K micro-texture). So Tile `i` and Tile `j` still invent their own 8K micro-texture independently. And CASR's all-to-all bottleneck attention across all tiles scales as `O(n^2)` and blurs high frequencies via MultiDiffusion averaging.
- **The Fix:**
  1. Build the sparse **Material Affinity Graph `G_x`** on the LR input `x` (using DINOv2 + superpixel/depth similarity on `x`, plus `Y_hat_global` features).
  2. Within each connected material cluster of `G_x`, pick the single **Anchor Tile `i*`** with the strongest contextual edges (highest restoration confidence in `Y_hat_global`).
  3. Run Anchor Tiles first. Cache a compact set of **top-K mid/high-layer self-attention KV tokens** from `i*` (with 2D positional encoding / RoPE stripped so only texture style, not spatial coordinates, is shared).
  4. When denoising/upsampling every sibling tile `j` in that material cluster, concatenate the Anchor Tile's Exemplar KV tokens into tile `j`'s self-attention keys/values gated by the spatial region mask.
- **Result:**
  - Every tile of the hat or water surface attends to the **same Anchor Tile exemplar tokens** for its 8K micro-texture (solving Mode 2 pairwise consistency).
  - Every tile follows the **same Global Coarse Anchor** for its restoration regime and structure (solving Mode 1 `TES` and `TAC` peaks).
  - Memory is `O(1)` per tile (sequential or batched tile execution with a tiny cached KV bank), working at 8K-16K without OOM!

---

## 5. Head-to-Head Comparison: Path A (Metric-Primary) vs. Path B (Method-Primary)

| Dimension | Path A: Metric-Primary (Unified Metric + VLM Judge + Learned TGRN Predictor) | Path B: Method-Primary (Constrained Notional GT Metric + Coarse-Anchored Exemplar SR) |
|---|---|---|
| **Core Claim** | Standard SR metrics (global PSNR/LPIPS/MUSIQ) are blind to tiling inconsistency; we introduce the first reference-grounded unary + all-pairs tile consistency benchmark, VLM evaluation protocol, and tile-graph metric predictor. | Tiled generative SR suffers from cross-tile regime switching and texture divergence (quantified by per-tile GT error peaks and VLM pair audits); our plug-and-play Coarse-Anchored + Affinity-KV inference eliminates these peaks across VOSR/SeedVR/CASR. |
| **Main CVPR Reviewer Risk** | IQA reviewers demand a large human MOS/2AFC benchmark (typically 500-1,000+ images, 20+ human raters) to validate a learned metric. Training a predictor on 150 VLM-labeled images risks being attacked as "distilling an unverified VLM on a tiny dataset." | Restoration reviewers ask: "How does this compare to CASR (CVPR 2026) and DemoFusion?" Must clearly show that CASR/DemoFusion still leave per-tile error peaks (`TAC`) or cost much more memory/time than our sparse anchor-tile routing. |
| **Circularity ("Training to the Test") Risk** | Low (no new SR method, or SR fix is just a small application at the end). | **Zero if and only if the notional metric is simple, closed-form (`Std/Peaks of per-tile LPIPS/DISTS/r_i` + off-the-shelf VLM pair check), and never trained on our method!** |
| **Engineering & Compute Effort Before Mid-Nov** | High data curation effort (generating outputs from 5-6 tiled SR methods across 200+ 8K images, running ~6,000 VLM pair queries, running human 2AFC study, training & ablating the Tile-Graph predictor). | Moderate and modular: 20-50 8K test images are enough; zero metric training; method is training-free / plug-and-play inference (or lightweight adapter) on top of VOSR and one diffusion SR baseline. |
| **What Happens if Simple Fixes (Overlap + Global Color + Shared Noise) Already Fix 80% of the Problem?** | **Path A survives:** The paper reports the metric + benchmark and the surprising empirical finding that simple controls beat complex tiled architectures. | **Path B must pivot to Path A:** If simple Pipeline 1 controls already flatten the error peaks, a complex new SR architecture becomes unnecessary. |

---

## 6. Candid Opinion & Recommended Strategy

1. **Do NOT train a black-box neural metric predictor on 100-200 images as the first deliverable.**
   - Even with the correct Tile-Graph Relational Architecture (Section 3.2), a learned metric trained on 150 images will be a magnet for reviewer attacks unless backed by a massive human IQA campaign.
   - Worse, you don't even need a learned neural predictor when `Y` is available! The **closed-form GT-referenced quantities** (per-tile normalized residual `r_i`, per-tile `LPIPS/DISTS` spread `TES` & peak count `TAC`, plus GT-gated cross-tile Gram/DINO discrepancy `MPTS` and the direct **VLM Pairwise Consistency Fraction `S_pair`**) already measure both tile-to-GT consistency and all-pairs cross-tile consistency **directly, transparently, and with zero training overfitting**.
2. **Start on Path B (Method-Primary with the Closed-Form Notional GT Metric + VLM Pairwise Audit), using the exact 2-Stage Metric Formulation as Section 3 of the paper.**
   - Why this is the strongest CVPR paper shape:
     - **Section 3 (Diagnosis & Metric Suite):** Present the 10-20 image probe showing that global PSNR/LPIPS hides severe per-tile restoration peaks (`TAC`) and cross-tile texture flips on VOSR/SeedVR/CASR. Define the **closed-form GT-referenced consistency suite**:
       1. **Unary Tile-to-GT Inconsistency (`TES` & `TAC`):** Variance and anomaly peak count of per-tile normalized residual `r_i` (and per-tile `LPIPS/DISTS`) against `Y_i`.
       2. **Pairwise Cross-Tile Consistency (`MPTS` & `VLM-Pair`):** Over comparable tile pairs `(i, j) in E_comp` selected by the GT Affinity Graph `G_Y`, report both closed-form texture distribution divergence `|d_tex(Y_hat_i, Y_hat_j) - d_tex(Y_i, Y_j)|` and the **2x2 VLM Pairwise Agreement Fraction `S_pair`** (validated by a small 200-pair human check).
     - **Section 4 (Method):** Diagnose *why* the peaks and pair disagreements occur (local context starvation causing per-tile restoration regime switching + independent high-frequency hallucination across tiles) and introduce **Coarse-Anchored + Affinity-Routed Exemplar Tiled SR**.
     - **Section 5 (Experiments):** Show that our method **chops off the per-tile error peaks (`TAC` drops, `TES` drops, `VLM-Pair` rises)** while **improving or preserving mean per-tile fidelity (`Mean LPIPS/DISTS/r_i`)**, outperforming naive overlap/color controls, SWITi, and CASR at a fraction of CASR's all-to-all attention cost.
3. **Keep Path A (Learned Tile-Graph Metric Predictor) as our built-in safety pivot or No-Reference distillation module:**
   - If the initial experiments show that simple Pipeline 1 controls already fix the SR models, or if we want a **No-Reference deployment metric** that runs on wild images without `Y`, we train the **Tile-Graph Relational Network (Section 3.2)** taking `(Y_hat, Up(x))` to predict the GT-backed unary anomalies `a_i` and pairwise VLM labels `c_ij`.

---

## 7. Immediate Execution Plan (Step-by-Step Decision Gate)

### Phase 1: The 48-Hour Probe (Agreed in `sr_meeting_2026-10-17.md` & `sr_handoff.md`)
1. **Curate 15-20 8K images (`Y`)** spanning both failure modes:
   - ~10 images with large continuous objects/surfaces spanning multiple tiles where restoration regime switching happens (portraits with hats/clothing, foreground-background depth transitions, buildings).
   - ~10 images with repeated homogeneous stochastic textures across distant tiles (water, foliage, brick/stone, fabric).
2. **Run the fixed 2x-twice Real-ESRGAN degradation -> 2K `x` -> 4x4 grid of 512 tiles -> 4x SR (`VOSR` first, plus `SeedVR` or `CASR`) -> 8K `Y_hat`.** Save both stitched `Y_hat` and the 16 unblended `2048x2048` tile interiors.
3. **Compute the Unary Per-Tile Table (`16 tiles` per image):**
   - Standard per-tile `PSNR(Y_hat_i, Y_i)`, `LPIPS(Y_hat_i, Y_i)`, `DISTS(Y_hat_i, Y_i)`.
   - The 5 content-normalized coordinates `r_i` from `sr_handoff.md` (Restoration, Structure, Texture, Tone, Artifacts).
   - Compute `Mean`, `TES (Std / IQR)`, and `TAC (number of tiles > median + tau)`.
4. **Compute the Pairwise Cross-Tile Table (`E_comp` comparable pairs from `G_Y`):**
   - Build `G_Y` using DINOv2 cosine similarity (`> 0.75`) + spatial adjacency on `Y_1..Y_16`.
   - For `(i, j) in E_comp`, compute closed-form `|GramDist(Y_hat_i, Y_hat_j) - GramDist(Y_i, Y_j)|` and run the **2x2 VLM Pairwise Prompt** (`[Y_i, Y_j; Y_hat_i, Y_hat_j]`) to get `c_ij in {0, 1}` and soft logprob `p_ij in [0, 1]`.

### Phase 2: The Decision Gate After Phase 1
- **Check 1:** Do the visibly inconsistent tiles in `Y_hat` show up as clear peaks in `TAC / TES` (Unary Mode 1) and/or low `c_ij` on `G_Y` pairs (Pairwise Mode 2), while global full-image LPIPS/PSNR stays insensitive?
  - Once confirmed, **freeze the evaluation suite (`Mean`, `TES`, `TAC`, `MPTS`, `VLM-Pair`)**. You now have the empirical motivation figure of the paper.
- **Check 2:** Run **Pipeline 1 (Shared Noise + Global VAE GroupNorm + Overlap Blend + Wavelet Color Fix)** on the same 20 images.
  - **If Pipeline 1 leaves major `TAC` peaks and `VLM-Pair` failures in place (Expected):** Lock in **Path B (Method-Primary)** and build the **Global Coarse Anchor + Sparse Affinity-Routed Exemplar KV** method to crush those peaks.
  - **If Pipeline 1 already eliminates almost all `TAC` peaks and `VLM-Pair` failures:** Lock in **Path A (Metric-Primary)**, scale the dataset to 150-200 images, validate the VLM pairwise labels with a 200-pair human study, and train the **Tile-Graph Relational Network (Section 3.2)** as a fast No-Reference (`Y_hat, x`) consistency predictor.
