# Frozen encoders for WCSI and GMCI: decisions

WCSI (`genet/`, no reference) and GMCI (`with_ref_genet/`, with ground truth) work in
pixels. This directory adds frozen, off-the-shelf encoders where they buy something,
keeps pixels where encoders are blind, and makes every choice an ablation arm. Nothing is
trained and nothing is downloaded.

## The answer

Yes, an encoder helps, in both settings, but as an addition, not a replacement:

1. **Which tiles should look alike** (the material graph). Biggest expected gain. WCSI and
   GMCI decide "same material" from Lab colour statistics and band energies, which confuse
   blue sky with a blue wall and split one material under a lighting change. DINOv2
   descriptors fix exactly that. Used in both settings: in GMCI the graph comes from the
   ground truth tiles, in WCSI from the bicubic-upsampled LR (never from the SR, so a method
   cannot move the graph it is judged by).
2. **Texture statistics compared as distributions**. Mid-level VGG-19 features (Gatys'
   texture network) carry far more texture information than the 14-D pixel state. They are
   compared by Bures–Wasserstein between per-tile Gaussians, never position by position,
   except where the ground truth is phase-locked.
3. **Semantic fidelity with a reference**: DINOv2 token distance per tile catches invented
   or changed content.
4. **Latent of an autoencoder (VAE)**: optional, the most redundant (a VAE latent is a
   compressed pixel space) and the most bias-prone (see D9). It is an ablation arm, on by
   default only because it is cheap.

"Global semantics" enters through (1). The graph is global: every tile is compared with every
tile. A single image-level embedding distance (CLS token, CLIP score) is not used. It is one
number per image, cannot localise a tiling failure, and barely moves between SR outputs of
the same image.

Pixels keep three bands no frozen encoder sees reliably: block-mean colour against the LR,
1-px seams, and the Nyquist octave.

## Decisions

**D1. Keep pixel terms for the ultra-low band, 1-px seams and the Nyquist octave.**
`pix_lr` (GMCI E_lr) / `nr_rg` (WCSI E_RG) for colour after D_s block averaging,
`pix_seam` / `nr_seam` for seam steps, `pix_nyq` / `nr_nyq` for the top octave. The Nyquist
probe is one-sided: it counts excess top-octave energy and aliasing peaks (ringing,
checkerboards), not missing high frequencies. Blur belongs to the mid-band feature terms.
A first version fitted the top-octave slope; on bicubic outputs the octave is nearly empty,
the fit is ill-conditioned, and that one term (about 950 JND) saturated the score at 0.
The proposal's three reasons for keeping pixels, corrected:
- Colour invariance comes from colour-jitter training. It is strong for DINOv2 and CLIP. VGG
  early layers stay colour-sensitive. So "encoders miss colour drift" is an empirical claim
  per encoder, tested by corruptions F2 and F7, not assumed.
- Stride: agreed. 1-px steps average out inside 8–14 px receptive patches.
- Nyquist: "encoder Jacobian nullspace" is too strong. Strided convolutions alias rather than
  null, and a VAE that reconstructs must encode high frequencies. We keep the pixel Nyquist
  probe because it is exact and cheap, not because encoders are provably blind.

**D2. Partitioned hybrid by default, union as an arm.** `F_hybrid` (with reference) uses
pixel terms only for the three bands above, and features for the mid band and the non-local
structure. `F_union` adds every GMCI pixel term too. If the union is not better, the
partition loses nothing.

**D3. Gate = pixel phase-congruency rigidity of the ground truth, pooled to each feature
grid.** The proposal's gate (sigmoid of z-scored latent gradient + CLS attention) is kept
only as the `gate=feat` arm (without the attention part). Gradient magnitude is high on
texture as well as on contours, so it conflates the two cases the gate exists to separate.
CLS attention marks salient objects, not phase-locked structure (a cat's fur would be
"rigid"). Arms `none` (γ = 0, pure distributional) and `all` (γ = 1, LPIPS-like) bracket it.

**D4. All feature distances live in a per-image, per-layer PCA basis (k = 16) with
Ledoit–Wolf shrinkage.** Full-channel Bures–Wasserstein (C = 256–1536) with a few hundred
tokens per tile is rank-deficient and costs a C×C matrix square root per tile. The basis is
fitted on anchor + test tokens of up to 16 evenly spaced tiles, so all tiles of an image
share coordinates. One JND = 20 % of the image's RMS feature spread in that layer
(`jnd_feat`, nominal).

**D5. Sheaf transport = Gaussian optimal-transport map between anchor tiles, on tile
statistics, not Procrustes on token matrices.** The proposed
\(\mathrm{SVD}((\tilde F_j^*)^\top \tilde F_i^*)\) pairs token k of tile i with token k of tile j,
but positions in two different tiles do not correspond. The cross-covariance is then
mostly noise and the rotation is arbitrary. Procrustes is right only when tokens correspond
(SR vs GT of the same tile), and that is not a sheaf edge. The proposed residual
\(T(\tilde F_i-\tilde F_i^*)-(\tilde F_j-\tilde F_j^*)\) also subtracts token matrices position by
position, which brings back the pointwise penalty the Wasserstein term was meant to remove.
We use the feature analogue of GMCI's 14-D state: mean and spread on the top `k_rho = 4` PCA
axes per texture/latent layer. The map is the linear part of the Monge map between the
anchor tiles' Gaussians on the means, and the ratio of anchor spreads on the spreads.

**D6. The transport is shrunk towards identity (positive-part James–Stein), and residuals
are widened by the sampling error of the tile statistics** (as WCSI widens its spectral
terms). Found while building this: without the shrinkage, the ground truth itself scored
about 11 JND of no-reference frustration. Between two tiles of one material, the OT map
differs from identity by about \(1/\sqrt{n}\), and multiplied by a large departure that reads
as inconsistency.

**D7. Without a reference, the sheaf state is the upgrade \(\psi(\mathrm{SR})-\psi(U(\mathrm{LR}))\),
and an edge needs both the same material and similar bicubic-anchor statistics**
(`tau_anchor` = 1 JND RMS, the feature-space version of WCSI's colour/band kernel). Found
while building this: on a scene with a sharp bar crossing a striped material, the bar tiles
legitimately gain more from SR than plain stripe tiles. With the semantic graph alone,
the ground truth scored 19 JND, against 48 for a real within-material clash. With the
anchor kernel it scores 4.2 against 50. `tau_anchor` was set on that one synthetic scene
and is swept in ABLATION.md.
**Limitation: the no-reference feature sheaf measures consistency, not quality.** Bicubic
scores exactly 0 (no upgrade is a perfectly consistent upgrade). Report it next to a quality
measure; WCSI's own E_RG already plays that role partly.

**D8. Default encoders.** Texture: VGG-19 `relu2_2, relu3_4, relu4_4` (strides 2, 4, 8). It
is the Gatys texture network and the backbone family of LPIPS/DISTS, so the comparison with
those baselines isolates how features are compared rather than which features are used.
Semantic: DINOv2 ViT-L/14 with registers, last layer. Registers remove the high-norm
artefact tokens that would dominate pooled descriptors. ViT-g costs about 4× for a graph,
so it is an arm. Latent: FLUX.1 16-channel VAE (stride 8) from the Apache-2.0 schnell
release (the dev release is non-commercial). Alternatives (ConvNeXt, SD3.5 VAE, a mid
DINOv2 layer, the weight-free Gabor control) are listed in
`configs/encoders.example.json`.

**D9. An encoder a method was built on never judges that method unflagged.** A latent
encoder from the method's own autoencoder family drops the latent terms for that method; any
other role conflict is flagged in the output and needs a cross-check run with another
encoder for that role. From the VOSR README: VOSR-1.4B decodes through the 16-channel
Qwen-Image VAE, VOSR-0.5B through the SD2.1 VAE, and DINOv2 weights ship with the release.
So DINOv2-based terms are flagged for VOSR and must be cross-checked with a non-DINOv2
semantic encoder (ConvNeXt stage 3 in the semantic slot). SeedVR trains its own causal video
VAE. The CASR entry stays empty until we know which CASR it is.

**D10. Material affinity = centred cosine.** DINOv2 descriptors of unrelated tiles have high
raw cosine because of a shared image-wide component. The image-mean descriptor is
subtracted first. `tau_sem` = 0.3 is nominal; distant edges are capped at the 8 strongest per
tile.

**D11. No resizing to fit an encoder.** Tiles are reflect-padded to the encoder's multiple
and the feature grid is cropped back (repo convention). DINOv2 therefore runs at native tile
resolution (a 512 px tile gives 36×36 tokens). At 8K native resolution a 14 px token sees
material, not objects. That is what the graph needs, and it means "semantic" here means
material identity. Object-level semantics would need a downscaled pass; not done.

**D12. JNDs are nominal and all weights are 1.** Nothing is fitted. Fitting weights on human
2AFC is a later step that needs a new study, because the public sets (BAPPS 64×64 patches,
PIPAL small patches) contain no tiling. They can calibrate only the per-tile fidelity terms.

**D13. A weight-free control encoder.** `encoders/gabor.py` (complex-cell Gabor energies plus
opponent colour, no files) is both the CPU test double and an ablation arm. If a learned
encoder does not beat it, the gain came from multi-scale filtering and distributional
comparison, not from what the network learned.

## Term map

| term | setting | space | band | encoder | replaces / adds |
|---|---|---|---|---|---|
| `pix_lr` / `nr_rg` | ref / noref | pixel | ultra-low (D_s, CIELAB) | — | kept from GMCI / WCSI |
| `pix_seam` / `nr_seam` | ref / noref | pixel | 1-px seam | — | kept |
| `pix_nyq` / `nr_nyq` | ref / noref | pixel | Nyquist octave | — | new one-sided probe (excess energy + aliasing peaks) |
| `feat_wadd` | ref | feature | mid (2×–8×) | VGG-19 | adds to GMCI E_wadd: gated pointwise + per-tile Bures–Wasserstein |
| `lat_wadd` | ref | latent | mid (8×) | VAE | same on the latent; dropped on conflict |
| `feat_pcp` | ref | feature | mid | VGG-19 | feature nematic tensor + feature spectrum W1 (band above LR Nyquist on that grid) |
| `feat_sem` | ref | semantic | 14 px tokens | DINOv2 | new: hallucinated / changed content |
| `feat_frust` | ref | feature | non-local | VGG-19 + DINOv2 graph | feature version of GMCI E_frust, plus `rho_feature` |
| `nr_cyto_dino` | noref | pixel | non-local | DINOv2 graph | WCSI E_cyto with the graph swapped |
| `nr_feat_sheaf` | noref | feature | non-local | VGG-19 + DINOv2 graph | new: upgrade consistency, plus `rho_feature` |
| `feat_point` | ref | feature | mid | VGG-19 | baseline only: ungated pointwise (an LPIPS analogue) |

Configurations (arms) are in `terms.py`: `B_pointwise_feature`, `C_pixel_morphogen` (= GMCI),
`D_feature_latent`, `E_feature_sheaf`, `F_hybrid` (default), `F_union`, and leave-one-out
arms `F_minus_*`. Without a reference: `N0_wcsi`, `N1_wcsi_dino_graph`, `N2_feature_sheaf`,
`N3_hybrid` (default), `N4_union`.

## What we did not adopt from the proposal, and why

- **"Milliseconds per image."** No. For a 33 MP image (8× of a ~0.5 MP LR) with 512 px tiles
  there are about 127 tiles, each encoded about 2.1 times (SR, anchor, plus the basis
  sample). DINOv2-L at 512² is roughly 0.4 TFLOP and VGG-19 to relu4_4 roughly 0.1 TFLOP,
  so the encoders take on the order of seconds per image per A100 in fp16. The CPU pixel
  terms (GMCI) dominate wall time. Budget minutes per image across the arms, not
  milliseconds.
- **Procrustes on token matrices and position-wise token residuals** (D5).
- **Latent-gradient + CLS-attention gate as the default** (D3).
- **"Universally applicable to medical / satellite."** DINOv2 and VGG are trained on natural
  images; transfer is unverified. Claim natural images only.
- **The ✅/❌ table.** It is a set of hypotheses; ABLATION.md turns it into measurements.
- **"Beats LPIPS and FID" as novelty.** Comparing texture statistics in a way that tolerates
  resampling is what DISTS does (Ding et al., TPAMI 2020). A Fréchet distance between the
  feature tokens of a single image is SIFID (Shaham et al., SinGAN, ICCV 2019). Both are prior
  art and DISTS must be a baseline. Our additions are the rigidity gate, the cross-tile
  sheaf with its transport, and the graph.
- **The ISR as written.** Its ratio explodes for arms invariant to F1, and it needs F1 to be
  a valid resample. We compute it on energies per arm and per image, report medians, and
  define F1 to change only content above the LR Nyquist frequency (see ABLATION.md). Found
  while building this: a naive 3 px shift of the high-pass residual changed the LR by ~10
  grey levels on near-Nyquist stripes, so it was not a valid resample.

## Cost and memory

Two passes per image: a basis pass on at most 16 tiles, then one pass over all tiles that
encodes test and anchor once each. Only per-tile statistics are kept, never full feature
maps. `rho_feature` needs a dense eigendecomposition of (tiles × state dimension). It is
skipped with a note above `rho_max_n` = 6000; at 512 px tiles on 33 MP with three VGG layers
(24 coordinates), that is about 250 tiles.

## Open items

- GMCI on the synthetic scene: E_seam reads about 8 JND for uniform above-Nyquist sharpening,
  and E_master is very large for a valid resample of near-Nyquist stripes. Check both on
  DIV8K before relying on `C_pixel_morphogen` or `F_union`.
- `tau_anchor`, `tau_sem` and `jnd_feat` are nominal (set on synthetic scenes); sweep them.
- The VOSR, SeedVR and CASR conflict entries need checking against the checkpoints actually
  run.
- The learned-encoder adapters (`encoders/local_models.py`) are not exercised on CPU; run
  `sh/check_encoders.sh` once on the GPU machine.
