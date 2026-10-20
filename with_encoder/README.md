# with_encoder

Hybrid pixel + frozen-encoder versions of WCSI (no reference) and GMCI (with ground truth)
for tiled ×4 / ×8 generative super-resolution. Nothing is trained and nothing is downloaded:
learned encoders load only from local files you point to. A weight-free Gabor encoder runs
everywhere: it is the CPU test double and the "untrained features" control arm.

Read `DESIGN.md` for what the encoders are used for and why pixels stay, and `ABLATION.md`
for the experiments.

| file | role |
|---|---|
| `hybrid_ref.py` | score with ground truth: GMCI pixel terms + Nyquist probe + gated feature/latent fidelity + DINOv2 semantic distance + feature sheaf (`rho_feature`) |
| `hybrid_noref.py` | score without reference: WCSI pixel terms + aliasing probe + DINOv2 material graph + feature sheaf on the SR-minus-bicubic upgrade |
| `terms.py` | every term (space, band, source) and every ablation configuration |
| `engine.py`, `featstats.py` | per-tile statistics, PCA bases, Bures–Wasserstein, Gaussian OT transport, sheaf frustration and coherence |
| `encoders/` | encoder interface, the Gabor control, local-only DINOv2 / VGG-19 / ConvNeXt / VAE adapters, config registry |
| `corruptions.py` | controlled tiled-SR failure modes made from ground truth |
| `ablate.py` | every arm on the corruptions; summary with catch ratios, ISR and coherence |
| `score_methods.py` | real outputs (bicubic, VOSR, CASR, SeedVR) from the `genet_ablation` config |
| `configs/encoders.example.json` | encoder stack, method-encoder conflicts, alternative encoders |
| `sh/` | launchers (not run here): check encoders, make corruptions, ablate, score methods |

Quick use:

```bash
python with_encoder/hybrid_ref.py SR.png GT.png LR.png --scale 8 --tile-sr 512 --encoders gabor
python with_encoder/hybrid_noref.py SR.png LR.png --scale 8 --tile 64 --encoders gabor
# learned encoders, after copying and editing the example config:
python with_encoder/hybrid_ref.py SR.png GT.png LR.png --scale 8 --tile-sr 512 \
    --encoders with_encoder/configs/encoders.json --method vosr --json out.json
python -m pytest -q with_encoder
```

Scores: with reference \(100\,e^{-F/\sum\alpha}\) (as GMCI); without reference
\(100/(1+F/\sum\alpha)\) (as WCSI). Energies are in nominal JND units and every term of every
configuration is returned, so arms can be recombined without rescoring. `genet/wcsi.py` and
`with_ref_genet/gmci.py` are used, not modified.
