# genet_ablation

WCSI scores for DIV8K at 4x and 8x: ground truth, bicubic, VOSR, CASR, SeedVR, plus the tile-size and sampler ablations in `PROTOCOL.md`.

The metric code is `genet/wcsi.py` and is not modified here. Nothing in this directory runs a super-resolution model or downloads data.

1. Copy `config.example.json` to `config.json` and set the directories you already have.
2. Read `PROTOCOL.md` before changing or running anything.
3. The shells in `sh/` are the entry points. They are not run as part of setup.

```bash
python3 genet_ablation/inventory.py --config genet_ablation/config.json
python3 genet_ablation/score_sweep.py --config genet_ablation/config.json --dry-run
```
