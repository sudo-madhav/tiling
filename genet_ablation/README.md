# genet_ablation

WCSI scores for DIV8K at 4x and 8x: ground truth, bicubic, VOSR, CASR, SeedVR, plus the tile-size and sampler ablations in `PROTOCOL.md`. `--metric` picks WCSI v1 (`genet/wcsi.py`, default), WCSI-2 without reference or WCSI-2 with reference (`genet/wcsi2.py`, see `genet/WCSI2.md`); `--devices cuda:0,cuda:1,...` runs WCSI-2 with one worker per GPU.

The metric code is in `genet/` and is not modified here. Nothing in this directory runs a super-resolution model or downloads data.

1. Copy `config.example.json` to `config.json` and set the directories you already have.
2. Read `PROTOCOL.md` before changing or running anything.
3. The shells in `sh/` are the entry points. They are not run as part of setup.

```bash
python3 genet_ablation/inventory.py --config genet_ablation/config.json
python3 genet_ablation/score_sweep.py --config genet_ablation/config.json --dry-run
python3 genet_ablation/score_sweep.py --config genet_ablation/config.json --metric wcsi2_ref --dry-run
GPUS=cuda:0,cuda:1 bash genet_ablation/sh/score_wcsi2.sh
```
