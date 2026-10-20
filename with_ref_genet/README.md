# with_ref_genet

Ground-truth Morphogenetic & Cohomological Index (GMCI). Unlike WCSI, this score sees the high-resolution ground truth as well as the low-resolution input.

```bash
python with_ref_genet/gmci.py SR.png GT.png LR.png --tile 64 --scale 8
```

`tile` is in low-resolution pixels. `--tile-sr 512` is the same grid quoted in full-resolution pixels.

Read `PROTOCOL.md` before changing the mathematics. The clone / shared-shift / checkerboard / island tests are in `test_gmci.py`.
