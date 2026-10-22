# with_ref_genet

**Superseded by WCSI-2 with reference** (`genet/wcsi2.py`, `wcsi2_ref_score`; see `genet/WCSI2.md`):

```bash
python genet/wcsi2.py ref SR.png GT.png LR.png --scale 8 --tile-sr 512 [--device cuda:0]
```

GMCI kept WCSI v1's zero-difference targets between tiles and took 5–6 minutes per 33 MP
image. WCSI-2 with reference uses the same per-tile description of the invented band as the
no-reference score, compares it with the ground-truth tile, and runs in about 35 s per
33 MP image on one CPU process, or on a GPU. It keeps GMCI's useful outputs:
`rho_coherence` and per-tile phenotypes ("faithful", "consistent departure",
"inconsistent departure", "faithful among inconsistent peers"). `gmci.py` stays for
reproducing earlier numbers and as the `C_pixel_gmci` arm in `with_encoder/`.

Ground-truth Morphogenetic & Cohomological Index (GMCI). Unlike WCSI, this score sees the high-resolution ground truth as well as the low-resolution input.

```bash
python with_ref_genet/gmci.py SR.png GT.png LR.png --tile 64 --scale 8
```

`tile` is in low-resolution pixels. `--tile-sr 512` is the same grid quoted in full-resolution pixels.

Read `PROTOCOL.md` before changing the mathematics. The clone / shared-shift / checkerboard / island tests are in `test_gmci.py`.
