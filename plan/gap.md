# Gap first — working order from 2026-10-13

The network in `plan/method_plan.md` and in `Template/sec/3_method.tex` is a **placeholder**.
It is not the next thing to build. Order, decided with the user:

1. **Now.** Crystallize the gap in current methods. This file.
2. **Parallel, later.** Survey relevant papers and list findings. Protocol in §4. Do not start it by inventing a method.
3. **Then.** Build a method from the user's ideas and from those findings. Entry condition in §5.
4. Writing the paper is not on the critical path. Ablations are last, after a method exists and works.

Judgement rules are Peng Sida's (`research-methodology`): the topic outranks the solver;
a cycle counts only if it verifies a technical cause; a paper is worth writing only if a reader
learns something new; do not stop at a result that merely looks acceptable.

## 1. The gap, as measured

Setting: one defocused photo in, one all-in-focus photo out, at the resolution the camera recorded.
Data: DPDD test, 76 scenes, 6720×4480, protocol `dpdd4`. Numbers from the paper tables
(`Template/sec/4_experiments.tex`) unless noted.

**The empty region.** No released pipeline is simultaneously as faithful as ×4 deblur + bicubic
and as textured as the non-learned same-image paste. That region is the green box in
`Template/fig/frontier.tex`. It is empty. Everything else we measured sits on one trade-off.

Three faces, all on the same 76 scenes.

### Identity

Running a released deblurrer at native resolution barely changes the photo.

| Method | Native PSNR | Δ vs input 23.06 | Δ at official 1680×1120 (input 23.93) |
|---|---|---|---|
| Restormer | 22.99 | −0.07 | +2.11 |
| LaKDNet-L | 23.00 | −0.06 | +2.38 |
| DRBNet | 23.12 | +0.06 | +1.86 |
| IFAN | 23.22 | +0.16 | +1.49 |
| Bokehlicious | 23.70 | +0.64 | +2.20 |

On our own ×4 rendering the same models gain +1.35 to +1.84 dB, and the ranking changes
(Bokehlicious leads; LaKDNet leads on the official files). Tile size 512 vs 1120 changes
native PSNR by ≤ 0.06 dB. TLC on Restormer changes nothing (−0.08 vs −0.07). The failure
is not the tile and not global test-time statistics.

### Ceiling

Deblur at ×4, then bicubic up to native, is the fidelity ceiling and looks worse than the blurry input.

| Anchor | Native PSNR | LPIPS ↓ | DISTS ↓ |
|---|---|---|---|
| Blurry input | 23.06 | 0.370 | 0.260 |
| DRBNet + bicubic | 24.58 | 0.573 | 0.292 |
| Bokehlicious + bicubic | **24.79** | 0.546 | 0.289 |

No learned or generative upsampler in the main table beats bicubic on PSNR or SSIM, with either anchor.
×2 deblur + bicubic up to native is worse than ×4, not better (handoff 3, all 76, native PSNR):
Restormer 23.69, DRBNet 23.84. The best operating scale among the ones we scored is ×4, not native
and not the midpoint.

### Invention

Generative upsamplers buy perception by changing the photograph, including where it was already sharp.
DRBNet anchor, paired PSNR change vs the input in the focal-plane bin: bicubic +0.44, VOSR +0.08,
S3Diff −0.99, OSEDiff −1.11. S3Diff on the Bokehlicious anchor is the perceptual extreme
(native PSNR 23.50, DISTS 0.255, MUSIQ 48.4) and is 1.29 dB below bicubic of the same anchor.

The non-learned paste (copy the input where it is sharp, and elsewhere add the high band of the
best-matching in-focus cell) reaches DISTS 0.232 within 0.2 dB of bicubic of the same anchor,
without the focal-plane damage. It marks the corner of the empty region. It does not fill it.
The matched-versus-random control is a four-scene pilot, not a 76-scene result, so "retrieval is
why the paste works" is not part of the gap statement.

## 2. What this gap is, and what it is not

**It is.** At native sensor resolution, released defocus deblurrers do not restore the photograph;
the faithful way to use them throws away native detail; the perceptual way invents it and rewrites
regions that were sharp. No released method occupies both the fidelity of the ×4 anchor and the
texture of real in-focus evidence from the same photo.

**It is not**, until a measurement says otherwise:

- "Defocus deblurring is unsolved." It is in good shape at 1680×1120. Abuolaim and Brown captured
  6720×4480 and evaluated at 1680×1120.
- "We need a stronger generative prior." That prior is the invention face.
- "Reference super-resolution closes it." Unmeasured here. ImageCritic (CVPR 2026) is a different
  task and already reports that ReFIR can worsen local detail. That goes in the survey (§4), not
  into this statement.
- "Training at native resolution cannot close it." **Unrun.** This is the hole that keeps the gap
  from being a fact about the task rather than a fact about released weights. See §3.

## 3. The one measurement that finishes the statement

**Restormer, finetuned on native DPDD crops**, then scored with the frozen protocol on all 76 scenes.
Spec already written: `docs/baselines.md` A1 (DPDD init, 512 native crops, AdamW 3e-4 cosine, 60k, L1,
paper tiling). One model. DRBNet (A2) and mixed-scale training (A3) stay off until this answers.

Binary, pre-registered:

- **Gap stands.** Native PSNR stays at least 1 dB under Bokehlicious ×4 + bicubic (24.79) and within
  about 1 dB of the blurry input (23.06). Released methods fail because native blur is outside the
  training distribution, and a straightforward native finetune does not remove the failure.
- **Gap collapses.** The finetune closes that distance. The topic sentence becomes "train at the
  test resolution." Stop. Do not replace it with an upsampler.

A 10k-step probe may be used only as a tripwire. If it already gains more than 1 dB over the released
Restormer at native resolution (22.99), finish the 60k. If it gains almost nothing, the 60k run is
still owed.

**Cause curve, only if the deblur-only points are missing.** We do not have "run the deblurrer at ×2
and score it at ×2" in the paper tables. ×2 + upsample-to-native is a different question and is
already answered (§1, ceiling). If `ours_x2` deblur-only metrics are not on disk under `dpdd4`,
run Restormer alone at ×2, score at 3360×2240, same 76 scenes. Plot gain over the input at ×4, ×2,
and ×1 for that one model.

- **Cause stands.** Gain falls as resolution increases.
- **Cause fails.** ×2 is already as bad as native. The abstract's "blur size grows with resolution"
  sentence is too smooth, and it gets rewritten before any method work.

No other baseline, no new upsampler, no ablation.

## 4. Survey protocol (later, in parallel with nothing that builds a model)

Purpose: a challenge-insight tree, which is a weapon list for our failure case. Not a reading list,
and not a tour of other people's tasks.

For each paper record only: the failure it actually demonstrates, the pipeline in one sentence,
the technical insight (the reason, not the module name), and which of our faces it touches
(identity / ceiling / invention / same-image evidence). Mark whether it was evaluated against
ground truth at native sensor resolution. If it was not, it does not close §2.

Read method sections, not whole papers. Verify the DOI before the insight is written down.
Walk one citation step backward and forward from DPDD (Abuolaim & Brown, ECCV 2020) and from
Bokehlicious (Seizinger et al., ICCV 2025) before adding papers from memory.

The tree must be able to answer, in four sentences:

- What is the end state of defocus deblurring at the resolution cameras record?
- What level have released methods reached, on which resolution?
- Which important failure is still open after §3?
- What is the current hotspot, and which of our faces does it not address?

The papers already inside this repository are not the survey. They are the measurements.
The reads below were retrieved outside the repo on 2026-10-13. None of them is a method we adopt.
Each card still has to say whether the paper scores against ground truth at native sensor
resolution. Checked once: the defocus papers below report the public DPDD release and RealDOF /
LFDOF / RTF, not 6720×4480. The UHD papers below restore motion blur, haze, or low light, not defocus.

**Identity — large blur and the wrong resolution**

- Zhai, Zeng, Ma, Zhao, Chen. Learnable Blur Kernel for Single-Image Defocus Deblurring in the Wild.
  arXiv:2211.14017. Read the CoC paragraph in the introduction and the multi-scale branch
  (downsample to 1/4 and 1/8 for large radius). They already say a larger circle of confusion
  needs a larger field. Record the resolution of the PSNR 25.56 number.
- Zhang, Foo, Rahmani, Liu, Soh. Performing Defocus Deblurring by Modeling its Formation Process.
  ICCV 2025. Read Figure 1 and Section 3.1–3.2 only. Blur is a large illuminated blob, and they
  shrink the blob. Record which datasets Table 1 and Table 2 actually use.
- Ren, Shu, Li, Wu, Li, Zuo. Reblurring-Guided Single Image Defocus Deblurring.
  arXiv:2409.17792, IJCV 2025. Read Figure 1 and the abstract. DPDD pairs are misaligned, and
  they train through a reblur so the loss does not sit on the shifted sharp photo. This is our
  1–2 px registration fact, published as a method. Record whether their test images are native.
- Feng, Zhou, Ye, Chen, Zhu. Residual Diffusion Deblurring Model for Single Image Defocus
  Deblurring. AAAI 2025. Read the paragraph that says existing methods fail on extensive,
  variable defocus, and the sentence that describes the pre-deblur module. Record the test
  resolution.

**Ceiling — people who refuse to downsample, on a different blur**

- Wu, Wang, Peng, Lu, Zheng. Dropout the High-Rate Downsampling. WACV 2025. D2Net.
  Read the first page. Their claim is that high-rate downsampling throws away the image, and
  their deblurring numbers are on UHD-Blur, which is motion blur. The card must say that this
  claim and our ceiling face disagree, and on which degradation.
- Zhao, Chen, Xu, Gu, Li, and coauthors. From Zero to Detail. CVPR 2025. ERR.
  Read Figure 3 and the three-stage paragraph: zero frequency, then low frequency, then a
  separate high-frequency stage at full resolution. Again UHD restoration, not defocus.
  The insight to keep is the split of bands, not the network.

**Invention — when a reference is allowed to speak**

- Wang, Wan, Zheng, Li, Hou, Jiang. Trust but Verify. ICLR 2026. arXiv:2602.01864. Ada-RefSR.
  Read the abstract and the paragraph that defines Adaptive Implicit Correlation Gating.
  The reference is used when it correlates with the input and suppressed when it does not.
  That is an abstain, measured on reference super-resolution, not on defocus.
- Bernasconi, Djelouah, Zhang, Gross, Schroers. ReBaIR: Reference-Based Image Restoration.
  ICCV Workshops. Read the first two pages and Figure 1. A frozen restorer stays frozen, and
  a second module writes reference features into its last maps. Their reference is another
  photograph (a close-up), not a sharp region of the same frame. Say whether the paper ever
  tries a same-image crop.

ImageCritic (CVPR 2026) and RefGC-SR² (arXiv:2606.15158) stay on the list as the two
generation papers already read: ReFIR can make a local detail error worse, and low-frequency
structure follows the degraded input while high-frequency statistics follow the reference.
Both use a clean external object photo, not a defocused scene.

## 5. Method work stays blocked until both of these exist

- §3 has a binary result, written into the handoff with the 76-scene numbers.
- §4 has the four sentences and a challenge-insight tree, each insight tied to a paper that was read.

When that is true, the method pass lists at least three pipelines for the core challenge
("who owns the native high band, and what stops it from rewriting structure"), including the
conservative one we already measured (×4 anchor + bicubic + non-learned paste). If that
conservative pipeline and a published method are tied, the failure case is not worth a new
network. The user's own ideas enter at that listing, next to the survey's insights, and one
pipeline is chosen. Ablations are designed after that pipeline works, not before.
