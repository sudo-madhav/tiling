# Secure-env runbook — handoff 7: close the gap statement

**One runbook per handoff.** Replaces the handoff-6 runbook (ReFIR / iRAG tile study). That study
is paused. The method in `plan/method_plan.md` is a placeholder. Read `plan/gap.md` first.

Audience: the Claude agent in the secure environment. The user runs the training and the deblur
commands by hand. You check each step and report. No new model, no ablation, no paper edit beyond
what the report needs.

## Why
The gap in released methods is measured (`plan/gap.md` §1) except for one hole: nobody has
finetuned a DPDD deblurrer on native crops and scored it on the 76. Until that number exists,
the gap may be "wrong training resolution" rather than "native blur is out of distribution."
A second, smaller hole: deblur-only at ×2, scored at ×2, is not in the paper tables. ×2 followed
by upsampling to native is already measured and is not this question.

## Standing rules
- Test set: all 76 scenes, protocol `dpdd4`. Val is for nothing in this handoff except that you
  do not touch the test set while training.
- One new training run: Restormer only (`docs/baselines.md` A1). Do not start DRBNet, mixed-scale
  training, ReFIR, iRAG, or `ours_v0`.
- Report outcomes and deviations. No internal paths, hostnames, or proprietary names.

## 1. Restormer finetuned at native resolution
Spec: DPDD-initialized Restormer, 512-pixel native crops from the 350 training scenes, AdamW
3e-4 with cosine, 60k steps, L1, batch 8. Score the released checkpoint and the finetuned
checkpoint with the same paper tiling (1120 / overlap 140) on `ours_x1`, metrics `dpdd4`.

A 10k-step probe is allowed only as a tripwire. If native PSNR is already more than 1 dB above
the released Restormer (22.99), continue to 60k. If it is not, still finish 60k and say so.

**Binary, do not reinterpret.**
- Gap stands: finetuned native PSNR ≤ 23.79 (at least 1 dB under 24.79) and ≤ 24.06 (within
  about 1 dB of the input 23.06).
- Gap collapses: anything else. Quote the number. Do not propose an upsampler in the report.

## 2. Deblur-only at ×2, only if it is not already on disk
Look for Restormer outputs on `ours_x2` scored at 3360×2240 under `dpdd4` (not ×2 followed by
an upsampler). If they exist, paste the mean PSNR and the paired gain over the ×2 input.
If they do not, run `restormer_dpdd` on `ours_x2` with its default paper tiling and score it
at that resolution, 76 scenes.

Plot, in the report, gain over the input at three scales for Restormer only:
×4 (ours rendering, +1.37 dB, PSNR 25.60 vs input 24.24), ×2 (your number), ×1 (released
−0.07 dB; plus the finetuned number from §1 on the same axes, marked as finetuned).

- Cause stands if the released-model gain falls from ×4 to ×2 to ×1.
- Cause fails if ×2 is already within 0.2 dB of ×1. Say so in one sentence.

## 3. What to report (`handoff/from_secure/<date>_h7.md`)
- The 76-scene table: input, released Restormer, finetuned Restormer. Native PSNR, SSIM, LPIPS,
  DISTS, focal-plane ΔPSNR, and PSNR at ×4. Paired bootstrap CIs.
- Which side of the §1 binary the finetune landed on.
- The three-scale gain plot as a table, and which side of the §2 binary it landed on.
- Deviations from A1 (steps actually run, crop size, batch, learning rate).
- Do not include a method proposal.

`handoff/make_patch.sh`.
