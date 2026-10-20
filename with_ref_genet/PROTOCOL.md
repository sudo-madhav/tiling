# GMCI protocol

Instruction for a model extending `with_ref_genet/`. The score is implemented in `gmci.py`. Do not fold it into `genet/wcsi.py` and do not change WCSI. WCSI answers a different question: consistency of a super-resolved image with its low-resolution input, with no ground truth.

## What the four numbers are for

A pixel residual or SSIM against the ground truth treats every high-frequency structure as locked to a coordinate. Foliage, weave, stone and skin are not. GMCI splits the ground-truth image into places where position is part of the object (a glint, a corner, a silhouette) and places where only the distribution is.

Report four things, not one:

| Output | Meaning |
| --- | --- |
| GMCI | 100 exp(−F / E_JND). 100 at zero energy. About 36.8 when the energies that exist average one JND. |
| rho_coherence | Fraction of the departure from the ground truth that lies in the smooth modes of the sheaf. Undefined when that departure is ~0. Near 1: the class left the master together. Near 0: the departures disagree. |
| energies | E_master, E_frust, E_seam, E_emerge, E_lr. Each is already in JND units. |
| phenotype per tile | clonal masterpiece, allometric polymorphism, somatic mutation, isolated clonal island. |

The phenotype axes are E_master (low if below 1) and E_frust (low if below 1). A standard full-reference metric cannot separate allometric polymorphism (left the master, but together) from an isolated clonal island (matched the master, neighbours did not).

## Inputs

`gmci_score(sr, gt, lr, scale=None, tile=None, overlap=None, tile_sr=None)`.

- `sr`, `gt`: full resolution. They are cropped to their common size.
- `lr`: the low-resolution input the model saw.
- `tile` is in LR pixels, matching WCSI. `tile_sr` is in full-resolution pixels and overrides it; it must divide the scale.
- Ground truth is the clean high-resolution image, not a second model output.

## How each spec object is computed

These are the choices that make the specification a function. They are deliberate.

**Rigidity γ.** Quadrature log-Gabor bank (even filter plus its Hilbert partner), 4 octaves from wavelength 3 px, 4 orientations. Phase is atan2(odd, even), so a step sits at one phase on every octave. γ is the amplitude-weighted agreement of that phase across octave pairs, with a noise floor at the median energy. The specification writes exp(i(φ_fine − 2 φ_coarse)) unweighted. The doubled phase is the phase-harmonic used in E_emerge; for rigidity the question is whether the phases agree, and a quadrature step has phase 0, so the two formulas agree on the edges rigidity is there to find. The median floor is what puts flat regions, noise and single-scale gratings at 0. γ is a property of the ground truth only.

**E_Wadd, positional.** CIEDE2000 between the coarse-grained SR tile and the coarse-grained ground-truth tile, multiplied by γ averaged over that LR block, divided by 1.5². A moved contour lands here. A moved texture, whose coarse image barely changes and whose γ is low, does not.

**E_Wadd, sliced Wasserstein.** Squared sliced Wasserstein-2 of the two Lab point clouds, 16 fixed-seed directions, divided by 1.5². This is the bimodal-fleck test: the amounts of each colour must match, and the positions of the flecks must not matter. Added to the positional term. A Gaussian Bures distance is what WCSI uses, because WCSI has no ground-truth cloud to transport.

**E_PCP.** Local nematic Q from the residual luminance. Pointwise Frank energy (Q, splay, bend), weighted by γ times the ground-truth order, so a locked contour must match the director in place. Plus the squared Wasserstein-1 of those same fields on the complementary weight (1 − γ) times order, so a texture is compared as a distribution. K0 = K1 = K3 = 1. One JND of bend or splay is a 4° director turn over 16 px. The vector used for splay is flipped to agree with the ground-truth director before differentiating; Q itself is headless and is not flipped.

**E_clock.** Sliced Wasserstein-1 between the windowed power spectra of the two residuals inside the band [1/(2s), 1/2], 12 directions, divided by 5% of the mean radial frequency, then squared. Plus the squared slope difference over 0.14². The band is the same generalisation WCSI uses; the specification's [1/8, 1/2] is this band at 8×.

**State ψ.** The same 14 numbers as WCSI, measured on the full-resolution tile and on the ground-truth tile directly. There is no κ. Both images are already at full resolution. Texture coordinates (Q, wavevector, slope) are set to zero departure when both residuals are below visibility 0.05, so an empty band cannot dominate.

**A\* and E_frust.** A is the identity on the Lab mean, the wavevector and the slope. On the six covariance entries it is the linear map Σ ↦ B Σ Bᵀ with B = (Σⱼ\* + εI)^(1/2) (Σᵢ\* + εI)^(−1/2), ε = 1.5², so a flat tile is not inverted. On Q it is the 2θ rotation by the ground-truth stroke-angle change from tile i to tile j. The edge residual is A δψᵢ − δψⱼ, divided by the JND of that coordinate, and the texture coordinates are multiplied by the smaller visibility. Neighbour edges have weight 1. Distant edges have the WCSI material affinity, computed on the ground-truth tiles. The energy is the weighted mean with one pseudo-pair of zero residual, the same prior WCSI uses so a class with no shared material is not graded on its single closest pair.

**rho_coherence.** Eigenvectors of that sheaf Laplacian (the quadratic form of the residuals above) with eigenvalue at most 5% of the largest, and at most 4 of them. Rho is the fraction of ||δΨ||² in those modes. A shared departure is the constant mode and scores near 1. A checkerboard of opposite departures is orthogonal to it and scores near 0.

**E_seam.** On a stitched image the two tiles share pixels, so the specification's overlap residual is identically zero. The term that is zero when SR equals the ground truth, and that uses γ, is: at each neighbour overlap, the colour step of SR minus the colour step of the ground truth at the same line, times γ, plus the gradient step excess, each divided by 1.5² plus the ground truth's own step. Median over rows, worst of the start, middle and end of the band. An edge the photograph already has is not a seam.

**E_emerge.** Phase-harmonic correlation of the luminance across the same octave pairs, real and imaginary parts, mean squared difference from the ground truth divided by 0.2². This is how fine structure sits on coarse structure, not where a particular leaf is.

**E_lr.** mean of max(0, CIEDE2000(coarse SR, LR)² − CIEDE2000(coarse ground truth, LR)²) / 1.5². The ground truth itself does not match a degraded LR. The penalty is only the extra break. It is zero when SR equals the ground truth, which is why a clone can still score near 100.

**GMCI.** F is the alpha-weighted sum of the energies that exist (default alphas all 1). E_JND is the sum of those alphas. A one-tile image has no frustration and no seam; they are left out and the divisor shrinks.

## Phenotypes

Per tile, E_master = E_Wadd + E_PCP + E_clock, and E_frust is the mean of the incident edge residuals.

| | E_frust < 1 | E_frust ≥ 1 |
| --- | --- | --- |
| E_master < 1 | clonal masterpiece | isolated clonal island |
| E_master ≥ 1 | allometric polymorphism | somatic mutation |

Thresholds are `phenotype_master` and `phenotype_frust`.

## What this directory does not do

It does not run VOSR, CASR or SeedVR, and it does not download DIV8K. Scoring a directory of triples belongs in `genet_ablation/` once someone points that harness at `gmci_score` instead of `wcsi_score`. Do that as a new script there, not by editing WCSI.

## Tests

`pytest -q with_ref_genet/test_gmci.py` builds tiny images. A clone scores high with a zero seam. One shared colour shift has high rho and low frustration; a checkerboard of opposite shifts has the reverse, with a real master cost in both. One unshifted tile inside a shifted class is an island; a shifted tile among shifted neighbours is allometric. Moving a hard edge costs more positional energy than rolling a noise texture by the same distance.
