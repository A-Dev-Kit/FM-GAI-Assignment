# Vendored upstream: SR3 (unofficial PyTorch implementation)

<!-- ADDED: provenance record for the vendored code. -->

| Field | Value |
|---|---|
| Repository | <https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement> |
| Commit | `01d27a7cbfa8502be1d8dbd4ee02fcbd5e44389d` (23 Dec 2022) |
| Licence | Apache License 2.0, see `LICENSE` in this folder |
| Paper | Saharia et al., *Image Super-Resolution via Iterative Refinement*, arXiv:2104.07636 |

## What was copied

Everything at that commit except `misc/` (README images) and `dataset/` (CelebA-HQ / FFHQ
sample images that are not part of this project). Files are otherwise byte-identical.

## What was changed

Exactly one line, in `model/sr3_modules/diffusion.py`, inside
`GaussianDiffusion.set_new_noise_schedule`, directly after the upstream schedule has been
converted to NumPy (upstream line 101):

```diff
@@ -99,6 +99,7 @@ class GaussianDiffusion(nn.Module):
             linear_end=schedule_opt['linear_end'])
         betas = betas.detach().cpu().numpy() if isinstance(
             betas, torch.Tensor) else betas
+        betas = betas ** schedule_opt.get('exponent', 1.0)  # ADDED: noise-schedule ablation, theta_t -> theta_t ** x
         alphas = 1. - betas
         alphas_cumprod = np.cumprod(alphas, axis=0)
         alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
```

Every buffer the model uses (`alphas_cumprod`, `sqrt_alphas_cumprod_prev`, the posterior
coefficients, and therefore the continuous noise level fed to the UNet) is derived from `betas`
after this line, so training and sampling both see the modified schedule. With no `exponent`
key the behaviour is identical to upstream.

Reproduce the diff with:

```bash
git clone https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement upstream
git -C upstream checkout 01d27a7cbfa8502be1d8dbd4ee02fcbd5e44389d
git diff --no-index upstream/model/sr3_modules/diffusion.py third_party/sr3/model/sr3_modules/diffusion.py
```

All other project logic lives in `src/sr3_ablation/`; only
`src/sr3_ablation/model/sr3_backend.py` imports from this folder.
