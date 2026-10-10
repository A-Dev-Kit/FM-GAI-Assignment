<!-- ADDED: report template. `python -m sr3_ablation report` fills every double-brace placeholder from
     outputs/ and writes report/build/report.{md,html,pdf}. Text marked [TEAM: ...] must be
     written by the team after the full runs; delete the markers before submitting. -->

# Noise Schedule Ablation in SR3: $\theta_t$ vs $\theta_t^{x}$ on Kvasir-SEG

**CSL7860 Foundation Models and Generative AI · IIT Jodhpur · Instructor: Dr. Angshuman Paul**

**Group:** Rikin Patel (M26AI1096) · Balasubramanian M (M26AI1030) · Sonalika Sharma (M26AI1120) · Anshul (M26AI1020)

## 1. Task and Model

Our task is 8× super-resolution of colonoscopy images. The model gets a 16×16 RGB crop of a colonoscopy frame showing a polyp and has to produce the 128×128 version. We make the 16×16 input by bicubically downsampling the 128×128 ground truth, and then upsample it back to 128×128 to use as the conditioning image, which is the format SR3 expects. We chose this task because the things a doctor looks at in these frames, the texture of the mucosa and the edge of the polyp, are the first details to disappear at low resolution.

The model is SR3 (Saharia et al., *Image Super-Resolution via Iterative Refinement*, IEEE TPAMI 2022, [arXiv:2104.07636](https://arxiv.org/abs/2104.07636)). We work from the unofficial PyTorch implementation by Janspiry, pinned to commit [`01d27a7`](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/tree/01d27a7cbfa8502be1d8dbd4ee02fcbd5e44389d), and its released checkpoint `I640000_E37_gen.pth` for 16→128, trained on FFHQ. SR3 is a conditional DDPM in pixel space. At every step its UNet receives the conditioning image concatenated with $x_t$, together with the continuous noise level $\sqrt{\bar\alpha}$, and predicts the noise $\epsilon$.

## 2. Dataset and Verification

We use Kvasir-SEG (Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset*, MMM 2020), downloaded from [datasets.simula.no/kvasir-seg](https://datasets.simula.no/kvasir-seg/). It has 1,000 colonoscopy frames, each with at least one polyp, stored as RGB JPEGs whose sizes range from 332×487 to 1920×1072. The dataset also ships polyp masks, which we do not need. Its licence allows research and education use, so the images are not included in our code submission.

For cleaning, every file is hashed with SHA-1 before the split and exact copies are dropped, so that the same picture cannot end up in both training and test. Kvasir-SEG has no official split and no classes, so we shuffle the remaining frames with seed 42 and take 100 for test, 100 for validation and the rest for training:

{{DATASET_TABLE}}

Each frame is resized by bicubic interpolation with a centre crop, to 128×128 for the target and 16×16 for the input, and the input is then upsampled back to 128×128. This follows upstream `data/prepare_data.py`. Many frames carry a small black box in the lower-left corner, which comes from the endoscope's display and not from the tissue. We left it in, because cropping or masking it would mean guessing where it is in each frame, and since all three runs see exactly the same processed images it does not favour any of them. The full manifest is in `outputs/dataset/manifest.csv`.

To show that Kvasir-SEG was not used for pretraining we take route (a), the stated pretraining data, which three independent sources agree on:

- The training config, [`config/sr_sr3_16_128.json` lines 17–19](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/01d27a7cbfa8502be1d8dbd4ee02fcbd5e44389d/config/sr_sr3_16_128.json#L17-L19), gives the training set as `"name": "FFHQ"` with `"dataroot": "dataset/ffhq_16_128"`, and line 29 names the validation set `"CelebaHQ"`.
- [README line 24](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/01d27a7cbfa8502be1d8dbd4ee02fcbd5e44389d/README.md?plain=1#L24) describes the checkpoint as "16×16 → 128×128 on FFHQ-CelebaHQ".
- The paper, [arXiv:2104.07636v2](https://arxiv.org/pdf/2104.07636v2), page 5, right column, lines 11–13, reports "training face super-resolution models on Flickr-Faces-HQ (FFHQ) and evaluating on CelebA-HQ".

None of the three mentions Kvasir-SEG. The domains are also very far apart. FFHQ and CelebA-HQ are aligned photos of human faces in ordinary light, while Kvasir-SEG frames are wide-angle views from inside the colon, lit only by the scope's lamp, with pink and red mucosa, bright specular reflections, vessels and mucus, and nothing resembling a face.

## 3. Schedule Analysis

SR3's native noise parameter is $\theta_t = \beta_t$, the per-step variance of the forward process

$$q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I\big), \qquad \bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s), \qquad \mathrm{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t},$$

with a linear schedule from $\beta_1 = 10^{-6}$ to $\beta_T = 10^{-2}$ and $T = 2000$. Every $\beta_t$ lies in $(0, 1)$, so no rescaling is needed before the power transform. Run B uses $\theta'_t = \beta_t^{x}$ with $x = {{EXPONENT}}$.

The direction of the change is easy to pin down. Because $0 < \beta_t < 1$ and $x > 1$, we have $\beta_t^{x} < \beta_t$ at every step, which makes each factor $1 - \beta_t^{x}$ larger. It follows that $\bar\alpha'_t > \bar\alpha_t$ and $\mathrm{SNR}'(t) > \mathrm{SNR}(t)$ at **every** timestep: the modified forward process destroys far less signal.

{{SCHEDULE_TABLE}}

What happens at $t = T$ is the interesting part. Under $\theta_t$ we get $\bar\alpha_T \approx 4\times10^{-5}$, so $x_T$ is essentially pure noise and matches the $\mathcal{N}(0, I)$ start of sampling. Under $\theta_t^{x}$, however, $\bar\alpha'_T \approx 0.62$, meaning $x_T = 0.78\,x_0 + 0.62\,\epsilon$ still carries most of the image and the SNR never falls below 1. Sampling nonetheless starts from pure noise. Run B therefore begins from a state its fine-tuned model has never seen in training, and the noise-level input $\sqrt{\bar\alpha'_t}$ it was trained on only spans $[0.78, 1]$ rather than $[0.007, 1]$.

The SNR values at the trajectory timesteps we save are:

{{TRAJECTORY_SNR_TABLE}}

{{SCHEDULE_FIGURE}}

This leaves us with three predictions to test against the results. First, Run B's training loss should fall below Run A's, simply because noise is easier to predict at high SNR. Second, Run B's reverse process should be unable to clear the initial noise, since each step's variance $\beta_t^{x}$ is tiny, so we expect visible residual noise or colour shifts and worse PSNR and SSIM than Run A. Third, Run B's early trajectory states should barely change, whereas Run A should move from noise to structure around $t \approx 500$, where its SNR crosses 1.

## 4. Experimental Setup

{{SETUP_TABLE}}

The data come from the Kvasir-SEG zip and go through the cleaning, split and resizing described in Section 2; nothing else is filtered out. Validation PSNR and SSIM are computed on a fixed subset of the validation split at each checkpoint epoch, so we can watch how training is going; we always evaluate the final checkpoint, and the test split is used only once per run, for that final evaluation. Runs A and B start from the same checkpoint and use the same data order and seeds; the noise schedule, in both training and sampling, is their only difference. Run 0 applies the pretrained checkpoint unchanged with the original schedule. Because test outputs reuse the same sampling seed per batch for every run, all three runs start each image from identical initial noise.

Environment: {{ENVIRONMENT}}

## 5. Implementation

We vendored the upstream code in `third_party/sr3/`. The **only** change to it is a single line in `model/sr3_modules/diffusion.py`, inside `set_new_noise_schedule`, marked `# ADDED`:

```python
betas = betas ** schedule_opt.get('exponent', 1.0)  # ADDED: noise-schedule ablation, theta_t -> theta_t ** x
```

Every buffer derived from `betas` is computed after this line, including `sqrt_alphas_cumprod_prev`, which drives the continuous noise level in both training and sampling, so all of them pick up the modified schedule. Everything else is new code in `src/sr3_ablation/`, where each file opens with an `# ADDED` header. The exponent is declared once, as `NOISE_EXPONENT_X = {{EXPONENT}}` at the top of `scripts/finetune.py`.

Three implementation details matter for the validity of the ablation.

We load weights only. Just the UNet weights (`denoise_fn.*`) come from the checkpoint, and the schedule is installed *afterwards*. Upstream `load_network` would also restore the optimiser, the step counters and the schedule buffers stored inside the checkpoint, which would silently undo the ablation.

We verify the schedule independently. `NoiseSchedule` recomputes $\theta_t^{x}$, $\bar\alpha_t$ and SNR in float64 without going through SR3, and compares the result against the buffers read back from the model before and after fine-tuning, and once more before sampling (`schedule.json`, `schedule_sampling.json`):

{{SCHEDULE_EVIDENCE}}

Sampling runs in batches. A reverse sampler calls upstream `p_sample` for $t = T, \dots, 1$ and notifies observers of each state $x_t$, and the trajectory recorder keeps the timesteps we asked for.

## 6. Results

### 6.1 Training loss

{{LOSS_FIGURE}}

{{LOSS_TABLE}}

{{VALIDATION_FIGURE}}

### 6.2 Samples during fine-tuning

{{SNAPSHOT_FIGURES}}

### 6.3 Test set

{{RESULTS_TABLE}}

{{COMPARISON_FIGURE}}

### 6.4 Reverse trajectories

{{TRAJECTORY_FIGURE}}

## 7. Analysis

[TEAM: compare the results with the three predictions in Section 3. Address at least: (a) whether Run B's lower training loss translates into better or worse samples, and why (train/test mismatch at $x_T$, the narrower noise-level range seen in training); (b) what the trajectories show about when structure appears in Run A vs Run B, using the SNR table; (c) how much fine-tuning alone helps (Run 0 vs Run A), and whether Run 0 shows face-like priors on tissue, for example mucosa smoothed into skin or lost specular highlights; (d) how Runs A and B differ on mucosal texture, vessels and polyp edges, and any failure cases in the comparison grid, such as residual noise, colour shifts or over-smoothing.]

## 8. Takeaway

[TEAM: two or three sentences on what this ablation shows about how sensitive SR3 is to the noise schedule, and when raising $\theta_t$ to a power would or would not be advisable.]

## 9. Contribution Statement

| Member | Contribution |
|---|---|
| Rikin Patel (M26AI1096) | [TEAM: contribution] |
| Balasubramanian M (M26AI1030) | [TEAM: contribution] |
| Sonalika Sharma (M26AI1120) | [TEAM: contribution] |
| Anshul (M26AI1020) | [TEAM: contribution] |
