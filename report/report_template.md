<!-- ADDED: report template. `python -m sr3_ablation report` fills every double-brace placeholder from
     outputs/ and writes report/build/report.{md,html,pdf}. Text marked [TEAM: ...] must be
     written by the team after the full runs; delete the markers before submitting. -->

# Noise Schedule Ablation in SR3: $\theta_t$ vs $\theta_t^{x}$ on Kvasir-SEG

**CSL7860 Foundation Models and Generative AI · IIT Jodhpur · Instructor: Dr. Angshuman Paul**

**Group:** Rikin Patel (M26AI1096) · Balasubramanian M (M26AI1030) · Sonalika Sharma (M26AI1120) · Anshul (M26AI1020)

## 1. Task and Model

We are doing 8× super-resolution on colonoscopy images. We hand the model a 16×16 RGB crop of a colonoscopy frame that has a polyp in it, and we want the 128×128 version of that frame back. We make the 16×16 input by bicubically downsampling the 128×128 ground truth, then stretch it back up to 128×128 and pass that in as the conditioning image, which is the format SR3 expects. We picked this task because what a doctor looks at in these frames is the texture of the mucosa and the edge of the polyp, and those are the first things to go once the resolution drops.

The model is SR3 (Saharia et al., *Image Super-Resolution via Iterative Refinement*, IEEE TPAMI 2022). The authors did not put out code with the paper, so we work from Janspiry's PyTorch implementation, pinned to commit `01d27a7`, and the checkpoint it publishes for 16→128, `I640000_E37_gen.pth`, which was trained on FFHQ. SR3 is a conditional DDPM in pixel space: at every step its UNet gets the conditioning image concatenated with $x_t$ and the continuous noise level $\sqrt{\bar\alpha}$, and predicts the noise $\epsilon$.

- Paper: [https://arxiv.org/abs/2104.07636v2](https://arxiv.org/abs/2104.07636v2)
- Code we build on, at commit `01d27a7`: [https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/tree/01d27a7](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/tree/01d27a7)

## 2. Dataset and Verification

We use Kvasir-SEG (Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset*, MMM 2020): 1,000 colonoscopy frames, each with at least one polyp, stored as RGB JPEGs that run from 332×487 up to 1920×1072. Every frame also comes with a hand-drawn outline of the polyp, but those masks are for segmentation and our task does not use them. The licence covers research and education, so we do not ship the images with our code.

- Dataset page: [https://datasets.simula.no/kvasir-seg/](https://datasets.simula.no/kvasir-seg/)
- The zip our code downloads: [https://datasets.simula.no/downloads/kvasir-seg.zip](https://datasets.simula.no/downloads/kvasir-seg.zip)

Before splitting we hash every file and drop exact copies, since the same frame turning up in both training and test would quietly flatter our scores. Kvasir-SEG has no official split and no classes, so we shuffle what is left with seed 42 and take 100 frames for test, 100 for validation and the rest for training:

{{DATASET_TABLE}}

We resize each frame with bicubic interpolation and a centre crop, to 128×128 for the target and 16×16 for the input, and then stretch the input back to 128×128. That is the same recipe as upstream `data/prepare_data.py`. Many frames have a small black box in the bottom-left corner that belongs to the endoscope's display rather than the tissue, and we left it alone: cropping it would mean guessing its size frame by frame, and all three runs see exactly the same processed images anyway. Our code writes the full manifest to `outputs/dataset/manifest.csv`.

To show that Kvasir-SEG was not part of the pretraining we take route (a), the stated pretraining data, and three separate sources agree on it.

The config used for training gives the training set as `"name": "FFHQ"` on line 17 with `"dataroot": "dataset/ffhq_16_128"` on line 19, and names the validation set `"CelebaHQ"` on line 29:
[https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/01d27a7/config/sr_sr3_16_128.json#L17-L29](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/01d27a7/config/sr_sr3_16_128.json#L17-L29)

The README describes the checkpoint we downloaded as "16×16 → 128×128 on FFHQ-CelebaHQ" on line 24, and asks the reader to prepare FFHQ and CelebA-HQ and nothing else on lines 99 to 100:
[https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/01d27a7/README.md?plain=1#L99-L100](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/01d27a7/README.md?plain=1#L99-L100)

The paper says the same on page 5, right column, lines 11 to 13: "training face super-resolution models on Flickr-Faces-HQ (FFHQ) and evaluating on CelebA-HQ":
[https://arxiv.org/pdf/2104.07636v2](https://arxiv.org/pdf/2104.07636v2)

None of them mentions Kvasir-SEG, or any medical images at all. The two domains also have close to nothing in common. FFHQ and CelebA-HQ are aligned photographs of human faces in ordinary light, while our frames are wide-angle shots from inside the colon, lit only by the lamp on the scope, full of pink and red mucosa, bright specular reflections, blood vessels and mucus, with no face anywhere in them.

## 3. Schedule Analysis

SR3's native noise parameter is $\theta_t = \beta_t$, the per-step variance of the forward process

$$q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I\big), \qquad \bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s), \qquad \mathrm{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t},$$

with a linear schedule from $\beta_1 = 10^{-6}$ to $\beta_T = 10^{-2}$ and $T = 2000$. Every $\beta_t$ lies in $(0, 1)$, so nothing needs rescaling before we apply the power. Run B uses $\theta'_t = \beta_t^{x}$ with $x = {{EXPONENT}}$.

Which way this moves is easy to work out. Since $0 < \beta_t < 1$ and $x > 1$, we get $\beta_t^{x} < \beta_t$ at every step, so each factor $1 - \beta_t^{x}$ is bigger, and that makes $\bar\alpha'_t > \bar\alpha_t$ and $\mathrm{SNR}'(t) > \mathrm{SNR}(t)$ at **every** timestep. The modified forward process destroys far less of the signal.

{{SCHEDULE_TABLE}}

The end of the schedule is where this really bites. Under $\theta_t$ we get $\bar\alpha_T \approx 4\times10^{-5}$, so $x_T$ is essentially pure noise and lines up with the $\mathcal{N}(0, I)$ that sampling starts from. Under $\theta_t^{x}$ we get $\bar\alpha'_T \approx 0.62$, so $x_T = 0.78\,x_0 + 0.62\,\epsilon$ still carries most of the image and the SNR never drops below 1. Sampling still starts from pure noise, though, which means Run B begins from a state its fine-tuned model never saw while training, and the noise level $\sqrt{\bar\alpha'_t}$ it did see only covers $[0.78, 1]$ instead of $[0.007, 1]$.

Here is the SNR at each of the timesteps we save:

{{TRAJECTORY_SNR_TABLE}}

{{SCHEDULE_FIGURE}}

That gives us three things to check against our results. Run B's training loss should come out below Run A's, simply because predicting noise is easier when the SNR is high. Run B's reverse process should struggle to clear the initial noise, because each step's variance $\beta_t^{x}$ is tiny, so we expect leftover noise or colour shifts and worse PSNR and SSIM than Run A. And Run B's early trajectory states should barely move, while Run A should go from noise to structure somewhere around $t \approx 500$, where its SNR crosses 1.

## 4. Experimental Setup

{{SETUP_TABLE}}

All of our data comes from the Kvasir-SEG zip and goes through the cleaning, split and resizing in Section 2; we do not filter anything else out. At each checkpoint epoch we super-resolve a fixed set of validation frames and record PSNR and SSIM, which is how we keep an eye on training, and we always evaluate the final checkpoint. We touch the test split once per run, for that final evaluation. Runs A and B start from the same checkpoint and see the same data in the same order with the same seeds, and the noise schedule is the only thing that differs between them, in training and in sampling. Run 0 is the pretrained checkpoint used as it comes, with the original schedule. Each test batch reuses the same sampling seed across runs, so all three start a given image from identical noise.

Environment: {{ENVIRONMENT}}

## 5. Implementation

All of our code sits in one public repository: [https://github.com/A-Dev-Kit/FM-GAI-Assignment](https://github.com/A-Dev-Kit/FM-GAI-Assignment)

We vendored the upstream code in `third_party/sr3/`. The **only** change we made to it is a single line in `model/sr3_modules/diffusion.py`, inside `set_new_noise_schedule`, marked `# ADDED`:

```python
betas = betas ** schedule_opt.get('exponent', 1.0)  # ADDED: noise-schedule ablation, theta_t -> theta_t ** x
```

Every buffer that comes off `betas` is computed after this line, including `sqrt_alphas_cumprod_prev`, which is what feeds the continuous noise level into the UNet in training and in sampling, so they all pick up the modified schedule. Everything else is new code under `src/sr3_ablation/`, and every file there opens with an `# ADDED` header. We write the exponent down once, as `NOISE_EXPONENT_X = {{EXPONENT}}` at the top of `scripts/finetune.py`.

- The file we changed: [https://github.com/A-Dev-Kit/FM-GAI-Assignment/blob/main/third_party/sr3/model/sr3_modules/diffusion.py](https://github.com/A-Dev-Kit/FM-GAI-Assignment/blob/main/third_party/sr3/model/sr3_modules/diffusion.py)
- What we changed and how to reproduce it: [https://github.com/A-Dev-Kit/FM-GAI-Assignment/blob/main/third_party/sr3/UPSTREAM.md](https://github.com/A-Dev-Kit/FM-GAI-Assignment/blob/main/third_party/sr3/UPSTREAM.md)
- Where $x$ is declared: [https://github.com/A-Dev-Kit/FM-GAI-Assignment/blob/main/scripts/finetune.py](https://github.com/A-Dev-Kit/FM-GAI-Assignment/blob/main/scripts/finetune.py)

Three details decide whether this comparison means anything at all.

We only take weights from the checkpoint. We load the UNet weights (`denoise_fn.*`) and install the schedule *afterwards*. Upstream's `load_network` would also restore the optimiser, the step counters and the schedule buffers saved inside the checkpoint, and that would quietly undo the ablation.

We also do not take SR3's word for the schedule. Our own `NoiseSchedule` recomputes $\theta_t^{x}$, $\bar\alpha_t$ and the SNR in float64 without going near SR3, and we compare it against the buffers we read back off the model before fine-tuning, after fine-tuning, and once more before sampling (`schedule.json` and `schedule_sampling.json`):

{{SCHEDULE_EVIDENCE}}

And sampling happens in batches. Our reverse sampler calls upstream `p_sample` for $t = T, \dots, 1$ and tells its observers about each state $x_t$, while the trajectory recorder keeps the timesteps we asked for.

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
