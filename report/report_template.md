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

with a linear schedule from $\beta_1 = 10^{-6}$ to $\beta_T = 10^{-2}$ and $T = 2000$. The schedule increases strictly with $t$, so every $\beta_t$ lies in $[\beta_1, \beta_T] = [10^{-6}, 10^{-2}] \subset (0, 1)$, and nothing needs rescaling before we apply the power. Run B uses $\theta'_t = \beta_t^{x}$ with $x = {{EXPONENT}}$. A positive power of a number in $(0, 1)$ stays in $(0, 1)$, so $\theta'_t$ is a valid per-step variance too, running from {{MODIFIED_BETA_T}} at $t = T$ down to {{MODIFIED_BETA_FIRST}} at $t = 1$.

Which way this moves is easy to work out. Since $0 < \beta_t < 1$ and $x > 1$, we get $\beta_t^{x} < \beta_t$ at every step, so each factor $1 - \beta_t^{x}$ is bigger, and that makes $\bar\alpha'_t > \bar\alpha_t$ and $\mathrm{SNR}'(t) > \mathrm{SNR}(t)$ at **every** timestep. The modified forward process destroys far less of the signal.

How much less follows from a short approximation. While the total noise added is still small, $1 - \bar\alpha_t \approx \sum_{s \le t} \theta_s$. Our $\beta_t$ is close to $\beta_T\, t / T$, so $\sum_{s \le t} \beta_s^{x} \approx \frac{T \beta_T^{x}}{x + 1} \left(\frac{t}{T}\right)^{x+1}$, which gives

$$\mathrm{SNR}'(t) \approx \frac{x + 1}{T\, \beta_T^{x}} \left(\frac{T}{t}\right)^{x+1}.$$

So under $\theta_t^{x}$ the SNR falls off only as a power of $t$, $t^{-(x+1)}$, and its lowest value, at $t = T$, is set by $T \beta_T^{x}$. For our numbers the formula gives about {{SNR_POWER_LAW_T}} at $t = T$, against an exact {{MODIFIED_SNR_T}}. Every extra unit of $x$ multiplies that floor by about $1/\beta_T = 100$. The approximation does not hold for the original schedule near $T$, because there the noise is no longer small: its SNR crosses 1 at $t = {{ORIGINAL_SNR_CROSSING}}$ and then keeps falling towards zero.

{{SCHEDULE_TABLE}}

The end of the schedule is where this really bites. Under $\theta_t$ we get $\bar\alpha_T \approx$ {{ORIGINAL_ALPHA_BAR_T}}, so $x_T$ is essentially pure noise and lines up with the $\mathcal{N}(0, I)$ that sampling starts from. Under $\theta_t^{x}$ we get $\bar\alpha'_T = {{MODIFIED_ALPHA_BAR_T}}$, so $x_T = {{XT_SIGNAL_COEF}}\,x_0 + {{XT_NOISE_COEF}}\,\epsilon$ is still almost the clean image, and the SNR never drops below 1. Sampling still starts from pure noise, though, so Run B begins its reverse process from a state its fine-tuned model has never seen.

Run B's model also loses its sense of time. SR3 tells the UNet the noise level through $\gamma = \sqrt{\bar\alpha}$, and `PositionalEncoding` in `model/sr3_modules/unet.py` turns it into $\sin(\gamma\,\omega_k)$ and $\cos(\gamma\,\omega_k)$ with frequencies $\omega_k \le 1$. Under $\theta_t$, training sees $\gamma$ anywhere in $[{{ORIGINAL_NOISE_LEVEL_MIN}}, 1]$. Under $\theta_t^{x}$ it only sees $[{{XT_SIGNAL_COEF}}, 1]$, so the embedding of every timestep sits within {{EMBEDDING_SHIFT}} radians of the embedding of a clean image. In effect, Run B's UNet is trained at a single, almost noise-free level.

Here is the SNR at each of the timesteps we save:

{{TRAJECTORY_SNR_TABLE}}

{{SCHEDULE_FIGURE}}

Three consequences follow, and each of them is something we can check against our results.

Start with the training loss. Under $\theta_t^{x}$ the noise is never more than {{XT_NOISE_COEF}} of $x_t$, so the UNet has to pick a faint noise field out of fine mucosal texture. For $\epsilon$-prediction the low-noise steps are the hard ones, so Run B's loss should sit at or above Run A's rather than below it, even though its inputs look cleaner.

The samples should be worse still. The reverse step estimates the clean image as $\hat x_0 = (x_t - \sqrt{1 - \bar\alpha'_t}\,\hat\epsilon) / \sqrt{\bar\alpha'_t}$, and with $\sqrt{1 - \bar\alpha'_t} \le {{XT_NOISE_COEF}}$ and $\sqrt{\bar\alpha'_t} \ge {{XT_SIGNAL_COEF}}$ that estimate is almost $x_t$ itself whatever the network predicts. Run B's chain can therefore hardly remove the noise it starts from, which should leave us with noisy outputs, colours saturated by the clipping to $[-1, 1]$, and PSNR and SSIM far below Run A's, probably below plain bicubic upsampling.

The trajectories should show the same thing from the inside. Run A ought to go from noise to structure somewhere around $t \approx {{ORIGINAL_SNR_CROSSING}}$, where its SNR crosses 1, while Run B's states should barely change between $x_{2000}$ and $x_0$.

## 4. Experimental Setup

{{SETUP_TABLE}}

All of our data comes from the Kvasir-SEG zip and goes through the cleaning, split and resizing in Section 2; we do not filter anything else out. At each checkpoint epoch we super-resolve a fixed set of validation frames and record PSNR and SSIM, which is how we keep an eye on training, and we always evaluate the final checkpoint. We touch the test split once per run, for that final evaluation. Runs A and B start from the same checkpoint and see the same data in the same order with the same seeds, and the noise schedule is the only thing that differs between them, in training and in sampling. Run 0 is the pretrained checkpoint used as it comes, with the original schedule. Each test batch reuses the same sampling seed across runs, so all three start a given image from identical noise.

Environment: {{ENVIRONMENT}}

## 5. Implementation

All of our code sits in one public repository: [https://github.com/A-Dev-Kit/FM-GAI-Assignment](https://github.com/A-Dev-Kit/FM-GAI-Assignment)

We vendored the upstream code in `third_party/sr3/`. The **only** change we made to it is a single line in `model/sr3_modules/diffusion.py`, inside `set_new_noise_schedule`, marked `# ADDED`. The table puts the upstream lines and ours side by side, with line numbers from both files:

{{CODE_DIFF}}

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

### 6.1 Run 0: the pretrained model on our inputs

Before any fine-tuning, we ran the FFHQ checkpoint as it comes, with its original schedule, on evenly spaced test images. Each row shows the 16×16 input, its bicubic upsampling (the conditioning image), the Run 0 output and the 128×128 ground truth.

{{PRETRAINED_FIGURE}}

### 6.2 Fine-tuning loss

Both runs log the mean training loss of every epoch to their own `loss.csv` (columns `epoch, train_loss`), and the plot shows the two curves on the same axes.

{{LOSS_FIGURE}}

{{LOSS_TABLE}}

{{VALIDATION_FIGURE}}

### 6.3 Samples at the checkpoint epochs

At each of the five equally spaced checkpoint epochs, both runs super-resolve the same fixed validation frames. Files are named `<run>_epoch<NN>_sample<k>.png`.

{{SNAPSHOT_FIGURES}}

The same samples, with Run A and Run B on adjacent rows, so the epoch where the two first part ways is easy to find:

{{CHECKPOINT_COMPARISON_FIGURE}}

### 6.4 Inference trajectories

Using the final model of each run, we saved the noisy sample $x_t$ at five equally spaced timesteps of the reverse process for three test images. $x_{1600}$ is the first saved state after the start at $x_{2000}$, and $x_0$ is the output. Files are named `<run>_t<tttt>_sample<k>.png`.

{{TRAJECTORY_FIGURE}}

### 6.5 Test-set scores

We rank the runs by PSNR, with SSIM alongside: every input has a pixel-aligned 128×128 ground truth, PSNR measures directly how far each output is from it, and SSIM checks that local structure such as polyp edges and mucosal texture survives. Every score is the mean ± standard deviation over the same test images, and bicubic upsampling is included as a floor.

{{RESULTS_TABLE}}

{{COMPARISON_FIGURE}}

## 7. Analysis

### 7.1 Which run won

Run A is the better run by a wide margin, and the gap is already there at the first checkpoint. At epoch 12 Run B's validation samples are saturated colour noise, and they are still that at epoch 60: its validation PSNR is 7.20 dB at epoch 12 and 7.18 dB at epoch 60. Run A's samples look like endoscopy frames at every checkpoint, and its validation PSNR climbs from 22.60 dB to 24.00 dB. On the test set Run A reaches 23.80 dB, 1.26 dB above bicubic upsampling, against 7.33 dB for Run B. All three of the checks we set up at the end of Section 3 came out the way the schedule analysis said they would.

### 7.2 Fine-tuning loss

Run B's mean training loss over the 60 epochs is 0.487, roughly ten times Run A's 0.047, and it hardly moves: 0.494 over the first ten epochs, 0.488 over the last ten. For comparison, always predicting $\hat\epsilon = 0$ would give an L1 loss of $\mathbb{E}|\epsilon| = \sqrt{2/\pi} \approx 0.80$, so Run B's UNet does recover part of the noise, just not much of it. The reason is the one Section 3 gives. Under $\theta_t^{x}$ the noise is never more than {{XT_NOISE_COEF}} of $x_t$, so the UNet is hunting for a faint noise field inside fine mucosal texture, and its noise-level input barely changes from one timestep to the next. Run A's loss is flat too, for the opposite reason: the FFHQ checkpoint already predicts noise well under its own schedule, so fine-tuning only has to adapt it to the new image statistics.

### 7.3 Why Run B's samples collapse

Two things go wrong at once, and they compound. Run B was only ever trained on $x_T = {{XT_SIGNAL_COEF}}\,x_0 + {{XT_NOISE_COEF}}\,\epsilon$, an SNR of {{MODIFIED_SNR_T}}, but sampling starts from $\mathcal{N}(0, I)$, whose SNR is 0, so the chain opens in a state the model has never seen. It then has no way to recover. Every step estimates $\hat x_0 = (x_t - \sqrt{1-\bar\alpha'_t}\,\hat\epsilon)/\sqrt{\bar\alpha'_t}$ with $\sqrt{1-\bar\alpha'_t} \le {{XT_NOISE_COEF}}$ and $\sqrt{\bar\alpha'_t} \ge {{XT_SIGNAL_COEF}}$, so $\hat x_0 \approx x_t$ whatever the network predicts, and the posterior variance of a single step is at most $\theta'_T = $ {{MODIFIED_BETA_T}}.

The chain therefore carries its starting noise all the way down to $x_0$ almost untouched, and clipping to $[-1, 1]$ for display turns that noise into the saturated colours in the grids. A PSNR of 7.3 dB, with a standard deviation of only 0.37 dB over 100 test images, is about what clipped Gaussian noise scores against almost any image. That is far below bicubic, which is what we expected.

### 7.4 Separating the bad start from the weak denoiser

A bad starting point and a denoiser that cannot denoise would both produce noise, so we ran one more experiment to tell them apart. We sampled Run B's final model a second time on the three trajectory images, with the same seed, but started it from the distribution it was actually trained on at $t = T$: $x_T = {{XT_SIGNAL_COEF}}\,\tilde x_0 + {{XT_NOISE_COEF}}\,\epsilon$, with the bicubic image $\tilde x_0$ standing in for the unknown $x_0$. That start uses the conditioning image twice, so it is not a fair competitor to the other runs and we keep it out of the results table. It is here only to show where Run B's chain goes wrong.

{{START_DIAGNOSTIC}}

Started this way the same model returns an image scoring 22.34 dB, which is essentially the 22.29 dB of the bicubic image it began from, and it looks like that image too: a slightly denoised bicubic upsampling with no new fine detail. So Run B's UNet does remove the small amount of noise it was trained on. What its chain cannot do is return anything much different from what it was handed. From pure noise that is noise; from a near-clean start it is the blurry conditioning image. Fixing the start alone would not turn Run B into a super-resolution model, because its schedule leaves no noisy steps in which detail could be generated.

### 7.5 Inference trajectories

Run A's states are noise at $x_{1600}$ and $x_{1200}$ (SNR 0.0016 and 0.028). The coarse layout, the overlay box and the bright regions appear at $x_{800}$ (SNR 0.25), the frame is clearly recognisable at $x_{400}$ (SNR 2.0), and $x_0$ is clean. That is the transition around the SNR crossing at $t = {{ORIGINAL_SNR_CROSSING}}$ that the table in Section 3 points to. Run B's five states all look alike, saturated noise from $x_{1600}$ right through to $x_0$. Under its own schedule $x_{1600}$ should already have an SNR of 381, so its reverse chain never visits the states its schedule describes.

### 7.6 What fine-tuning fixed in Run 0

The FFHQ checkpoint used as it comes scores 21.74 dB, 0.80 dB *below* bicubic, and its outputs show why: they are sharp, but they look like faces. Mucosa comes out as smooth, softly shaded skin with streaks and wrinkle-like creases. The many small specular highlights of wet tissue are mostly missing or replaced by a few large, smooth ones. Colours drift towards brown and grey, and in some frames folds turn into dark, hair-like strands (test_00083 in the comparison grid). Fine-tuning with the original schedule fixes most of this. Run A gains 2.06 dB over Run 0, restores the small highlights, keeps polyp outlines crisp, and even redraws the white marks of the endoscope's on-screen overlay. Its SSIM (0.627) only matches bicubic (0.629), though: the fine texture Run A generates is plausible but not pixel-aligned with the ground truth, and SSIM penalises any such invented detail.

### 7.7 Texture, vessels, polyp edges and where Run A still fails

For Run B there is nothing to compare, because no texture, vessel or edge survives in any of its test outputs. Run A's failures are milder and of the kind generative super-resolution from a 16×16 input usually has. The granular surface of polyps comes out smoother than in the ground truth. Thin dark vessels are drawn where the low-resolution input suggests them rather than where they really are, and small red spots on polyps go missing. Specular highlights are plausible but not in their true positions.

## 8. Takeaway

SR3 is extremely sensitive to its noise schedule. Raising $\theta_t$ to the power {{EXPONENT}} keeps every step a valid variance, yet it shrinks the end-of-schedule noise until $\bar\alpha_T \approx$ {{MODIFIED_ALPHA_BAR_T}}, and that alone turns a model that beats bicubic by 1.26 dB into one that outputs pure noise. The deciding quantity is not $\theta_t$ itself but the terminal SNR it produces. A power transform is only advisable if $T$ or $\beta_T$ is raised as well, so that $\bar\alpha_T$ is still close to 0 and sampling from $\mathcal{N}(0, I)$ still matches the end of the forward process. With $T = 2000$ and $\beta_T = 10^{-2}$ fixed, any $x > 1$ pushes $\bar\alpha_T$ up, and by $x = {{EXPONENT}}$ it is almost 1.

## 9. Contribution Statement

Each of us signs below to confirm that the table describes who was responsible for which part of the work, and that we all read and agreed to the final report.

| Member | Roll number | Responsible for | Signature | Date |
|---|---|---|---|---|
| Rikin Patel | M26AI1096 | [TEAM: contribution] | | |
| Balasubramanian M | M26AI1030 | [TEAM: contribution] | | |
| Sonalika Sharma | M26AI1120 | [TEAM: contribution] | | |
| Anshul | M26AI1020 | [TEAM: contribution] | | |
