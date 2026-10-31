# ADDED: new file, not part of the upstream SR3 codebase.
"""Sections 7–8 and supplementary Run S analysis (filled after the full GPU run)."""  # ADDED

from __future__ import annotations  # ADDED

from sr3_ablation.io.jsonio import read_json  # ADDED
from sr3_ablation.reporting.sections.base import ReportContext, ReportSection  # ADDED
from sr3_ablation.runs import run_name  # ADDED

SUPPLEMENTARY_COMMAND = "python -m sr3_ablation finetune --run S  (supplementary, optional)"  # ADDED


def _metric(run: str, key: str, context: ReportContext) -> float | None:  # ADDED
    path = context.run_paths(run).metrics_summary  # ADDED
    if not path.is_file():  # ADDED
        return None  # ADDED
    data = read_json(path)  # ADDED
    mean = data.get("mean", {})  # ADDED
    value = mean.get(key)  # ADDED
    return float(value) if value is not None else None  # ADDED


class QualitativeAnalysisSection(ReportSection):  # ADDED
    """Section 7 prose; numbers come from ``test_metrics.json`` when present."""  # ADDED

    placeholder = "QUALITATIVE_ANALYSIS"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        run_a = _metric(run_name("A"), "psnr", context)  # ADDED
        run_b = _metric(run_name("B"), "psnr", context)  # ADDED
        run_0 = _metric(run_name("0"), "psnr", context)  # ADDED
        bicubic = _metric("bicubic", "psnr", context)  # ADDED
        if None in (run_a, run_b, run_0, bicubic):  # ADDED
            scores = (  # ADDED
                "> *Test-set PSNR numbers will be inserted after "  # ADDED
                "`evaluate --run 0/A/B`; review the figures before submitting.*\n\n"  # ADDED
            )  # ADDED
        else:  # ADDED
            gap_a = run_a - bicubic  # ADDED
            scores = (  # ADDED
                f"Run A reaches **{run_a:.2f} dB** on the test set ({gap_a:+.2f} dB vs bicubic "  # ADDED
                f"at {bicubic:.2f} dB), Run 0 scores **{run_0:.2f} dB**, and Run B reaches "  # ADDED
                f"**{run_b:.2f} dB**.\n\n"  # ADDED
            )  # ADDED
        crossing = context.original.first_t_snr_below_one()  # ADDED
        alpha_last = context.modified.summary().alpha_bar_last  # ADDED
        parts = [  # ADDED
            "### 7.1 Which run won",  # ADDED
            "",  # ADDED
            scores  # ADDED
            + "Run A should win on PSNR. Check validation and test grids to confirm the gap "  # ADDED
            "at epoch 12 and at the final checkpoint.",  # ADDED
            "",  # ADDED
            "### 7.2 Fine-tuning loss",  # ADDED
            "",  # ADDED
            "Compare `loss.csv` for Runs A and B. Under $\\theta_t^{x}$ the per-step noise "  # ADDED
            "stays small, so Run B's mean training loss should sit well above Run A's.",  # ADDED
            "",  # ADDED
            "### 7.3 Why Run B's samples collapse",  # ADDED
            "",  # ADDED
            f"Run B was trained with $\\bar\\alpha'_T \\approx$ {alpha_last:.4f} while sampling "  # ADDED
            "still starts from $\\mathcal{N}(0, I)$. Expect saturated, noise-like outputs.",  # ADDED
            "",  # ADDED
            "### 7.4 Separating the bad start from the weak denoiser",  # ADDED
            "",  # ADDED
            "See the start-state diagnostic figure (`diagnose-start --run B`).",  # ADDED
            "",  # ADDED
            "### 7.5 Inference trajectories",  # ADDED
            "",  # ADDED
            "Compare Run A and Run B at $t \\in \\{2000, 1500, 1000, 500, 0\\}$. Run A should "  # ADDED
            f"show a noise-to-structure transition near $t = {crossing}; Run B's states should "  # ADDED
            "barely change.",  # ADDED
            "",  # ADDED
            "### 7.6 What fine-tuning fixed in Run 0",  # ADDED
            "",  # ADDED
            "Run 0 should look like generic FFHQ faces on colonoscopy frames; Run A should "  # ADDED
            "restore mucosa texture and polyp edges.",  # ADDED
            "",  # ADDED
            "### 7.7 Texture, vessels, polyp edges and where Run A still fails",  # ADDED
            "",  # ADDED
            "[TEAM: one or two sentences per theme after reviewing `test_comparison.png` and "  # ADDED
            "checkpoint grids.]",  # ADDED
        ]  # ADDED
        return "\n".join(parts)  # ADDED


class TakeawaySection(ReportSection):  # ADDED
    """Section 8 closing paragraph."""  # ADDED

    placeholder = "TAKEAWAY"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        run_a = _metric(run_name("A"), "psnr", context)  # ADDED
        run_b = _metric(run_name("B"), "psnr", context)  # ADDED
        bicubic = _metric("bicubic", "psnr", context)  # ADDED
        alpha_bar = context.modified.summary().alpha_bar_last  # ADDED
        if None in (run_a, run_b, bicubic):  # ADDED
            gap = "[TEAM: dB gap vs bicubic after evaluate]"  # ADDED
            run_b_text = "[TEAM: Run B test PSNR]"  # ADDED
        else:  # ADDED
            gap = f"{run_a - bicubic:.2f} dB"  # ADDED
            run_b_text = f"{run_b:.2f} dB"  # ADDED
        return (  # ADDED
            f"SR3 is extremely sensitive to its noise schedule. Raising $\\theta_t$ to the power "  # ADDED
            f"{context.exponent:.2f} keeps every step a valid variance, yet it shrinks the "  # ADDED
            f"end-of-schedule noise until $\\bar\\alpha_T \\approx$ {alpha_bar:.4f}, which is "  # ADDED
            f"enough to turn a model that beats bicubic by {gap} into one that scores about "  # ADDED
            f"{run_b_text}. The deciding quantity is the terminal SNR the transform produces, "  # ADDED
            f"not the symbol written in code ($\\beta_t$ vs $\\bar\\alpha_t$)."  # ADDED
        )  # ADDED


class SupplementaryAnalysisSection(ReportSection):  # ADDED
    """Explains supplementary Run S ($\\bar\\alpha_t^{x}$) vs declared Run B ($\\beta_t^{x}$)."""  # ADDED

    placeholder = "SUPPLEMENTARY_ANALYSIS"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        run_s = _metric(run_name("S"), "psnr", context)  # ADDED
        run_b = _metric(run_name("B"), "psnr", context)  # ADDED
        body = (  # ADDED
            "### 7.8 Supplementary Run S ($\\bar\\alpha_t^{x}$, not part of the submission zip)\n\n"  # ADDED
            "Our Phase 1 declaration fixes $\\theta_t = \\beta_t$, so **Run B** applies "  # ADDED
            f"$\\beta_t^{{{context.exponent:.2f}}}$, which *reduces* per-step noise and pushes "  # ADDED
            f"$\\bar\\alpha'_T$ up to about {context.modified.summary().alpha_bar_last:.4f}. "  # ADDED
            "Teammate code that raises **$\\bar\\alpha_t$** to the same power instead adds noise "  # ADDED
            "at every step (because $\\bar\\alpha_t^{x} < \\bar\\alpha_t$), keeps "  # ADDED
            "$\\bar\\alpha'_T$ near 0, and can still train a usable model — that is Run S in our "  # ADDED
            "repo, mirroring the high Run B scores seen when $\\bar\\alpha$ is powered rather "  # ADDED
            "than $\\beta$.\n\n"  # ADDED
        )  # ADDED
        if run_s is None:  # ADDED
            return body + (  # ADDED
                "> *Run S metrics appear after `finetune --run S` and `evaluate --run S`.*\n"  # ADDED
            )  # ADDED
        compare = ""  # ADDED
        if run_b is not None:  # ADDED
            compare = (  # ADDED
                f"On the same test split Run S reached **{run_s:.2f} dB** PSNR vs "  # ADDED
                f"**{run_b:.2f} dB** for declared Run B.\n"  # ADDED
            )  # ADDED
        return body + compare  # ADDED


class ContributionTableSection(ReportSection):  # ADDED
    """Section 9 contribution table (from the agreed group split)."""  # ADDED

    placeholder = "CONTRIBUTION_TABLE"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        rows = [  # ADDED
            "| Member | Roll number | Responsible for | Signature | Date |",  # ADDED
            "|---|---|---|---|---|",  # ADDED
            (  # ADDED
                "| Rikin Patel | M26AI1096 | Alternate schedule experiments, paired statistics, "  # ADDED
                "frequency/LPIPS plots in the teammate repo | | |"  # ADDED
            ),  # ADDED
            (  # ADDED
                "| Balasubramanian M | M26AI1030 | Phase 1 declaration, schedule derivations, "  # ADDED
                "review | | |"  # ADDED
            ),  # ADDED
            (  # ADDED
                "| Sonalika Sharma | M26AI1120 | Dataset split/preprocessing aligned with the "  # ADDED
                "declaration, baseline metrics, report prose | | |"  # ADDED
            ),  # ADDED
            (  # ADDED
                "| Anshul | M26AI1020 | Integrated pipeline (`sr3_ablation`), tests, report "  # ADDED
                "builder, submission zip, supervisor | | |"  # ADDED
            ),  # ADDED
        ]  # ADDED
        return "\n".join(rows)  # ADDED
