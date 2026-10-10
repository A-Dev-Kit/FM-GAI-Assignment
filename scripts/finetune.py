# ADDED: new file, not part of the upstream SR3 codebase.
"""Fine-tune the pretrained SR3 16->128 model on Kvasir-SEG for Run A or Run B.

    python scripts/finetune.py --run A        # theta_t (original schedule)
    python scripts/finetune.py --run B        # theta_t ** NOISE_EXPONENT_X

Optional: --config FILE (repeatable; default configs/base.toml). Outputs go to
outputs/runA or outputs/runB. NOISE_EXPONENT_X below is the single source of truth for x;
every other command reads it from this file.
"""

NOISE_EXPONENT_X = 1.60  # exponent x of the ablation, Run B uses theta_t ** x, x in (1.5, 3.0)


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    from sr3_ablation.config import load_config
    from sr3_ablation.pipelines.finetune import finetune
    from sr3_ablation.runs import run_spec

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, choices=["A", "B"])
    parser.add_argument(
        "--config", action="append", type=Path, help="TOML file(s), merged in order"
    )
    args = parser.parse_args(argv)
    results = finetune(load_config(args.config), run_spec(args.run, NOISE_EXPONENT_X))
    for result in results:
        print(f"epoch {result.epoch}: train_loss={result.train_loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
