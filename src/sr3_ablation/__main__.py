# ADDED: new file, not part of the upstream SR3 codebase.
"""Entry point for ``python -m sr3_ablation``."""  # ADDED

from sr3_ablation.cli import main  # ADDED

if __name__ == "__main__":  # ADDED
    raise SystemExit(main())  # ADDED
