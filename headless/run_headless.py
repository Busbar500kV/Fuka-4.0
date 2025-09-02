from __future__ import annotations
import argparse
from pathlib import Path
from fuka.runner import run_headless


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    args = ap.parse_args()

    # expand & resolve
    cfg = str(Path(args.config).expanduser())
    dr  = str(Path(args.data_root).expanduser())
    run_headless(cfg, dr)


if __name__ == "__main__":
    main()