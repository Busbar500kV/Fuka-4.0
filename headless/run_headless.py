import argparse, os
from pathlib import Path
from fuka.runner import run_headless

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fuka/config_default.json")
    ap.add_argument("--data_root", default="data")
    args = ap.parse_args()

    Path(args.data_root).mkdir(parents=True, exist_ok=True)
    run_headless(args.config, args.data_root)
    print("Done. Data under:", args.data_root)