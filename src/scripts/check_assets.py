"""Check the files required by the historical train/serve commands.

This check verifies presence, not compatibility, model accuracy, or data rights.
"""
import argparse
from pathlib import Path


REQUIRED = {
    "train": ("PandA.db",),
    "serve": (
        "price_prediction_model.keras", "preprocessor.pkl", "y_scaler.pkl",
        "category_mapping.json", "city_mapping.json", "district_mapping.json",
    ),
}


def missing_assets(root: Path, mode: str) -> list[str]:
    if mode not in REQUIRED:
        raise ValueError(f"Unknown mode: {mode}")
    return [name for name in REQUIRED[mode] if not (root / name).is_file()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check runtime asset presence (not validity)")
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="working directory from which the scripts will run")
    parser.add_argument("--mode", choices=sorted(REQUIRED), required=True)
    args = parser.parse_args(argv)
    missing = missing_assets(args.root, args.mode)
    if missing:
        print(f"BLOCKED: missing {args.mode} assets in {args.root.resolve()}:")
        for filename in missing:
            print(f"  - {filename}")
        return 1
    print(f"PASS: all {args.mode} asset paths exist in {args.root.resolve()}")
    print("Contents, dependency compatibility, and reported metrics are not verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
