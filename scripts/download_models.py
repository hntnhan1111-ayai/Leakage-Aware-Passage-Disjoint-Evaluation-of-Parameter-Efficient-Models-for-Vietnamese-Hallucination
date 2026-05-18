import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-modern", action="store_true")
    args = parser.parse_args()
    from src.models.download import download_one
    from src.utils.env import load_hf_token

    token = load_hf_token(required=True)
    names = ["vistral", "phobert", "xlmr"]
    if args.include_modern:
        names += ["qwen3_4b"]
    for name in names:
        download_one(name, token)


if __name__ == "__main__":
    main()
