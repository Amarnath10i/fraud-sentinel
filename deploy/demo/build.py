"""Assemble the public-demo build context in build/demo, optionally upload it.

    uv run python deploy/demo/build.py                              # assemble only
    cd build/demo && railway up                                     # deploy on Railway
    uv run --with huggingface_hub python deploy/demo/build.py --upload-hf USER/fraud-sentinel

The context is one Docker image: API + live replay + static frontend, no
PostgreSQL. Building it needs PostgreSQL with a registered production model
(for the registry snapshot), the processed data, the online-state snapshot
(start `sentinel serve` once) and Node.js for the static frontend export.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from sentinel.config import ROOT
from sentinel.serve.export import export_demo

HERE = Path(__file__).resolve().parent
COPY = ["pyproject.toml", "uv.lock", "src", "sql", "configs", "reports"]


def assemble(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in COPY:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, out / name, ignore=ignore)
        else:
            shutil.copy2(src, out / name)
    for name in ("Dockerfile", "README.md"):
        shutil.copy2(HERE / name, out / name)

    export_demo(out / "demo")

    # static frontend, calling the API on its own origin
    env = os.environ | {"NEXT_OUTPUT": "export", "NEXT_PUBLIC_API_URL": ""}
    subprocess.run("npm run build", cwd=ROOT / "frontend", env=env, shell=True, check=True)
    shutil.copytree(ROOT / "frontend" / "out", out / "web")


def upload(out: Path, repo_id: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        folder_path=out,
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy fraud-sentinel",
        delete_patterns=["*"],  # the Space mirrors the build context exactly
    )
    return f"https://huggingface.co/spaces/{repo_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "demo")
    parser.add_argument(
        "--upload-hf", metavar="REPO_ID", help="Hugging Face Space, e.g. amarnath10/fraud-sentinel"
    )
    args = parser.parse_args()
    assemble(args.out)
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    print(f"build context: {args.out} ({size:.0f} MB)")
    if args.upload_hf:
        print(upload(args.out, args.upload_hf))


if __name__ == "__main__":
    main()
