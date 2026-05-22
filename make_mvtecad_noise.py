#!/usr/bin/env python3
"""Build a noisy MVTec-AD dataset by randomly injecting anomalous test images into train/good."""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional


class RunLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.handle = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log_path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.close()

    def log(self, message: str):
        print(message)
        if self.handle is not None:
            self.handle.write(message + "\n")
            self.handle.flush()



def copy_or_link(src: Path, dst: Path, symlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if symlink:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)



def replicate_split(src_root: Path, dst_root: Path, split: str):
    for obj_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        src_split = obj_dir / split
        if not src_split.is_dir():
            continue
        dst_split = dst_root / obj_dir.name / split
        shutil.copytree(src_split, dst_split, dirs_exist_ok=True)



def list_files(directory: Path):
    return sorted(path for path in directory.iterdir() if path.is_file())



def collect_anomaly_images(obj_dir: Path):
    samples = []
    test_dir = obj_dir / "test"
    if not test_dir.is_dir():
        return samples
    for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir() and p.name != "good"):
        for image_path in list_files(defect_dir):
            samples.append((defect_dir.name, image_path))
    return samples



def write_manifest(manifest_path: Path, entries: List[str]):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(entries)
    if content:
        content += "\n"
    manifest_path.write_text(content, encoding="utf-8")



def build_dataset(
    source_dir: Path,
    dest_dir: Path,
    noise_ratio: float,
    seed: int,
    symlink: bool,
    manifest_path: Path,
    log_path: Path,
):
    rng = random.Random(seed)
    manifest_entries: List[str] = []
    total_good = 0
    total_injected = 0

    with RunLogger(log_path) as logger:
        log = logger.log
        log(f"[start] source={source_dir}")
        log(f"[start] dest={dest_dir}")
        log(f"[start] noise_ratio={noise_ratio}, seed={seed}, symlink={symlink}")

        for obj_dir in sorted(p for p in source_dir.iterdir() if (p / "train" / "good").is_dir()):
            src_good_dir = obj_dir / "train" / "good"
            dst_good_dir = dest_dir / obj_dir.name / "train" / "good"
            shutil.copytree(src_good_dir, dst_good_dir, dirs_exist_ok=True)

            good_images = list_files(src_good_dir)
            anomaly_images = collect_anomaly_images(obj_dir)
            requested_noise = int(round(len(good_images) * noise_ratio))
            if noise_ratio > 0 and requested_noise == 0 and good_images and anomaly_images:
                requested_noise = 1
            actual_noise = min(requested_noise, len(anomaly_images))

            if requested_noise > len(anomaly_images):
                log(
                    f"[warn] {obj_dir.name}: requested {requested_noise} noisy samples, "
                    f"but only {len(anomaly_images)} anomalous test images are available"
                )

            selected_samples = rng.sample(anomaly_images, actual_noise) if actual_noise else []
            for defect_name, src_image in selected_samples:
                dst_image = dst_good_dir / f"{defect_name}_{src_image.name}"
                copy_or_link(src_image, dst_image, symlink)
                manifest_entries.append(dst_image.relative_to(dest_dir).as_posix())

            final_count = len(good_images) + actual_noise
            total_good += len(good_images)
            total_injected += actual_noise
            log(
                f"[done] {obj_dir.name}: good={len(good_images)}, injected={actual_noise}, "
                f"final_train={final_count}"
            )

        replicate_split(source_dir, dest_dir, "test")
        replicate_split(source_dir, dest_dir, "ground_truth")
        write_manifest(manifest_path, manifest_entries)

        log(f"[manifest] wrote {len(manifest_entries)} entries to {manifest_path}")
        log(f"[summary] original_good={total_good}, injected={total_injected}, final_train={total_good + total_injected}")
        log(f"[ready] Noisy dataset written to {dest_dir}")
        log(f"[ready] Log saved to {log_path}")



def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a noisy MVTec-AD dataset by injecting anomalous samples into train/good."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("../mvtec_anomaly_detection"), help="Pristine MVTec-AD root directory")
    parser.add_argument("--dest-dir", type=Path, default=Path("../noisy_datasets/noisy_mvtec"), help="Output root for the noisy dataset")
    parser.add_argument(
        "--noise-ratio",
        type=float,
        default=0.1,
        help="Injected anomaly count divided by the original train/good count for each class (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for anomaly sampling")
    parser.add_argument("--symlink", action="store_true", help="Use symbolic links instead of copying")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Output path for inject_defects.txt (default: <dest-dir>/inject_defects.txt)",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Output path for run log (default: <dest-dir>/build_noise.log)",
    )
    args = parser.parse_args()

    if args.noise_ratio < 0:
        parser.error("--noise-ratio must be non-negative")

    if args.manifest_path is None:
        args.manifest_path = args.dest_dir / "inject_defects.txt"
    if args.log_path is None:
        args.log_path = args.dest_dir / "build_noise.log"

    return args



def main():
    args = parse_args()
    build_dataset(
        args.source_dir,
        args.dest_dir,
        args.noise_ratio,
        args.seed,
        args.symlink,
        args.manifest_path,
        args.log_path,
    )


if __name__ == "__main__":
    main()
