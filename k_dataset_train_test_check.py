"""
MNISQ Datasets: Combined Download, Extract & Size Report
================================================================================

Downloads and extracts all three MNISQ base-QASM archives (MNIST,
Fashion-MNIST, Kuzushiji-MNIST), skipping anything already present, then
counts and reports the discovered train and test set sizes for each
dataset separately -- both totals and per-class breakdowns.

This does NOT run any training or testing -- it only fetches the data and
tells you exactly how many usable QASM-label pairs exist in each split, so
you know what you're working with before running the training scripts.

Install once:
pip install requests tqdm pandas
"""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

# =============================================================================
# SHARED CONFIGURATION
# =============================================================================

FIDELITY = "f90"  # "f80", "f90", or "f95" -- must match what the training scripts expect
N_CLASSES = 10

PROJECT_ROOT = Path.cwd()

OFFICIAL_BASE_URL = (
    "https://qulacs-quantum-datasets.s3.us-west-1.amazonaws.com"
)

# Per-dataset folders, archive names, and the tokens used to identify each
# split's files -- matching each training script's own DATA_ROOT layout
# exactly, so the training scripts find the data without any path changes.
DATASET_CONFIGS = {
    "MNIST": {
        "data_root": PROJECT_ROOT / "mnisq_mnist_data",
        "train_archive": f"base_train_orig_mnist_784_{FIDELITY}.zip",
        "test_archive": f"base_test_mnist_784_{FIDELITY}.zip",
        "train_token": "base_train_orig_mnist_784",
        "test_token": "base_test_mnist_784",
        "dataset_token": "mnist_784",
    },
    "Fashion-MNIST": {
        "data_root": PROJECT_ROOT / "mnisq_fashionmnist_data",
        "train_archive": f"base_train_orig_Fashion-MNIST_{FIDELITY}.zip",
        "test_archive": f"base_test_Fashion-MNIST_{FIDELITY}.zip",
        "train_token": "base_train_orig_fashion-mnist",
        "test_token": "base_test_fashion-mnist",
        "dataset_token": "fashion-mnist",
    },
    "Kuzushiji-MNIST": {
        "data_root": PROJECT_ROOT / "mnisq_kuzushiji_data",
        "train_archive": f"base_train_orig_Kuzushiji-MNIST_{FIDELITY}.zip",
        "test_archive": f"base_test_Kuzushiji-MNIST_{FIDELITY}.zip",
        "train_token": "base_train_orig_kuzushiji-mnist",
        "test_token": "base_test_kuzushiji-mnist",
        "dataset_token": "kuzushiji-mnist",
    },
}

for config in DATASET_CONFIGS.values():
    for subfolder in ["downloads", "extracted"]:
        (config["data_root"] / subfolder).mkdir(parents=True, exist_ok=True)


# =============================================================================
# GENERIC DOWNLOAD / EXTRACT HELPERS (shared by all three datasets)
# =============================================================================

def human_size(number_of_bytes: int | float) -> str:
    value = float(number_of_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def remote_file_size(url: str, timeout: int = 60) -> int | None:
    """Return remote file size when the server provides Content-Length."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        response.raise_for_status()
        size = response.headers.get("Content-Length")
        return int(size) if size is not None else None
    except requests.RequestException:
        return None


def download_file(
    url: str,
    destination: Path,
    chunk_size: int = 1024 * 1024,
    timeout: int = 120,
) -> Path:
    """
    Stream-download a file. Skips the download entirely if `destination`
    already exists and matches the server-reported size. A partial '.part'
    file is used so a failed/interrupted download is never mistaken for a
    complete ZIP archive.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    expected_remote_size = remote_file_size(url)

    if destination.exists():
        local_size = destination.stat().st_size
        if expected_remote_size is None or local_size == expected_remote_size:
            print(f"Already present, skipping download: {destination}")
            return destination
        print(
            f"Existing file size ({human_size(local_size)}) does not match "
            f"the server ({human_size(expected_remote_size)}); re-downloading."
        )
        destination.unlink()

    partial_path = destination.with_suffix(destination.suffix + ".part")
    if partial_path.exists():
        partial_path.unlink()

    print(f"\nDownloading:\n{url}")

    try:
        with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as response:
            response.raise_for_status()

            total_size_header = response.headers.get("Content-Length")
            total_size = (
                int(total_size_header) if total_size_header is not None else expected_remote_size
            )
            if total_size is not None:
                print(f"Expected download size: {human_size(total_size)}")

            with open(partial_path, "wb") as file_handle:
                progress = tqdm(
                    total=total_size, unit="B", unit_scale=True,
                    unit_divisor=1024, desc=destination.name,
                )
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    file_handle.write(chunk)
                    progress.update(len(chunk))
                progress.close()

        if not zipfile.is_zipfile(partial_path):
            raise RuntimeError(f"Downloaded file is not a valid ZIP archive: {partial_path}")

        partial_path.replace(destination)
        print(f"Saved archive: {destination}")
        return destination

    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise


def safe_extract_zip(zip_path: Path, extraction_directory: Path) -> None:
    """Extract ZIP while preventing path-traversal (zip-slip) entries."""
    extraction_directory.mkdir(parents=True, exist_ok=True)
    extraction_root = extraction_directory.resolve()

    with zipfile.ZipFile(zip_path, "r") as archive:
        members = archive.infolist()
        for member in members:
            member_destination = (extraction_directory / member.filename).resolve()
            if os.path.commonpath([str(extraction_root), str(member_destination)]) != str(extraction_root):
                raise RuntimeError(f"Unsafe ZIP path detected: {member.filename}")

        for member in tqdm(members, desc=f"Extracting {zip_path.name}", unit="file"):
            archive.extract(member, extraction_directory)


def archive_marker(extract_root: Path, archive_name: str) -> Path:
    """Marker file dropped after successful extraction, so re-runs are fast no-ops."""
    return extract_root / f".{archive_name}.extracted"


def ensure_archive_downloaded_and_extracted(
    archive_name: str, url: str, download_root: Path, extract_root: Path,
) -> None:
    """Idempotent: downloads only if missing, extracts only if not already extracted."""
    archive_path = download_root / archive_name
    marker_path = archive_marker(extract_root, archive_name)

    if marker_path.exists():
        print(f"Already extracted, skipping: {archive_name}")
        return

    download_file(url, archive_path)

    print(f"\nExtracting {archive_path.name}...")
    safe_extract_zip(archive_path, extract_root)

    marker_path.write_text(f"Extracted from {archive_path}\n", encoding="utf-8")
    print(f"Extraction complete: {archive_path.name}")


def ensure_dataset_downloaded(dataset_name: str, config: dict) -> None:
    print("=" * 78)
    print(f"Checking MNISQ {dataset_name} dataset")
    print(f"Data root: {config['data_root'].resolve()}")
    print("=" * 78)

    download_root = config["data_root"] / "downloads"
    extract_root = config["data_root"] / "extracted"

    train_url = f"{OFFICIAL_BASE_URL}/{config['train_archive']}"
    test_url = f"{OFFICIAL_BASE_URL}/{config['test_archive']}"

    ensure_archive_downloaded_and_extracted(config["train_archive"], train_url, download_root, extract_root)
    ensure_archive_downloaded_and_extracted(config["test_archive"], test_url, download_root, extract_root)

    print(f"{dataset_name} dataset ready. Extracted under: {extract_root.resolve()}")


# =============================================================================
# SIZE COUNTING (train vs test, per dataset)
# =============================================================================

def is_qasm_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() == ".qasm":
        return True
    try:
        beginning = path.read_text(encoding="utf-8", errors="ignore")[:300].lower()
    except OSError:
        return False
    return "openqasm" in beginning and "qreg" in beginning


def numeric_identifier(path: Path) -> str | None:
    matches = re.findall(r"\d+", path.stem)
    return matches[-1] if matches else None


def read_label(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
    matches = re.findall(r"-?\d+", text)
    if not matches:
        return None
    label = int(matches[0])
    return label if label in range(N_CLASSES) else None


def count_split_samples(extract_root: Path, split: str, required_token: str, dataset_token: str) -> pd.DataFrame:
    """
    Finds QASM-label pairs for one split (train or test) and returns a
    DataFrame with sample_id / label for every matched pair, mirroring the
    matching logic used by the training/testing scripts.
    """
    all_files = [p for p in extract_root.rglob("*") if p.is_file()]
    qasm_all = [p for p in all_files if is_qasm_file(p)]

    def matches_split(p: Path) -> bool:
        s = str(p).lower()
        if required_token in s and FIDELITY.lower() in s:
            return True
        return split in s and dataset_token in s and FIDELITY.lower() in s

    qasm_files = [p for p in qasm_all if matches_split(p)]
    if not qasm_files:
        qasm_files = [p for p in qasm_all if split in str(p).lower()]

    label_files = [
        p for p in all_files
        if "label" in str(p).lower()
        and (required_token in str(p).lower() or (split in str(p).lower() and dataset_token in str(p).lower()))
    ]

    label_index: dict[str, Path] = {}
    for p in sorted(label_files):
        ident = numeric_identifier(p)
        if ident is not None:
            label_index.setdefault(ident, p)

    records = []
    for qasm_path in qasm_files:
        ident = numeric_identifier(qasm_path)
        if ident is None or ident not in label_index:
            continue
        label = read_label(label_index[ident])
        if label is None:
            continue
        records.append({"sample_id": ident, "label": label})

    return pd.DataFrame(records)


def report_dataset_sizes(dataset_name: str, config: dict) -> dict:
    extract_root = config["data_root"] / "extracted"

    train_df = count_split_samples(extract_root, "train", config["train_token"], config["dataset_token"])
    test_df = count_split_samples(extract_root, "test", config["test_token"], config["dataset_token"])

    print(f"\n{dataset_name}:")
    print(f"  Train set: {len(train_df):,} samples")
    if not train_df.empty:
        print("    Per-class counts:")
        print(train_df["label"].value_counts().sort_index().to_string().replace("\n", "\n    "))

    print(f"  Test set:  {len(test_df):,} samples")
    if not test_df.empty:
        print("    Per-class counts:")
        print(test_df["label"].value_counts().sort_index().to_string().replace("\n", "\n    "))

    return {
        "dataset": dataset_name,
        "train_total": len(train_df),
        "test_total": len(test_df),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("MNISQ: DOWNLOAD ALL 3 DATASETS + REPORT TRAIN/TEST SIZES")
    print("=" * 78)

    for dataset_name, config in DATASET_CONFIGS.items():
        ensure_dataset_downloaded(dataset_name, config)
        print()

    print("=" * 78)
    print("TRAIN / TEST SIZE REPORT")
    print("=" * 78)

    summary = []
    for dataset_name, config in DATASET_CONFIGS.items():
        summary.append(report_dataset_sizes(dataset_name, config))

    print("\n" + "=" * 78)
    print("SUMMARY (all 3 datasets)")
    print("=" * 78)
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    print(
        f"\nCombined train total: {summary_df['train_total'].sum():,}"
        f" | Combined test total: {summary_df['test_total'].sum():,}"
    )


if __name__ == "__main__":
    main()