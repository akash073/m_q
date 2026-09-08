"""
MNISQ Kuzushiji-MNIST: Training Script
================================================================================

What this script does:
1. Checks and downloads the official MNISQ Kuzushiji-MNIST base-QASM archives.
2. Extracts the archives safely.
3. Automatically discovers QASM and label files.
4. Selects a subset from the 10 Kuzushiji-MNIST classes — either a fixed
   per-class count, or (with USE_FULL_DATASET = True) every available
   sample, balanced to the smallest class size.
5. Executes each original 10-qubit MNISQ circuit on a PennyLane simulator.
6. Builds a state-fidelity quantum kernel.
7. Trains and saves a multiclass SVM model bundle.

IMPORTANT — kernel-SVM memory scaling:
    fidelity_kernel_memmap() builds a full N x N matrix, where N is the
    number of training circuits. This is O(N^2) in storage and O(N^2)-O(N^3)
    in fit time. Rather than allocating that matrix in RAM (which fails once
    N is large — e.g. N=60,000 needs ~26.82 GB), the kernel is now computed
    block-by-block and written directly to a disk-backed .npy memmap. Peak
    RAM is bounded by MAX_KERNEL_MEMORY_GB regardless of how many samples
    you use; the full matrix instead needs that much free DISK space, and
    SVM training will be slower than a pure in-RAM kernel because libsvm's
    SMO optimizer accesses the kernel matrix in a near-random pattern,
    which now means disk I/O instead of RAM access.

Install once:
pip install pennylane scikit-learn pandas numpy matplotlib joblib requests tqdm
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import pennylane as qml
import requests
from sklearn.svm import SVC
from tqdm.auto import tqdm


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

SEED = 42
SOURCE_QUBITS = 10
N_QUBITS = 1
N_CLASSES = 10

# Official MNISQ choices:
FIDELITY = "f90"  # "f80", "f90", or "f95"

# --- Dataset size control -----------------------------------------------
# USE_FULL_DATASET = False (default-safe): use a fixed number of samples
#     per class, as before. Good for a quick pipeline check.
# USE_FULL_DATASET = True: use every available sample per class, capped to
#     the smallest class count so classes stay balanced. This is the "full
#     dataset" mode. With the disk-backed kernel below, this no longer
#     needs to be capped for RAM reasons — only disk space matters.
USE_FULL_DATASET = True

# Only used when USE_FULL_DATASET is False.
TRAIN_SAMPLES_PER_CLASS = 50
TEST_SAMPLES_PER_CLASS = 20

# Peak-RAM budget for kernel computation. With the disk-backed memmap
# kernel, this bounds the size of ONE block (block_size rows) held in RAM
# at a time — NOT the size of the full N x N matrix, which now lives on
# disk instead. Raise this only if you know your machine has that much
# free RAM; lower it (or lower block_size) if you're memory-constrained.
MAX_KERNEL_MEMORY_GB = 8.0

# Rows processed per block when streaming the kernel matrix to disk.
# Peak RAM for kernel computation ~= KERNEL_BLOCK_SIZE * N * 8 bytes.
# This is auto-shrunk at runtime if it would exceed MAX_KERNEL_MEMORY_GB
# for your actual N (see check_kernel_memory_budget).
KERNEL_BLOCK_SIZE = 500

# PennyLane simulator:
PENNYLANE_DEVICE = "default.qubit"

# Main local folders:
PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT / "mnisq_kuzushiji_data"
DOWNLOAD_ROOT = PROJECT_ROOT / "mnisq_kuzushiji_data" / "downloads"
EXTRACT_ROOT =PROJECT_ROOT / "mnisq_kuzushiji_data" / "extracted"
OUTPUT_ROOT = PROJECT_ROOT / "mnisq_kuzushiji_results_q1"
STATE_CACHE_ROOT = OUTPUT_ROOT / "state_cache_q1"

MODEL_BUNDLE_PATH = OUTPUT_ROOT / "kuzushiji_quantum_kernel_model_q1.npz"
ARTIFACT_CONFIG_PATH = OUTPUT_ROOT / "kuzushiji_artifact_config_q1.json"
TRAIN_KERNEL_PATH = OUTPUT_ROOT / "kuzushiji_train_kernel_q1.npy"
TRAIN_METADATA_PATH = OUTPUT_ROOT / "kuzushiji_selected_train_samples_q1.csv"
TRAIN_STATES_PATH = OUTPUT_ROOT / "kuzushiji_train_states_q1.npy"

# Official MNISQ data source:
OFFICIAL_BASE_URL = (
    "https://qulacs-quantum-datasets.s3.us-west-1.amazonaws.com"
)

TRAIN_ARCHIVE_NAME = f"base_train_orig_Kuzushiji-MNIST_{FIDELITY}.zip"
TEST_ARCHIVE_NAME = f"base_test_Kuzushiji-MNIST_{FIDELITY}.zip"

TRAIN_URL = f"{OFFICIAL_BASE_URL}/{TRAIN_ARCHIVE_NAME}"
TEST_URL = f"{OFFICIAL_BASE_URL}/{TEST_ARCHIVE_NAME}"

# NOTE: the original script had FashionMNIST class names here
# ("T-shirt/top", "Trouser", ...) left over from a copy-paste of the
# FashionMNIST training script. Kuzushiji-MNIST's 10 classes are actually
# hiragana characters, corrected below (romanized for readability). This
# only affects the human-readable labels saved in the model bundle/metadata
# — it does not change training behavior, since training uses the numeric
# `label` column (0-9) discovered from the dataset's own label files.
CLASS_NAMES = [
    "o",     # 0 - お
    "ki",    # 1 - き
    "su",    # 2 - す
    "tsu",   # 3 - つ
    "na",    # 4 - な
    "ha",    # 5 - は
    "ma",    # 6 - ま
    "ya",    # 7 - や
    "re",    # 8 - れ
    "wo",    # 9 - を
]

for folder in [
    DATA_ROOT,
    DOWNLOAD_ROOT,
    EXTRACT_ROOT,
    OUTPUT_ROOT,
    STATE_CACHE_ROOT,
]:
    folder.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)


# =============================================================================
# 2. DOWNLOAD AND EXTRACTION
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
        response = requests.head(
            url,
            allow_redirects=True,
            timeout=timeout,
        )
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
    Stream-download a file. A partial '.part' file is used so a failed download
    is not mistaken for a complete ZIP archive.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    expected_remote_size = remote_file_size(url)

    if destination.exists():
        local_size = destination.stat().st_size

        if expected_remote_size is None or local_size == expected_remote_size:
            print(f"Archive already exists: {destination}")
            return destination

        print(
            "Existing archive size does not match the server; "
            "downloading it again."
        )
        destination.unlink()

    partial_path = destination.with_suffix(destination.suffix + ".part")

    if partial_path.exists():
        partial_path.unlink()

    print(f"\nDownloading:\n{url}")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=timeout,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()

            total_size_header = response.headers.get("Content-Length")
            total_size = (
                int(total_size_header)
                if total_size_header is not None
                else expected_remote_size
            )

            if total_size is not None:
                print(f"Expected download size: {human_size(total_size)}")

            with open(partial_path, "wb") as file_handle:
                progress = tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=destination.name,
                )

                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue

                    file_handle.write(chunk)
                    progress.update(len(chunk))

                progress.close()

        if not zipfile.is_zipfile(partial_path):
            raise RuntimeError(
                f"Downloaded file is not a valid ZIP archive: {partial_path}"
            )

        partial_path.replace(destination)
        print(f"Saved archive: {destination}")
        return destination

    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise


def safe_extract_zip(zip_path: Path, extraction_directory: Path) -> None:
    """Extract ZIP while preventing path-traversal entries."""
    extraction_directory.mkdir(parents=True, exist_ok=True)
    extraction_root = extraction_directory.resolve()

    with zipfile.ZipFile(zip_path, "r") as archive:
        members = archive.infolist()

        for member in members:
            member_destination = (
                extraction_directory / member.filename
            ).resolve()

            if (
                os.path.commonpath(
                    [str(extraction_root), str(member_destination)]
                )
                != str(extraction_root)
            ):
                raise RuntimeError(
                    f"Unsafe ZIP path detected: {member.filename}"
                )

        for member in tqdm(
            members,
            desc=f"Extracting {zip_path.name}",
            unit="file",
        ):
            archive.extract(member, extraction_directory)


def archive_marker(archive_name: str) -> Path:
    return EXTRACT_ROOT / f".{archive_name}.extracted"


def ensure_archive_downloaded_and_extracted(
    archive_name: str,
    url: str,
) -> None:
    archive_path = DOWNLOAD_ROOT / archive_name
    marker_path = archive_marker(archive_name)

    if marker_path.exists():
        print(f"Already extracted: {archive_name}")
        return

    download_file(url, archive_path)

    print(f"\nExtracting {archive_path.name}...")
    safe_extract_zip(archive_path, EXTRACT_ROOT)

    marker_path.write_text(
        f"Extracted from {archive_path}\n",
        encoding="utf-8",
    )

    print(f"Extraction complete: {archive_path.name}")


def ensure_dataset_available() -> None:
    print("=" * 78)
    print("Checking MNISQ Kuzushiji-MNIST dataset")
    print("=" * 78)

    ensure_archive_downloaded_and_extracted(
        TRAIN_ARCHIVE_NAME,
        TRAIN_URL,
    )
    ensure_archive_downloaded_and_extracted(
        TEST_ARCHIVE_NAME,
        TEST_URL,
    )


# =============================================================================
# 3. DATASET DISCOVERY
# =============================================================================

def is_qasm_file(path: Path) -> bool:
    if not path.is_file():
        return False

    if path.suffix.lower() == ".qasm":
        return True

    if "qasm" not in {part.lower() for part in path.parts}:
        return False

    try:
        beginning = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )[:200].lower()
    except OSError:
        return False

    return "openqasm" in beginning or "qreg" in beginning


def find_qasm_files(split: str) -> list[Path]:
    split = split.lower()

    if split == "train":
        required_archive_text = "base_train_orig_kuzushiji-mnist"
    elif split == "test":
        required_archive_text = "base_test_kuzushiji-mnist"
    else:
        raise ValueError("split must be 'train' or 'test'")

    all_candidates = [
        path for path in EXTRACT_ROOT.rglob("*")
        if is_qasm_file(path)
    ]

    exact_candidates = [
        path
        for path in all_candidates
        if required_archive_text in str(path).lower()
        and FIDELITY.lower() in str(path).lower()
    ]

    if exact_candidates:
        return sorted(exact_candidates)

    split_candidates = [
        path
        for path in all_candidates
        if split in str(path).lower()
        and "kuzushiji" in str(path).lower()
        and FIDELITY.lower() in str(path).lower()
    ]

    if split_candidates:
        return sorted(split_candidates)

    raise FileNotFoundError(
        f"No {split} QASM files were found under:\n{EXTRACT_ROOT}"
    )


def numeric_identifier(path: Path) -> str | None:
    """Extract the final integer from a filename."""
    matches = re.findall(r"\d+", path.stem)
    return matches[-1] if matches else None


def read_label(path: Path) -> int:
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).strip()

    matches = re.findall(r"-?\d+", text)

    if not matches:
        raise ValueError(f"No integer label found in {path}")

    label = int(matches[0])

    if label not in range(N_CLASSES):
        raise ValueError(
            f"Invalid label {label} in {path}; expected 0 through 9."
        )

    return label


def candidate_label_files(split: str) -> list[Path]:
    split = split.lower()

    if split == "train":
        required_archive_text = "base_train_orig_kuzushiji-mnist"
    else:
        required_archive_text = "base_test_kuzushiji-mnist"

    candidates: list[Path] = []

    for path in EXTRACT_ROOT.rglob("*"):
        if not path.is_file():
            continue

        lower_path = str(path).lower()

        if "label" not in lower_path:
            continue

        if (
            required_archive_text not in lower_path
            and not (
                split in lower_path
                and "kuzushiji" in lower_path
                and FIDELITY.lower() in lower_path
            )
        ):
            continue

        candidates.append(path)

    return sorted(candidates)


def build_label_index(split: str) -> dict[str, Path]:
    labels = candidate_label_files(split)

    if not labels:
        raise FileNotFoundError(
            f"No {split} label files were found under {EXTRACT_ROOT}."
        )

    index: dict[str, Path] = {}

    for path in labels:
        identifier = numeric_identifier(path)

        if identifier is not None:
            index.setdefault(identifier, path)

    return index


def discover_samples(split: str) -> pd.DataFrame:
    qasm_files = find_qasm_files(split)
    label_index = build_label_index(split)

    records: list[dict] = []
    unmatched: list[str] = []

    for qasm_path in tqdm(
        qasm_files,
        desc=f"Matching {split} QASM files to labels",
    ):
        identifier = numeric_identifier(qasm_path)

        if identifier is None or identifier not in label_index:
            unmatched.append(str(qasm_path))
            continue

        label_path = label_index[identifier]

        try:
            label = read_label(label_path)
        except ValueError:
            unmatched.append(str(qasm_path))
            continue

        records.append(
            {
                "sample_id": identifier,
                "qasm_path": str(qasm_path),
                "label_path": str(label_path),
                "label": label,
            }
        )

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise RuntimeError(
            "QASM files were discovered, but no QASM-label pairs could be created."
        )

    if unmatched:
        pd.DataFrame({"qasm_path": unmatched}).to_csv(
            OUTPUT_ROOT / f"unmatched_{split}_qasm.csv",
            index=False,
        )
        print(
            f"Warning: {len(unmatched):,} {split} QASM files were unmatched."
        )

    print(f"\n{split.title()} samples discovered: {len(dataframe):,}")
    print(dataframe["label"].value_counts().sort_index())

    return dataframe


def balanced_sample(
    dataframe: pd.DataFrame,
    samples_per_class: int,
    seed: int,
) -> pd.DataFrame:
    selected_groups: list[pd.DataFrame] = []

    for class_id in range(N_CLASSES):
        class_rows = dataframe[dataframe["label"] == class_id]

        if len(class_rows) < samples_per_class:
            raise ValueError(
                f"Class {class_id} contains only {len(class_rows)} samples, "
                f"but {samples_per_class} were requested."
            )

        selected_groups.append(
            class_rows.sample(
                n=samples_per_class,
                replace=False,
                random_state=seed + class_id,
            )
        )

    return (
        pd.concat(selected_groups, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def full_balanced_sample(
    dataframe: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """
    'Full dataset' mode: use every available sample per class, capped to
    the smallest class count so classes stay balanced (an SVM trained on
    a badly imbalanced kernel tends to be biased toward majority classes).

    Prints the true per-class counts so you can see exactly how much data
    is available before committing to a run.
    """
    class_counts = dataframe["label"].value_counts().sort_index()
    min_count = int(class_counts.min())

    print(f"\nPer-class sample counts ({len(dataframe):,} total discovered):")
    print(class_counts.to_string())
    print(
        f"\nUsing {min_count:,} samples per class (smallest class size) "
        f"= {min_count * N_CLASSES:,} total (balanced full-dataset mode)."
    )

    return balanced_sample(
        dataframe,
        samples_per_class=min_count,
        seed=seed,
    )


def select_training_samples(
    dataframe: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Dispatch to full-dataset or fixed-count sampling based on config."""
    if USE_FULL_DATASET:
        return full_balanced_sample(dataframe, seed=seed)
    return balanced_sample(
        dataframe,
        samples_per_class=TRAIN_SAMPLES_PER_CLASS,
        seed=seed,
    )


def select_test_samples(
    dataframe: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Dispatch to full-dataset or fixed-count sampling based on config."""
    if USE_FULL_DATASET:
        return full_balanced_sample(dataframe, seed=seed)
    return balanced_sample(
        dataframe,
        samples_per_class=TEST_SAMPLES_PER_CLASS,
        seed=seed,
    )


# =============================================================================
# 4. QASM PARSING AND PENNYLANE EXECUTION
# =============================================================================

def state_cache_path(qasm_path: Path) -> Path:
    stat = qasm_path.stat()
    fingerprint = (
        f"{qasm_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|"
        f"{PENNYLANE_DEVICE}"
    )

    digest = hashlib.sha256(
        fingerprint.encode("utf-8")
    ).hexdigest()

    return STATE_CACHE_ROOT / f"{digest}.npy"


def remove_qasm_comments(qasm_text: str) -> str:
    """Remove // and /* ... */ comments from OpenQASM text."""
    without_block_comments = re.sub(
        r"/\*.*?\*/",
        "",
        qasm_text,
        flags=re.DOTALL,
    )

    return re.sub(
        r"//.*?$",
        "",
        without_block_comments,
        flags=re.MULTILINE,
    )


def split_qasm_statements(qasm_text: str) -> list[str]:
    """Split an OpenQASM program into semicolon-terminated statements."""
    text_without_comments = remove_qasm_comments(qasm_text)
    return [
        statement.strip()
        for statement in text_without_comments.split(";")
        if statement.strip()
    ]


def safe_qasm_angle(expression: str) -> float:
    """Evaluate an OpenQASM numeric angle safely."""
    expression = expression.strip()
    expression = expression.replace("^", "**")

    if not re.fullmatch(
        r"[0-9eEpiPI+\-*/().\s*]+",
        expression,
    ):
        raise ValueError(
            f"Unsafe or unsupported QASM parameter expression: {expression}"
        )

    try:
        value = eval(
            expression,
            {"__builtins__": {}},
            {"pi": np.pi, "PI": np.pi},
        )
    except Exception as error:
        raise ValueError(
            f"Could not evaluate QASM parameter: {expression}"
        ) from error

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(
            f"Non-finite QASM parameter: {expression}"
        )

    return value


def parse_parameter_list(parameter_text: str | None) -> list[float]:
    if parameter_text is None or not parameter_text.strip():
        return []

    return [
        safe_qasm_angle(part)
        for part in parameter_text.split(",")
    ]


def parse_wire_list(
    operand_text: str,
    register_sizes: dict[str, int],
) -> list[int]:
    """Convert operands such as q[2],q[7] to integer PennyLane wires."""
    offsets: dict[str, int] = {}
    running_offset = 0

    for register_name, register_size in register_sizes.items():
        offsets[register_name] = running_offset
        running_offset += register_size

    operands = [
        operand.strip()
        for operand in operand_text.split(",")
        if operand.strip()
    ]

    wires: list[int] = []

    for operand in operands:
        match = re.fullmatch(
            r"([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]",
            operand,
        )

        if match is None:
            raise ValueError(
                f"Unsupported QASM qubit operand: {operand}"
            )

        register_name = match.group(1)
        register_index = int(match.group(2))

        if register_name not in register_sizes:
            raise ValueError(
                f"Unknown quantum register: {register_name}"
            )

        if not 0 <= register_index < register_sizes[register_name]:
            raise ValueError(
                f"Qubit index out of range: {operand}"
            )

        wires.append(
            offsets[register_name] + register_index
        )

    return wires


def parse_mnisq_qasm(
    qasm_text: str,
) -> tuple[dict[str, int], list[tuple[str, list[float], list[int]]]]:
    """Parse a flat MNISQ OpenQASM-2 state-preparation circuit."""
    statements = split_qasm_statements(qasm_text)

    register_sizes: dict[str, int] = {}
    raw_gate_statements: list[str] = []

    ignored_prefixes = (
        "openqasm",
        "include",
        "creg",
        "measure",
        "barrier",
        "reset",
    )

    for statement in statements:
        lowered = statement.lower().strip()

        qreg_match = re.fullmatch(
            r"qreg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]",
            statement,
            flags=re.IGNORECASE,
        )

        if qreg_match is not None:
            register_name = qreg_match.group(1)
            register_size = int(qreg_match.group(2))
            register_sizes[register_name] = register_size
            continue

        if lowered.startswith(ignored_prefixes):
            continue

        if lowered.startswith(("gate ", "opaque ")):
            raise ValueError(
                "Custom gate declarations were found."
            )

        raw_gate_statements.append(statement)

    if not register_sizes:
        raise ValueError("No qreg declaration was found.")

    total_qubits = sum(register_sizes.values())

    if total_qubits != SOURCE_QUBITS:
        raise ValueError(
            f"The source circuit declares {total_qubits} total qubits; "
            f"expected {SOURCE_QUBITS} for the MNISQ dataset."
        )

    operations: list[
        tuple[str, list[float], list[int]]
    ] = []

    gate_pattern = re.compile(
        r"^([A-Za-z_]\w*)"
        r"(?:\s*\((.*?)\))?"
        r"\s+(.+)$",
        flags=re.DOTALL,
    )

    for statement in raw_gate_statements:
        match = gate_pattern.fullmatch(statement.strip())

        if match is None:
            raise ValueError(
                f"Could not parse QASM statement: {statement}"
            )

        gate_name = match.group(1).lower()
        parameters = parse_parameter_list(match.group(2))
        wires = parse_wire_list(
            match.group(3),
            register_sizes,
        )

        operations.append(
            (gate_name, parameters, wires)
        )

    return register_sizes, operations


def require_gate_shape(
    gate_name: str,
    parameters: list[float],
    wires: list[int],
    expected_parameters: int,
    expected_wires: int,
) -> None:
    if len(parameters) != expected_parameters:
        raise ValueError(
            f"Gate '{gate_name}' expects {expected_parameters} "
            f"parameters but received {len(parameters)}."
        )

    if len(wires) != expected_wires:
        raise ValueError(
            f"Gate '{gate_name}' expects {expected_wires} "
            f"wires but received {len(wires)}."
        )


def apply_qasm_gate(
    gate_name: str,
    parameters: list[float],
    wires: list[int],
) -> None:
    """Map a parsed OpenQASM gate directly to a PennyLane operation."""

    if gate_name in {"u", "u3"}:
        require_gate_shape(gate_name, parameters, wires, 3, 1)
        qml.U3(
            parameters[0],
            parameters[1],
            parameters[2],
            wires=wires[0],
        )

    elif gate_name == "u2":
        require_gate_shape(gate_name, parameters, wires, 2, 1)
        qml.U3(
            np.pi / 2,
            parameters[0],
            parameters[1],
            wires=wires[0],
        )

    elif gate_name in {"u1", "p", "phase"}:
        require_gate_shape(gate_name, parameters, wires, 1, 1)
        qml.PhaseShift(parameters[0], wires=wires[0])

    elif gate_name == "rx":
        require_gate_shape(gate_name, parameters, wires, 1, 1)
        qml.RX(parameters[0], wires=wires[0])

    elif gate_name == "ry":
        require_gate_shape(gate_name, parameters, wires, 1, 1)
        qml.RY(parameters[0], wires=wires[0])

    elif gate_name == "rz":
        require_gate_shape(gate_name, parameters, wires, 1, 1)
        qml.RZ(parameters[0], wires=wires[0])

    elif gate_name == "x":
        require_gate_shape(gate_name, parameters, wires, 0, 1)
        qml.PauliX(wires=wires[0])

    elif gate_name == "y":
        require_gate_shape(gate_name, parameters, wires, 0, 1)
        qml.PauliY(wires=wires[0])

    elif gate_name == "z":
        require_gate_shape(gate_name, parameters, wires, 0, 1)
        qml.PauliZ(wires=wires[0])

    elif gate_name == "h":
        require_gate_shape(gate_name, parameters, wires, 0, 1)
        qml.Hadamard(wires=wires[0])

    elif gate_name == "s":
        require_gate_shape(gate_name, parameters, wires, 0, 1)
        qml.S(wires=wires[0])

    elif gate_name == "sdg":
        require_gate_shape(gate_name, parameters, wires, 0, 1)
        qml.adjoint(qml.S)(wires=wires[0])

    elif gate_name == "t":
        require_gate_shape(gate_name, parameters, wires, 0, 1)
        qml.T(wires=wires[0])

    elif gate_name == "tdg":
        require_gate_shape(gate_name, parameters, wires, 0, 1)
        qml.adjoint(qml.T)(wires=wires[0])

    elif gate_name == "sx":
        require_gate_shape(gate_name, parameters, wires, 0, 1)
        qml.SX(wires=wires[0])

    elif gate_name == "sxdg":
        require_gate_shape(gate_name, parameters, wires, 0, 1)
        qml.adjoint(qml.SX)(wires=wires[0])

    elif gate_name in {"id", "i"}:
        require_gate_shape(gate_name, parameters, wires, 0, 1)
        qml.Identity(wires=wires[0])

    elif gate_name in {"cx", "cnot"}:
        require_gate_shape(gate_name, parameters, wires, 0, 2)
        qml.CNOT(wires=wires)

    elif gate_name == "cy":
        require_gate_shape(gate_name, parameters, wires, 0, 2)
        qml.CY(wires=wires)

    elif gate_name == "cz":
        require_gate_shape(gate_name, parameters, wires, 0, 2)
        qml.CZ(wires=wires)

    elif gate_name == "swap":
        require_gate_shape(gate_name, parameters, wires, 0, 2)
        qml.SWAP(wires=wires)

    elif gate_name == "crx":
        require_gate_shape(gate_name, parameters, wires, 1, 2)
        qml.CRX(parameters[0], wires=wires)

    elif gate_name == "cry":
        require_gate_shape(gate_name, parameters, wires, 1, 2)
        qml.CRY(parameters[0], wires=wires)

    elif gate_name == "crz":
        require_gate_shape(gate_name, parameters, wires, 1, 2)
        qml.CRZ(parameters[0], wires=wires)

    elif gate_name in {"cp", "cu1"}:
        require_gate_shape(gate_name, parameters, wires, 1, 2)
        qml.ControlledPhaseShift(
            parameters[0],
            wires=wires,
        )

    elif gate_name in {"ccx", "toffoli"}:
        require_gate_shape(gate_name, parameters, wires, 0, 3)
        qml.Toffoli(wires=wires)

    else:
        raise ValueError(
            f"Unsupported QASM gate '{gate_name}'."
        )


def _one_qubit_pure_state_from_source_state(
    source_state: np.ndarray,
    target_wire: int = 0,
) -> np.ndarray:
    source_state = np.asarray(
        source_state,
        dtype=np.complex128,
    ).reshape([2] * SOURCE_QUBITS)

    psi = np.moveaxis(
        source_state,
        target_wire,
        0,
    ).reshape(2, -1)

    rho = psi @ psi.conj().T
    rho = (rho + rho.conj().T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    q1_state = eigenvectors[:, np.argmax(eigenvalues)]

    nonzero = np.flatnonzero(np.abs(q1_state) > 1e-12)
    if len(nonzero):
        phase = np.angle(q1_state[nonzero[0]])
        q1_state = q1_state * np.exp(-1j * phase)

    norm = np.linalg.norm(q1_state)
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Could not construct a valid q=1 state.")

    q1_state = q1_state / norm

    if q1_state.size != 2 ** N_QUBITS:
        raise RuntimeError(
            f"Unexpected q=1 state length {q1_state.size}; "
            f"expected {2 ** N_QUBITS}."
        )

    return q1_state


def execute_qasm_with_pennylane(qasm_path: Path) -> np.ndarray:
    cache_path = state_cache_path(qasm_path)

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if cached.shape == (2,):
            return cached

    qasm_text = qasm_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if "densematrix" in qasm_text.lower():
        raise ValueError(
            "DenseMatrix was found. Download the MNISQ base archives."
        )

    try:
        _, operations = parse_mnisq_qasm(qasm_text)
    except Exception as parse_error:
        diagnostic_path = OUTPUT_ROOT / "failed_original_circuit.qasm"
        diagnostic_path.write_text(qasm_text, encoding="utf-8")
        raise RuntimeError(
            f"Manual QASM parsing failed for {qasm_path.name}: "
            f"{parse_error}."
        ) from parse_error

    device = qml.device(
        PENNYLANE_DEVICE,
        wires=SOURCE_QUBITS,
        shots=None,
    )

    @qml.qnode(
        device,
        interface=None,
        diff_method=None,
    )
    def circuit_state():
        for gate_name, parameters, wires in operations:
            apply_qasm_gate(gate_name, parameters, wires)
        return qml.state()

    source_state = np.asarray(
        circuit_state(),
        dtype=np.complex128,
    ).reshape(-1)

    expected_source_dimension = 2 ** SOURCE_QUBITS
    if source_state.size != expected_source_dimension:
        raise RuntimeError(
            f"Unexpected source state length {source_state.size}; "
            f"expected {expected_source_dimension}."
        )

    source_state = source_state / np.linalg.norm(source_state)

    q1_state = _one_qubit_pure_state_from_source_state(
        source_state,
        target_wire=0,
    )

    np.save(cache_path, q1_state, allow_pickle=False)

    return q1_state


def validate_first_circuit(metadata: pd.DataFrame, split_name: str) -> None:
    """Execute one circuit before processing the complete split."""
    if metadata.empty:
        raise RuntimeError(f"No {split_name} samples are available.")

    first_path = Path(metadata.iloc[0]["qasm_path"])

    print(f"\nValidating one {split_name} circuit:")
    print(first_path)

    try:
        state = execute_qasm_with_pennylane(first_path)
    except Exception as error:
        print("\nFirst-circuit validation failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error message: {error}")

        print("\nInstalled versions:")
        print(f"  PennyLane: {qml.__version__}")

        raise RuntimeError(
            "The first MNISQ circuit could not be imported."
        ) from error

    print(
        "First circuit executed successfully. "
        f"State shape: {state.shape}, norm: {np.linalg.norm(state):.8f}"
    )


def create_state_matrix(
    metadata: pd.DataFrame,
    split_name: str,
) -> np.ndarray:
    states: list[np.ndarray] = []
    failures: list[dict] = []

    for row in tqdm(
        metadata.itertuples(index=False),
        total=len(metadata),
        desc=f"Simulating {split_name} circuits",
    ):
        qasm_path = Path(row.qasm_path)

        try:
            states.append(execute_qasm_with_pennylane(qasm_path))
        except Exception as error:
            failures.append(
                {
                    "sample_id": row.sample_id,
                    "qasm_path": str(qasm_path),
                    "error": repr(error),
                }
            )

    if failures:
        failure_path = OUTPUT_ROOT / f"{split_name}_simulation_failures.csv"
        pd.DataFrame(failures).to_csv(failure_path, index=False)

        raise RuntimeError(
            f"{len(failures)} {split_name} circuits failed."
        )

    return np.stack(states, axis=0)


# =============================================================================
# 5. QUANTUM KERNEL AND CLASSIFIER (disk-backed, RAM-bounded)
# =============================================================================

def estimate_kernel_memory_gb(n: int) -> float:
    """Estimated storage (GB) for an n x n float64 kernel matrix."""
    return (n * n * 8) / (1024 ** 3)


def safe_block_size(n: int, max_gb: float, requested_block_size: int) -> int:
    """
    Largest block_size (rows processed per chunk) such that one block
    (block_size x n float64 values) stays within max_gb of RAM.
    Never returns less than 1, never more than requested_block_size.
    """
    max_block_bytes = max_gb * (1024 ** 3)
    bytes_per_row = max(n, 1) * 8
    max_rows_by_ram = max(1, int(max_block_bytes // bytes_per_row))
    return max(1, min(requested_block_size, max_rows_by_ram))


def check_kernel_memory_budget(
    n: int,
    requested_block_size: int = KERNEL_BLOCK_SIZE,
) -> int:
    """
    With the disk-backed kernel, the full N x N matrix lives on disk, not
    in RAM — so this checks two different things instead of one:

      1) Peak RAM: bounded by block_size (rows held in memory at once),
         which is auto-shrunk here if the requested block_size would
         exceed MAX_KERNEL_MEMORY_GB for this N.
      2) Free disk space: the full N x N float64 matrix must fit on disk
         at OUTPUT_ROOT.

    Returns the (possibly shrunk) block_size to actually use.
    """
    full_matrix_gb = estimate_kernel_memory_gb(n)
    effective_block_size = safe_block_size(
        n, MAX_KERNEL_MEMORY_GB, requested_block_size
    )
    working_set_gb = (effective_block_size * n * 8) / (1024 ** 3)

    print(
        f"\nFull kernel matrix (disk-backed): {n:,} x {n:,} "
        f"-> approx. {full_matrix_gb:.2f} GB on disk"
    )
    print(
        f"Kernel block size: requested={requested_block_size}, "
        f"using={effective_block_size} "
        f"-> peak RAM approx. {working_set_gb:.3f} GB "
        f"(budget: {MAX_KERNEL_MEMORY_GB} GB)"
    )

    if effective_block_size < 1:
        raise MemoryError(
            f"Even a single kernel row ({n:,} floats = "
            f"{(n * 8) / (1024 ** 3):.3f} GB) exceeds "
            f"MAX_KERNEL_MEMORY_GB ({MAX_KERNEL_MEMORY_GB} GB). "
            f"You would need to raise MAX_KERNEL_MEMORY_GB, or reduce N."
        )

    free_bytes = shutil.disk_usage(OUTPUT_ROOT).free
    free_gb = free_bytes / (1024 ** 3)

    # Leave a small safety margin (1 GB or 5%, whichever is larger) so the
    # disk doesn't fill to zero.
    required_with_margin_gb = full_matrix_gb + max(1.0, full_matrix_gb * 0.05)

    if required_with_margin_gb > free_gb:
        raise OSError(
            f"Full kernel matrix needs ~{full_matrix_gb:.2f} GB of free disk "
            f"space (plus safety margin -> {required_with_margin_gb:.2f} GB) "
            f"at {OUTPUT_ROOT.resolve()}, but only {free_gb:.2f} GB is "
            f"available.\n"
            f"Options:\n"
            f"  1) Free up disk space on that drive (or point OUTPUT_ROOT at "
            f"a drive with more free space).\n"
            f"  2) Lower the sample count (set USE_FULL_DATASET = False and "
            f"pick a smaller TRAIN_SAMPLES_PER_CLASS) to shrink N.\n"
            f"  3) Use an approximate kernel method (e.g. Nystrom "
            f"approximation or random features) instead of an exact "
            f"precomputed-kernel SVM for very large N."
        )

    return effective_block_size


def fidelity_kernel_memmap(
    states_a: np.ndarray,
    states_b: np.ndarray,
    memmap_path: Path,
    block_size: int,
) -> np.memmap:
    """
    K(i,j) = |<psi_i|psi_j>|^2

    Computes the fidelity kernel block-by-block and writes each block
    straight to a disk-backed .npy memmap, instead of allocating the full
    N x N matrix in RAM. Peak RAM usage is bounded by
    block_size * len(states_b) * 8 bytes, NOT by the size of the full
    kernel matrix — so this scales to any N as long as there is enough
    free disk space.
    """
    n_a = len(states_a)
    n_b = len(states_b)

    if memmap_path.exists():
        memmap_path.unlink()

    kernel = np.lib.format.open_memmap(
        memmap_path,
        mode="w+",
        dtype=np.float64,
        shape=(n_a, n_b),
    )

    for start in tqdm(
        range(0, n_a, block_size),
        desc=f"Computing fidelity kernel (disk-backed, block={block_size})",
    ):
        end = min(start + block_size, n_a)
        overlaps = states_a[start:end].conj() @ states_b.T
        block = np.abs(overlaps) ** 2
        np.clip(block, 0.0, 1.0, out=block)
        kernel[start:end] = block

    kernel.flush()
    return kernel


def current_artifact_config() -> dict:
    """Configuration that determines whether saved artifacts are reusable."""
    return {
        "dataset": "MNISQ Kuzushiji-MNIST",
        "fidelity": FIDELITY,
        "number_of_qubits": N_QUBITS,
        "source_number_of_qubits": SOURCE_QUBITS,
        "qubit_reduction": "wire0_reduced_density_dominant_eigenvector",
        "number_of_classes": N_CLASSES,
        "pennylane_device": PENNYLANE_DEVICE,
        "use_full_dataset": USE_FULL_DATASET,
        "train_samples_per_class": TRAIN_SAMPLES_PER_CLASS,
        "test_samples_per_class": TEST_SAMPLES_PER_CLASS,
        "seed": SEED,
        "kernel": "state_fidelity_squared",
        "kernel_storage": "disk_backed_memmap",
        "max_kernel_memory_gb": MAX_KERNEL_MEMORY_GB,
        "svm_C": 10.0,
        "svm_class_weight": "balanced",
        "svm_decision_function_shape": "ovr",
    }


def save_artifact_config() -> None:
    with open(
        ARTIFACT_CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            current_artifact_config(),
            file_handle,
            indent=4,
        )


def save_model_bundle(
    classifier: SVC,
    train_metadata: pd.DataFrame,
    train_labels: np.ndarray,
) -> None:
    """Save the fitted SVM together with the exact training-circuit order in NPZ format."""
    model_npz_path = MODEL_BUNDLE_PATH

    # Save SVM model separately using joblib (SVM objects need joblib)
    svm_model_path = OUTPUT_ROOT / "kuzushiji_svm_model_q1.joblib"
    joblib.dump(classifier, svm_model_path, compress=3)

    # Prepare metadata
    train_sample_ids = train_metadata["sample_id"].astype(str).tolist()
    train_qasm_paths = train_metadata["qasm_path"].astype(str).tolist()
    class_names_str = "|".join(CLASS_NAMES)
    config_str = json.dumps(current_artifact_config())

    # Save everything to NPZ
    np.savez_compressed(
        model_npz_path,
        train_labels=np.asarray(train_labels, dtype=np.int64),
        train_sample_ids=np.array(train_sample_ids, dtype=object),
        train_qasm_paths=np.array(train_qasm_paths, dtype=object),
        class_names=np.array(CLASS_NAMES, dtype=object),
        configuration=config_str,
        svm_model_path=str(svm_model_path),
    )

    print(f"Saved model metadata: {model_npz_path.resolve()}")
    print(f"Saved SVM model: {svm_model_path.resolve()}")


# =============================================================================
# 6. MAIN TRAINING FUNCTION
# =============================================================================

def main() -> None:
    start_time = time.perf_counter()

    ensure_dataset_available()

    print("\nDiscovering extracted files...")
    train_all = discover_samples("train")
    test_all = discover_samples("test")

    print(
        f"\nSample selection mode: "
        f"{'FULL DATASET (balanced to smallest class)' if USE_FULL_DATASET else 'FIXED PER-CLASS COUNT'}"
    )

    train_metadata = select_training_samples(train_all, seed=SEED)
    test_metadata = select_test_samples(test_all, seed=SEED + 1000)

    train_labels = train_metadata["label"].to_numpy(
        dtype=np.int64
    )
    test_labels = test_metadata["label"].to_numpy(
        dtype=np.int64
    )

    print(
        f"\nSelected {len(train_metadata):,} training circuits and "
        f"{len(test_metadata):,} test circuits."
    )

    # Check RAM (per-block) and disk (full matrix) budgets before spending
    # any time simulating circuits. This no longer rejects large N outright
    # — it shrinks the block size to respect MAX_KERNEL_MEMORY_GB and only
    # fails if there isn't enough free disk space for the full matrix.
    effective_block_size = check_kernel_memory_budget(
        len(train_metadata),
        requested_block_size=KERNEL_BLOCK_SIZE,
    )

    train_metadata.to_csv(
        TRAIN_METADATA_PATH,
        index=False,
    )

    print(
        "\nTesting the import pipeline before the full simulation..."
    )
    validate_first_circuit(
        train_metadata,
        "train",
    )

    print(
        f"\nExecuting original {SOURCE_QUBITS}-qubit Kuzushiji-MNIST circuits and reducing to q={N_QUBITS}..."
    )
    train_states = create_state_matrix(
        train_metadata,
        "train",
    )

    print(
        f"Train state matrix: {train_states.shape}"
    )

    np.save(
        TRAIN_STATES_PATH,
        train_states,
        allow_pickle=False,
    )

    train_kernel = fidelity_kernel_memmap(
        train_states,
        train_states,
        memmap_path=TRAIN_KERNEL_PATH,
        block_size=effective_block_size,
    )

    np.fill_diagonal(
        train_kernel,
        1.0,
    )
    train_kernel.flush()

    print("\nTraining multiclass SVM...")
    print(
        "Note: the kernel matrix is disk-backed, so libsvm's kernel "
        "lookups will hit disk during fitting. This is expected to be "
        "slower than an in-RAM kernel of the same size."
    )
    classifier = SVC(
        kernel="precomputed",
        C=10.0,
        class_weight="balanced",
        decision_function_shape="ovr",
        probability=True,
        random_state=SEED,
    )

    classifier.fit(
        train_kernel,
        train_labels,
    )

    save_model_bundle(
        classifier,
        train_metadata,
        train_labels,
    )
    save_artifact_config()

    total_seconds = (
        time.perf_counter() - start_time
    )

    print("\n" + "=" * 78)
    print("Q=1 KUZUSHIJI-MNIST TRAINING COMPLETE")
    print("=" * 78)
    print(
        f"Saved trained model bundle: "
        f"{MODEL_BUNDLE_PATH.resolve()}"
    )
    print(
        f"Total runtime: {total_seconds:.2f} seconds"
    )
    print(
        f"Outputs saved in: {OUTPUT_ROOT.resolve()}"
    )


if __name__ == "__main__":
    main()