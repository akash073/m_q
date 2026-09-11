from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pennylane as qml
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.auto import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
SOURCE_QUBITS = 10
N_QUBITS = 1
N_CLASSES = 10
FIDELITY = "f90"

# --- Test-set size control -----------------------------------------------
# USE_ALL_TEST_SAMPLES = True:  use every discovered test sample per dataset
#     (dataset test counts are allowed to differ from each other).
# USE_ALL_TEST_SAMPLES = False: use a balanced "10 set" instead — exactly
#     TEST_SAMPLES_PER_CLASS samples from EACH of the N_CLASSES classes,
#     for TEST_SAMPLES_PER_CLASS * N_CLASSES total samples per dataset
#     (10 per class x 10 classes = 100 total, by default).
USE_ALL_TEST_SAMPLES = True
TEST_SAMPLES_PER_CLASS = 200#10

PENNYLANE_DEVICE = "default.qubit"

PROJECT_ROOT = Path.cwd()
OUTPUT_ROOT = PROJECT_ROOT / "quantum_test_results"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Folder structure exactly matching the user's project screenshot.
# Fashion fallbacks are included because some earlier scripts used
# mnisq_fashion_data / mnisq_fashion_results_q1.
DATASET_CONFIGS = {
    "MNIST": {
        "data_roots": [
            PROJECT_ROOT / "mnisq_mnist_data",
        ],
        "result_roots": [
            PROJECT_ROOT / "mnisq_mnist_results_q1",
        ],
        "archive_tokens": ["base_test_mnist_784", "mnist_784"],
        "dataset_tokens": ["mnist_784"],
        "class_names": [str(i) for i in range(10)],
    },
    "FashionMNIST": {
        "data_roots": [
            PROJECT_ROOT / "mnisq_fashionmnist_data",
            PROJECT_ROOT / "mnisq_fashion_data",
        ],
        "result_roots": [
            PROJECT_ROOT / "mnisq_fashionmnist_results_q1",
            PROJECT_ROOT / "mnisq_fashion_results_q1",
        ],
        "archive_tokens": ["base_test_fashion-mnist", "fashion-mnist"],
        "dataset_tokens": ["fashion-mnist", "fashionmnist"],
        "class_names": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
    },
    "Kuzushiji-MNIST": {
        "data_roots": [
            PROJECT_ROOT / "mnisq_kuzushiji_data",
        ],
        "result_roots": [
            PROJECT_ROOT / "mnisq_kuzushiji_results_q1",
        ],
        "archive_tokens": ["base_test_kuzushiji-mnist", "kuzushiji-mnist"],
        "dataset_tokens": ["kuzushiji-mnist", "kuzushiji"],
        "class_names": [f"Class {i}" for i in range(10)],
    },
}

np.random.seed(SEED)

# ============================================================
# OPTIONAL ENERGY TRACKING
# ============================================================

try:
    from codecarbon import EmissionsTracker
    import codecarbon
    CODECARBON_AVAILABLE = True
    CODECARBON_VERSION = codecarbon.__version__
except Exception:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False
    CODECARBON_VERSION = "unavailable"
    print("CodeCarbon not available. Energy values will be set to 0.")

# ============================================================
# HARDWARE / ENVIRONMENT FINGERPRINT
# ============================================================

def get_cpu_model():
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info().get("brand_raw", "Unknown")
    except Exception:
        return platform.processor() or "Unknown"


CPU_MODEL_NAME = get_cpu_model()
CPU_ARCH = platform.machine()
CPU_CORE_COUNT = psutil.cpu_count(logical=False)
CPU_THREAD_COUNT = psutil.cpu_count(logical=True)
CPU_TDP_W = None
SYSTEM_RAM_TOTAL_GB = round(psutil.virtual_memory().total / (1024 ** 3), 2)
OS_NAME = platform.system()
OS_VERSION = platform.version()
OS_ARCHITECTURE = platform.machine()
PYTHON_VERSION = sys.version.split()[0]
PENNYLANE_VERSION = getattr(qml, "__version__", "unknown")


def get_os_full_name():
    system = platform.system()
    architecture = platform.machine()
    if system == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
            )
            product_name = winreg.QueryValueEx(key, "ProductName")[0]
            display_version = winreg.QueryValueEx(key, "DisplayVersion")[0]
            current_build = winreg.QueryValueEx(key, "CurrentBuild")[0]
            return f"{product_name} {display_version} Build {current_build} {architecture}"
        except Exception:
            return f"Windows {platform.release()} {architecture}"
    if system == "Linux":
        return f"Linux {platform.release()} {architecture}"
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]} {architecture}"
    return f"{system} {platform.release()} {architecture}"


OS_FULL_NAME = get_os_full_name()


def make_stable_device_id():
    raw = f"{socket.gethostname()}-{platform.system()}-{platform.machine()}-{CPU_MODEL_NAME}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


DEVICE_UUID = make_stable_device_id()
DEVICE_SHORT = DEVICE_UUID[:8]
DEVICE_LOG_DIR = OUTPUT_ROOT /  f"{DEVICE_SHORT}_quantum"
DEVICE_LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_hostname():
    return socket.gethostname()


def get_cpu_usage():
    try:
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None


def get_ram_usage():
    try:
        return psutil.virtual_memory().percent
    except Exception:
        return None


def get_cpu_freq():
    try:
        freq = psutil.cpu_freq()
        return round(freq.current, 2) if freq else None
    except Exception:
        return None


def get_memory_footprint_mb():
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 4)
    except Exception:
        return None


def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
            if key in temps:
                vals = [x.current for x in temps[key] if x.current and x.current > 0]
                if vals:
                    return round(sum(vals) / len(vals), 1)
    except Exception:
        pass
    return None


def get_cpu_power_draw_w():
    return None


def get_cpu_cores_used():
    try:
        return sum(1 for p in psutil.cpu_percent(percpu=True) if p > 1.0)
    except Exception:
        return None

# ============================================================
# PATH / ARTIFACT DISCOVERY
# ============================================================

def first_existing(paths: list[Path], description: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {description}. Tried:\n" + "\n".join(str(p) for p in paths)
    )


def score_artifact(path: Path, required_words: tuple[str, ...], preferred_words: tuple[str, ...]) -> int:
    name = path.name.lower()
    if any(word not in name for word in required_words):
        return -1
    return sum(10 for word in preferred_words if word in name) - len(name) // 100


def find_best_file(root: Path, suffix: str, required_words=(), preferred_words=()) -> Path:
    candidates = [p for p in root.rglob(f"*{suffix}") if p.is_file()]
    ranked = []
    for p in candidates:
        s = score_artifact(p, tuple(required_words), tuple(preferred_words))
        if s >= 0:
            ranked.append((s, p))
    if not ranked:
        raise FileNotFoundError(
            f"No matching {suffix} artifact found under {root}. "
            f"Required words={required_words}"
        )
    ranked.sort(key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    return ranked[0][1]


def find_model_artifacts(result_root: Path):
    # Prefer q1-named artifacts, but tolerate earlier naming variations.
    try:
        svm_path = find_best_file(
            result_root, ".joblib",
            required_words=("svm",),
            preferred_words=("q1",),
        )
    except FileNotFoundError:
        svm_path = find_best_file(result_root, ".joblib", preferred_words=("q1",))

    try:
        train_states_path = find_best_file(
            result_root, ".npy",
            required_words=("train", "states"),
            preferred_words=("q1",),
        )
    except FileNotFoundError:
        train_states_path = find_best_file(
            result_root, ".npy",
            required_words=("states",),
            preferred_words=("train", "q1"),
        )

    # NPZ metadata is optional for testing, but useful for validation.
    npz_files = list(result_root.rglob("*.npz"))
    model_npz_path = None
    if npz_files:
        model_npz_path = sorted(
            npz_files,
            key=lambda p: (("q1" in p.name.lower()), p.stat().st_mtime),
            reverse=True,
        )[0]

    return svm_path, train_states_path, model_npz_path

# ============================================================
# QASM DISCOVERY / LABEL MATCHING
# ============================================================

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


def read_label(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    matches = re.findall(r"-?\d+", text)
    if not matches:
        raise ValueError(f"No integer label found in {path}")
    label = int(matches[0])
    if label not in range(N_CLASSES):
        raise ValueError(f"Invalid label {label} in {path}")
    return label


def discover_test_samples(data_root: Path, config: dict) -> pd.DataFrame:
    # Usually data lives in an extracted/ subfolder, but search recursively from data_root.
    all_files = [p for p in data_root.rglob("*") if p.is_file()]
    qasm_all = [p for p in all_files if is_qasm_file(p)]

    archive_tokens = [x.lower() for x in config["archive_tokens"]]
    dataset_tokens = [x.lower() for x in config["dataset_tokens"]]

    def path_matches_test(p: Path) -> bool:
        s = str(p).lower()
        if any(tok in s for tok in archive_tokens):
            return True
        return "test" in s and any(tok in s for tok in dataset_tokens) and FIDELITY.lower() in s

    qasm_files = [p for p in qasm_all if path_matches_test(p)]
    if not qasm_files:
        # Last-resort fallback: all QASM files containing test.
        qasm_files = [p for p in qasm_all if "test" in str(p).lower()]
    if not qasm_files:
        raise FileNotFoundError(f"No test QASM files found under {data_root}")

    label_files = []
    for p in all_files:
        s = str(p).lower()
        if "label" not in s:
            continue
        if any(tok in s for tok in archive_tokens) or (
            "test" in s and any(tok in s for tok in dataset_tokens)
        ):
            label_files.append(p)

    if not label_files:
        raise FileNotFoundError(f"No test label files found under {data_root}")

    label_index = {}
    for p in sorted(label_files):
        ident = numeric_identifier(p)
        if ident is not None:
            label_index.setdefault(ident, p)

    records = []
    unmatched = 0
    for qasm_path in qasm_files:
        ident = numeric_identifier(qasm_path)
        if ident is None or ident not in label_index:
            unmatched += 1
            continue
        try:
            label = read_label(label_index[ident])
        except Exception:
            unmatched += 1
            continue
        records.append({
            "sample_id": ident,
            "qasm_path": str(qasm_path),
            "label_path": str(label_index[ident]),
            "label": label,
        })

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(
            f"QASM files were found under {data_root}, but no QASM-label pairs were created."
        )

    if unmatched:
        print(f"Warning: {unmatched} test QASM files were unmatched in {data_root.name}.")

    return df


def balanced_sample(df: pd.DataFrame, samples_per_class: int, seed: int, dataset_name: str) -> pd.DataFrame:
    """Select exactly `samples_per_class` rows from each of the N_CLASSES classes."""
    groups = []
    for class_id in range(N_CLASSES):
        class_rows = df[df["label"] == class_id]
        if len(class_rows) < samples_per_class:
            raise ValueError(
                f"{dataset_name}: class {class_id} has only {len(class_rows)} test "
                f"samples, but {samples_per_class} were requested for the '10 set'."
            )
        groups.append(
            class_rows.sample(
                n=samples_per_class,
                replace=False,
                random_state=seed + class_id,
            )
        )

    return (
        pd.concat(groups, ignore_index=True)
        .sort_values(["label", "sample_id"])
        .reset_index(drop=True)
    )


def select_test_samples(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Either use every discovered test sample (USE_ALL_TEST_SAMPLES = True), or
    a balanced '10 set': exactly TEST_SAMPLES_PER_CLASS samples per class
    (USE_ALL_TEST_SAMPLES = False), for TEST_SAMPLES_PER_CLASS * N_CLASSES
    total samples.
    """
    if df.empty:
        raise ValueError(f"{dataset_name}: no test samples were discovered.")

    if USE_ALL_TEST_SAMPLES:
        selected = (
            df.copy()
            .sort_values(["label", "sample_id"])
            .reset_index(drop=True)
        )
        print(f"{dataset_name}: using ALL {len(selected):,} discovered test samples.")
    else:
        selected = balanced_sample(
            df,
            samples_per_class=TEST_SAMPLES_PER_CLASS,
            seed=SEED,
            dataset_name=dataset_name,
        )
        print(
            f"{dataset_name}: using a balanced '10 set' -> "
            f"{TEST_SAMPLES_PER_CLASS} samples/class x {N_CLASSES} classes "
            f"= {len(selected):,} total test samples."
        )

    print("Class counts:")
    print(selected["label"].value_counts().sort_index().to_string())

    return selected


# ============================================================
# QASM PARSER / EXECUTOR
# ============================================================

def remove_qasm_comments(qasm_text: str) -> str:
    qasm_text = re.sub(r"/\*.*?\*/", "", qasm_text, flags=re.DOTALL)
    return re.sub(r"//.*?$", "", qasm_text, flags=re.MULTILINE)


def split_qasm_statements(qasm_text: str) -> list[str]:
    return [s.strip() for s in remove_qasm_comments(qasm_text).split(";") if s.strip()]


def safe_qasm_angle(expression: str) -> float:
    expression = expression.strip().replace("^", "**")
    if not re.fullmatch(r"[0-9eEpiPI+\-*/().\s*]+", expression):
        raise ValueError(f"Unsupported QASM parameter expression: {expression}")
    value = eval(expression, {"__builtins__": {}}, {"pi": np.pi, "PI": np.pi})
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite QASM parameter: {expression}")
    return value


def parse_parameter_list(text: str | None) -> list[float]:
    if text is None or not text.strip():
        return []
    return [safe_qasm_angle(x) for x in text.split(",")]


def parse_wire_list(operand_text: str, register_sizes: dict[str, int]) -> list[int]:
    offsets = {}
    running = 0
    for name, size in register_sizes.items():
        offsets[name] = running
        running += size

    wires = []
    for operand in [x.strip() for x in operand_text.split(",") if x.strip()]:
        match = re.fullmatch(r"([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]", operand)
        if match is None:
            raise ValueError(f"Unsupported QASM qubit operand: {operand}")
        reg, idx = match.group(1), int(match.group(2))
        if reg not in register_sizes or not 0 <= idx < register_sizes[reg]:
            raise ValueError(f"Invalid QASM qubit operand: {operand}")
        wires.append(offsets[reg] + idx)
    return wires


def parse_mnisq_qasm(qasm_text: str):
    statements = split_qasm_statements(qasm_text)
    register_sizes = {}
    raw_gates = []
    ignored_prefixes = ("openqasm", "include", "creg", "measure", "barrier", "reset")

    for statement in statements:
        lowered = statement.lower().strip()
        qreg_match = re.fullmatch(
            r"qreg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]",
            statement,
            flags=re.IGNORECASE,
        )
        if qreg_match:
            register_sizes[qreg_match.group(1)] = int(qreg_match.group(2))
            continue
        if lowered.startswith(ignored_prefixes):
            continue
        if lowered.startswith(("gate ", "opaque ")):
            raise ValueError("Custom gate declarations were found.")
        raw_gates.append(statement)

    if not register_sizes:
        raise ValueError("No qreg declaration found.")

    total_qubits = sum(register_sizes.values())
    if total_qubits != SOURCE_QUBITS:
        raise ValueError(
            f"Source circuit declares {total_qubits} qubits; expected {SOURCE_QUBITS}."
        )

    gate_pattern = re.compile(
        r"^([A-Za-z_]\w*)(?:\s*\((.*?)\))?\s+(.+)$",
        flags=re.DOTALL,
    )
    operations = []
    for statement in raw_gates:
        match = gate_pattern.fullmatch(statement.strip())
        if match is None:
            raise ValueError(f"Could not parse QASM statement: {statement}")
        operations.append((
            match.group(1).lower(),
            parse_parameter_list(match.group(2)),
            parse_wire_list(match.group(3), register_sizes),
        ))
    return operations


def require_shape(name, params, wires, n_params, n_wires):
    if len(params) != n_params or len(wires) != n_wires:
        raise ValueError(
            f"Gate {name} expects {n_params} parameter(s) and {n_wires} wire(s); "
            f"got {len(params)} and {len(wires)}."
        )


def apply_qasm_gate(name, p, w):
    if name in {"u", "u3"}:
        require_shape(name, p, w, 3, 1); qml.U3(p[0], p[1], p[2], wires=w[0])
    elif name == "u2":
        require_shape(name, p, w, 2, 1); qml.U3(np.pi / 2, p[0], p[1], wires=w[0])
    elif name in {"u1", "p", "phase"}:
        require_shape(name, p, w, 1, 1); qml.PhaseShift(p[0], wires=w[0])
    elif name == "rx":
        require_shape(name, p, w, 1, 1); qml.RX(p[0], wires=w[0])
    elif name == "ry":
        require_shape(name, p, w, 1, 1); qml.RY(p[0], wires=w[0])
    elif name == "rz":
        require_shape(name, p, w, 1, 1); qml.RZ(p[0], wires=w[0])
    elif name == "x":
        require_shape(name, p, w, 0, 1); qml.PauliX(wires=w[0])
    elif name == "y":
        require_shape(name, p, w, 0, 1); qml.PauliY(wires=w[0])
    elif name == "z":
        require_shape(name, p, w, 0, 1); qml.PauliZ(wires=w[0])
    elif name == "h":
        require_shape(name, p, w, 0, 1); qml.Hadamard(wires=w[0])
    elif name == "s":
        require_shape(name, p, w, 0, 1); qml.S(wires=w[0])
    elif name == "sdg":
        require_shape(name, p, w, 0, 1); qml.adjoint(qml.S)(wires=w[0])
    elif name == "t":
        require_shape(name, p, w, 0, 1); qml.T(wires=w[0])
    elif name == "tdg":
        require_shape(name, p, w, 0, 1); qml.adjoint(qml.T)(wires=w[0])
    elif name == "sx":
        require_shape(name, p, w, 0, 1); qml.SX(wires=w[0])
    elif name == "sxdg":
        require_shape(name, p, w, 0, 1); qml.adjoint(qml.SX)(wires=w[0])
    elif name in {"id", "i"}:
        require_shape(name, p, w, 0, 1); qml.Identity(wires=w[0])
    elif name in {"cx", "cnot"}:
        require_shape(name, p, w, 0, 2); qml.CNOT(wires=w)
    elif name == "cy":
        require_shape(name, p, w, 0, 2); qml.CY(wires=w)
    elif name == "cz":
        require_shape(name, p, w, 0, 2); qml.CZ(wires=w)
    elif name == "swap":
        require_shape(name, p, w, 0, 2); qml.SWAP(wires=w)
    elif name == "crx":
        require_shape(name, p, w, 1, 2); qml.CRX(p[0], wires=w)
    elif name == "cry":
        require_shape(name, p, w, 1, 2); qml.CRY(p[0], wires=w)
    elif name == "crz":
        require_shape(name, p, w, 1, 2); qml.CRZ(p[0], wires=w)
    elif name in {"cp", "cu1"}:
        require_shape(name, p, w, 1, 2); qml.ControlledPhaseShift(p[0], wires=w)
    elif name in {"ccx", "toffoli"}:
        require_shape(name, p, w, 0, 3); qml.Toffoli(wires=w)
    else:
        raise ValueError(f"Unsupported QASM gate: {name}")


def reduce_to_q1(source_state: np.ndarray, target_wire: int = 0) -> np.ndarray:
    source_state = np.asarray(source_state, dtype=np.complex128).reshape([2] * SOURCE_QUBITS)
    psi = np.moveaxis(source_state, target_wire, 0).reshape(2, -1)
    rho = psi @ psi.conj().T
    rho = (rho + rho.conj().T) / 2.0
    vals, vecs = np.linalg.eigh(rho)
    q1 = vecs[:, np.argmax(vals)]
    nz = np.flatnonzero(np.abs(q1) > 1e-12)
    if len(nz):
        q1 = q1 * np.exp(-1j * np.angle(q1[nz[0]]))
    norm = np.linalg.norm(q1)
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Invalid q=1 reduced state.")
    return q1 / norm


def execute_qasm_to_q1(qasm_path: Path, cache_root: Path) -> np.ndarray:
    cache_root.mkdir(parents=True, exist_ok=True)
    stat = qasm_path.stat()
    key = f"{qasm_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{PENNYLANE_DEVICE}|q1"
    cache_path = cache_root / f"{hashlib.sha256(key.encode()).hexdigest()}.npy"
    if cache_path.exists():
        x = np.load(cache_path, allow_pickle=False)
        if x.shape == (2,):
            return x

    qasm_text = qasm_path.read_text(encoding="utf-8", errors="ignore")
    operations = parse_mnisq_qasm(qasm_text)
    dev = qml.device(PENNYLANE_DEVICE, wires=SOURCE_QUBITS, shots=None)

    @qml.qnode(dev, interface=None, diff_method=None)
    def circuit_state():
        for name, params, wires in operations:
            apply_qasm_gate(name, params, wires)
        return qml.state()

    source_state = np.asarray(circuit_state(), dtype=np.complex128).reshape(-1)
    if source_state.size != 2 ** SOURCE_QUBITS:
        raise RuntimeError(f"Unexpected source state dimension: {source_state.size}")
    source_state /= np.linalg.norm(source_state)
    q1 = reduce_to_q1(source_state)
    np.save(cache_path, q1, allow_pickle=False)
    return q1

# ============================================================
# KERNEL / PREDICTION QUALITY / ENERGY
# ============================================================

def fidelity_kernel_row(q1_state: np.ndarray, train_states: np.ndarray) -> np.ndarray:
    overlaps = q1_state.conj().reshape(1, -1) @ train_states.T
    return np.clip(np.abs(overlaps) ** 2, 0.0, 1.0)


def prediction_quality(classifier, kernel_row):
    confidence = None
    margin = None
    entropy = None
    try:
        d = np.asarray(classifier.decision_function(kernel_row)).reshape(-1)
        if d.size >= 2:
            top2 = np.sort(d)[-2:]
            margin = float(top2[-1] - top2[-2])
        elif d.size == 1:
            margin = float(abs(d[0]))
    except Exception:
        pass
    try:
        p = np.asarray(classifier.predict_proba(kernel_row)).reshape(-1)
        if p.size:
            p = np.clip(p.astype(float), 1e-12, 1.0)
            confidence = float(np.max(p))
            entropy = float(-np.sum(p * np.log(p)))
    except Exception:
        pass
    return (
        round(confidence, 6) if confidence is not None else None,
        round(margin, 6) if margin is not None else None,
        round(entropy, 6) if entropy is not None else None,
    )


def extract_energy_data(tracker, emissions_value):
    fd = getattr(tracker, "final_emissions_data", None)
    cpu = getattr(fd, "cpu_energy", 0) if fd else 0
    gpu = getattr(fd, "gpu_energy", 0) if fd else 0
    ram = getattr(fd, "ram_energy", 0) if fd else 0
    total = getattr(fd, "energy_consumed", 0) if fd else 0
    ci = None
    if emissions_value is not None and total and total > 0:
        ci = float(emissions_value) / float(total)
    return cpu or 0, gpu or 0, ram or 0, total or 0, ci


def predict_with_energy(classifier, kernel_row, dataset_name):
    tracker = None
    if CODECARBON_AVAILABLE:
        try:
            tracker = EmissionsTracker(
                project_name=f"mnisq_{dataset_name}_q1_fingerprinting",
                output_dir=str(OUTPUT_ROOT / "codecarbon"),
                output_file=f"{dataset_name.lower().replace('-', '_')}.csv",
                log_level="error",
                save_to_file=True,
            )
            tracker.start()
        except Exception:
            tracker = None

    t0 = time.perf_counter()
    pred = int(classifier.predict(kernel_row)[0])
    exec_time = time.perf_counter() - t0

    if tracker is not None:
        try:
            emissions = tracker.stop()
            cpu_e, gpu_e, ram_e, total_e, ci = extract_energy_data(tracker, emissions)
        except Exception:
            emissions, cpu_e, gpu_e, ram_e, total_e, ci = 0, 0, 0, 0, 0, None
    else:
        emissions, cpu_e, gpu_e, ram_e, total_e, ci = 0, 0, 0, 0, 0, None

    return pred, exec_time, cpu_e, gpu_e, ram_e, total_e, emissions, ci

# ============================================================
# TELEMETRY ROW
# ============================================================

def build_row(
    dataset_name,
    model_name,
    sample_index,
    sample_id,
    qasm_path,
    true_label,
    prediction,
    exec_time,
    confidence,
    margin,
    entropy,
    cpu_energy,
    gpu_energy,
    ram_energy,
    total_energy,
    emissions,
    carbon_intensity,
    svm_path,
    train_states_path,
):
    input_tokens = 784
    output_tokens = 1
    total_tokens = input_tokens + output_tokens

    energy_per_token_kwh = 0.0
    joules_per_token = 0.0
    watts_estimated = 0.0
    gpu_energy_pct = 0.0
    cpu_energy_pct = 0.0

    if total_energy and total_energy > 0:
        energy_per_token_kwh = total_energy / total_tokens
        joules_per_token = (total_energy * 3_600_000) / total_tokens
        if exec_time > 0:
            watts_estimated = (total_energy * 3_600_000) / exec_time
        gpu_energy_pct = (gpu_energy / total_energy) * 100
        cpu_energy_pct = (cpu_energy / total_energy) * 100

    return {
        # Identity
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unique_device_id": DEVICE_UUID,
        "device_short_id": DEVICE_SHORT,
        "pc_name": get_hostname(),
        "collection_mode": "automated_edge",

        # Sample
        "dataset": dataset_name,
        "sample_index": sample_index,
        "sample_id": sample_id,
        "qasm_path": qasm_path,
        "true_label": int(true_label),
        "prediction": int(prediction),
        "correct": int(prediction) == int(true_label),

        # Model identity
        "model_type": model_name,
        "parameters": None,
        "model_flops": None,
        "svm_model_path": str(svm_path),
        "train_states_path": str(train_states_path),

        # Prediction quality
        "confidence_score": confidence,
        "logit_margin": margin,
        "entropy": entropy,

        # Timing
        "execution_time_sec": round(exec_time, 10),

        # Energy
        "cpu_energy_kwh": cpu_energy,
        "gpu_energy_kwh": gpu_energy,
        "ram_energy_kwh": ram_energy,
        "total_energy_kwh": total_energy,
        "total_emissions_kg": emissions,
        "carbon_intensity_kgco2_kwh": carbon_intensity,
        "codecarbon_version": CODECARBON_VERSION,

        # Efficiency
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": round(total_tokens / exec_time, 4) if exec_time > 0 else None,
        "joules_per_token": round(joules_per_token, 6),
        "energy_per_token_kwh": round(energy_per_token_kwh, 12),
        "watts_estimated": round(watts_estimated, 4),
        "gpu_energy_pct_of_total": round(gpu_energy_pct, 2),
        "cpu_energy_pct_of_total": round(cpu_energy_pct, 2),

        # CPU
        "cpu_model": CPU_MODEL_NAME,
        "cpu_architecture": CPU_ARCH,
        "cpu_core_count": CPU_CORE_COUNT,
        "cpu_thread_count": CPU_THREAD_COUNT,
        "cpu_core": CPU_CORE_COUNT,
        "cpu_thread": CPU_THREAD_COUNT,
        "cpu_tdp_w": CPU_TDP_W,
        "cpu_usage_pct": get_cpu_usage(),
        "cpu_clock_mhz": get_cpu_freq(),
        "cpu_temp_c": get_cpu_temp(),
        "cpu_power_draw_w": get_cpu_power_draw_w(),
        "cpu_cores_used": get_cpu_cores_used(),

        # GPU: default.qubit CPU simulation
        "gpu_model": "No GPU used by default.qubit",
        "gpu_core": None,
        "gpu_thread": None,
        "gpu_driver_version": None,
        "gpu_compute_capability": None,
        "gpu_power_limit_w": None,
        "gpu_memory_total_mb": None,
        "gpu_power_draw_w": None,
        "gpu_utilization_pct": None,
        "gpu_temp_c": None,
        "gpu_memory_used_mb": None,
        "gpu_sm_clock_mhz": None,
        "gpu_memory_clock_mhz": None,
        "cuda_driver_version": None,
        "cuda_available": False,
        "device_type": PENNYLANE_DEVICE,

        # RAM / environment
        "ram_usage_pct": get_ram_usage(),
        "memory_footprint_mb": get_memory_footprint_mb(),
        "system_ram_total_gb": SYSTEM_RAM_TOTAL_GB,
        "os_name": OS_NAME,
        "os_version": OS_VERSION,
        "os_architecture": OS_ARCHITECTURE,
        "os_full_name": OS_FULL_NAME,
        "python_version": PYTHON_VERSION,
        "torch_version": None,

        # Final model metrics: backfilled per dataset
        "model_accuracy": None,
        "model_precision_weighted": None,
        "model_recall_weighted": None,
        "model_f1_weighted": None,

        # Custom / quantum context
        "quantum_computing": 1,
        "model_under_attack": 0,
        "source_qubits": SOURCE_QUBITS,
        "n_qubits": N_QUBITS,
        "fidelity": FIDELITY,
        "pennylane_device": PENNYLANE_DEVICE,
        "pennylane_version": PENNYLANE_VERSION,
    }

# ============================================================
# DATASET TEST
# ============================================================

def test_one_dataset(dataset_name: str, config: dict) -> pd.DataFrame:
    print("\n" + "=" * 78)
    mode_label = "ALL available" if USE_ALL_TEST_SAMPLES else f"balanced '10 set' ({TEST_SAMPLES_PER_CLASS}/class)"
    print(f"Testing {dataset_name}: {mode_label} q=1 test samples")
    print("=" * 78)

    data_root = first_existing(config["data_roots"], f"{dataset_name} data folder")
    result_root = first_existing(config["result_roots"], f"{dataset_name} q1 results folder")

    svm_path, train_states_path, model_npz_path = find_model_artifacts(result_root)
    print(f"Data root       : {data_root}")
    print(f"Results root    : {result_root}")
    print(f"SVM             : {svm_path.name}")
    print(f"Training states : {train_states_path.name}")
    if model_npz_path:
        print(f"Model metadata  : {model_npz_path.name}")

    classifier = joblib.load(svm_path)
    train_states = np.load(train_states_path, allow_pickle=False)

    if train_states.ndim != 2 or train_states.shape[1] != 2 ** N_QUBITS:
        raise ValueError(
            f"{dataset_name}: expected q=1 training states with shape (N, 2), "
            f"but found {train_states.shape} in {train_states_path}."
        )

    discovered = discover_test_samples(data_root, config)
    selected = select_test_samples(discovered, dataset_name)
    print(f"Total selected test samples: {len(selected):,}")

    tag = "10set" if not USE_ALL_TEST_SAMPLES else "all"
    # selected.to_csv(
    #     OUTPUT_ROOT / f"selected_{dataset_name.lower().replace('-', '_')}_{tag}.csv",
    #     index=False,
    # )

    cache_root = result_root / "fingerprinting_test_state_cache_q1"
    rows = []

    for i, sample in tqdm(
        selected.iterrows(),
        total=len(selected),
        desc=f"{dataset_name} fingerprinting",
        unit="sample",
    ):
        qasm_path = Path(sample["qasm_path"])

        # Circuit simulation + q1 reduction is required to build the kernel row.
        # execution_time_sec below intentionally measures classifier prediction,
        # matching the user's earlier telemetry pattern.
        q1_state = execute_qasm_to_q1(qasm_path, cache_root)
        kernel_row = fidelity_kernel_row(q1_state, train_states)

        confidence, margin, entropy = prediction_quality(classifier, kernel_row)

        pred, exec_time, cpu_e, gpu_e, ram_e, total_e, emissions, ci = predict_with_energy(
            classifier,
            kernel_row,
            dataset_name,
        )

        rows.append(build_row(
            dataset_name=dataset_name,
            model_name=f"MNISQ_{dataset_name}_QuantumKernelSVM_q1",
            sample_index=i,
            sample_id=sample["sample_id"],
            qasm_path=str(qasm_path),
            true_label=int(sample["label"]),
            prediction=pred,
            exec_time=exec_time,
            confidence=confidence,
            margin=margin,
            entropy=entropy,
            cpu_energy=cpu_e,
            gpu_energy=gpu_e,
            ram_energy=ram_e,
            total_energy=total_e,
            emissions=emissions,
            carbon_intensity=ci,
            svm_path=svm_path,
            train_states_path=train_states_path,
        ))

    df = pd.DataFrame(rows)

    # Backfill model-level metrics into all rows for this dataset.
    y_true = df["true_label"].astype(int)
    y_pred = df["prediction"].astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    df["model_accuracy"] = float(acc)
    df["model_precision_weighted"] = float(prec)
    df["model_recall_weighted"] = float(rec)
    df["model_f1_weighted"] = float(f1)

    dataset_csv = (
        DEVICE_LOG_DIR
        / f"mnisq_q1_fingerprint_{dataset_name.lower().replace('-', '_')}_{tag}.csv"
    )
    df.to_csv(dataset_csv, index=False)

    print(
        f"{dataset_name} -> Accuracy={acc:.4f}, Precision={prec:.4f}, "
        f"Recall={rec:.4f}, F1={f1:.4f}"
    )
    print(f"Saved: {dataset_csv}")

    return df

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("MNISQ q=1 MULTI-DATASET FINGERPRINTING TEST")
    print("Datasets: MNIST, FashionMNIST, Kuzushiji-MNIST")
    if USE_ALL_TEST_SAMPLES:
        print("Mode: ALL discovered test samples per dataset")
        print("Dataset test counts are allowed to differ.")
    else:
        print(
            f"Mode: balanced '10 set' -> {TEST_SAMPLES_PER_CLASS} samples/class "
            f"x {N_CLASSES} classes = {TEST_SAMPLES_PER_CLASS * N_CLASSES} total "
            f"per dataset"
        )
    print(f"Device ID: {DEVICE_SHORT}")
    print("=" * 78)

    all_results = []
    failures = []

    for dataset_name, config in DATASET_CONFIGS.items():
        try:
            all_results.append(test_one_dataset(dataset_name, config))
        except Exception as exc:
            failures.append({"dataset": dataset_name, "error": repr(exc)})
            print(f"\n[{dataset_name} FAILED] {type(exc).__name__}: {exc}")

    if all_results:
        print('success')
        # combined = pd.concat(all_results, ignore_index=True)

        # combined_path = (
        #     DEVICE_LOG_DIR
        #     / "mnisq_q1_fingerprinting_all_3datasets_all_samples.csv"
        # )
        # combined.to_csv(combined_path, index=False)

        # summary = (
        #     combined.groupby("dataset")
        #     .agg(
        #         samples=("prediction", "size"),
        #         accuracy=("correct", "mean"),
        #         avg_execution_time_sec=("execution_time_sec", "mean"),
        #         total_execution_time_sec=("execution_time_sec", "sum"),
        #         avg_total_energy_kwh=("total_energy_kwh", "mean"),
        #         total_energy_kwh=("total_energy_kwh", "sum"),
        #     )
        #     .reset_index()
        # )

        # summary_path = (
        #     DEVICE_LOG_DIR
        #     / "mnisq_q1_fingerprinting_summary_all_samples.csv"
        # )
        # summary.to_csv(summary_path, index=False)

        # print("\n" + "=" * 78)
        # print("FULL MULTI-DATASET TEST COMPLETE")
        # print("=" * 78)
        # print(f"Combined rows saved: {len(combined):,}")
        # print(f"Combined CSV: {combined_path}")
        # print(f"Summary CSV : {summary_path}")
        # print("\nDataset summary:")
        # print(summary.to_string(index=False))

    if failures:
        failure_path = DEVICE_LOG_DIR / "mnisq_q1_fingerprinting_failures.csv"
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        print(f"\nFailures saved: {failure_path}")

    if not all_results:
        raise RuntimeError("All three dataset tests failed. Check the failure CSV above.")

    print("\nDone.")



"""
MNISQ Datasets: Combined Download & Extract
================================================================================

Downloads and extracts all three MNISQ base-QASM archives (MNIST,
Fashion-MNIST, Kuzushiji-MNIST) in one run, each into its own directory,
skipping anything already present. Function names are dataset-specific
(ensure_mnist_dataset_available, ensure_fashionmnist_dataset_available,
ensure_kuzushiji_dataset_available) so it's always clear which dataset a
given call is fetching, even though they all share the same generic
download/extract machinery underneath.

Run this once before the three training scripts, or run it standalone
just to pre-fetch/verify all three datasets.

Install once:
pip install requests tqdm
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

# =============================================================================
# SHARED CONFIGURATION
# =============================================================================

FIDELITY = "f90"  # "f80", "f90", or "f95" -- must match what the training scripts expect

PROJECT_ROOT = Path.cwd()

OFFICIAL_BASE_URL = (
    "https://qulacs-quantum-datasets.s3.us-west-1.amazonaws.com"
)

# Per-dataset folders and archive names, matching each training script's
# own DATA_ROOT / DOWNLOAD_ROOT / EXTRACT_ROOT layout exactly, so the
# training scripts find the data without any path changes.
MNIST_DATA_ROOT = PROJECT_ROOT / "mnisq_mnist_data"
MNIST_DOWNLOAD_ROOT = MNIST_DATA_ROOT / "downloads"
MNIST_EXTRACT_ROOT = MNIST_DATA_ROOT / "extracted"
MNIST_TRAIN_ARCHIVE_NAME = f"base_train_orig_mnist_784_{FIDELITY}.zip"
MNIST_TEST_ARCHIVE_NAME = f"base_test_mnist_784_{FIDELITY}.zip"

FASHIONMNIST_DATA_ROOT = PROJECT_ROOT / "mnisq_fashionmnist_data"
FASHIONMNIST_DOWNLOAD_ROOT = FASHIONMNIST_DATA_ROOT / "downloads"
FASHIONMNIST_EXTRACT_ROOT = FASHIONMNIST_DATA_ROOT / "extracted"
FASHIONMNIST_TRAIN_ARCHIVE_NAME = f"base_train_orig_Fashion-MNIST_{FIDELITY}.zip"
FASHIONMNIST_TEST_ARCHIVE_NAME = f"base_test_Fashion-MNIST_{FIDELITY}.zip"

KUZUSHIJI_DATA_ROOT = PROJECT_ROOT / "mnisq_kuzushiji_data"
KUZUSHIJI_DOWNLOAD_ROOT = KUZUSHIJI_DATA_ROOT / "downloads"
KUZUSHIJI_EXTRACT_ROOT = KUZUSHIJI_DATA_ROOT / "extracted"
KUZUSHIJI_TRAIN_ARCHIVE_NAME = f"base_train_orig_Kuzushiji-MNIST_{FIDELITY}.zip"
KUZUSHIJI_TEST_ARCHIVE_NAME = f"base_test_Kuzushiji-MNIST_{FIDELITY}.zip"

for folder in [
    MNIST_DATA_ROOT, MNIST_DOWNLOAD_ROOT, MNIST_EXTRACT_ROOT,
    FASHIONMNIST_DATA_ROOT, FASHIONMNIST_DOWNLOAD_ROOT, FASHIONMNIST_EXTRACT_ROOT,
    KUZUSHIJI_DATA_ROOT, KUZUSHIJI_DOWNLOAD_ROOT, KUZUSHIJI_EXTRACT_ROOT,
]:
    folder.mkdir(parents=True, exist_ok=True)


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
            f"the server ({human_size(expected_remote_size)}); "
            f"re-downloading."
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
    """Extract ZIP while preventing path-traversal (zip-slip) entries."""
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


def archive_marker(extract_root: Path, archive_name: str) -> Path:
    """
    A small marker file dropped after successful extraction. Its presence
    means "already extracted" so re-running this script is a fast no-op
    instead of re-unzipping thousands of files every time.
    """
    return extract_root / f".{archive_name}.extracted"


def ensure_archive_downloaded_and_extracted(
    archive_name: str,
    url: str,
    download_root: Path,
    extract_root: Path,
) -> None:
    """
    Generic worker: downloads `archive_name` from `url` into `download_root`
    only if not already present, then extracts it into `extract_root` only
    if not already extracted (per the marker file). Safe to call repeatedly
    -- fully idempotent. Each dataset-specific ensure_*_dataset_available()
    function below calls this twice (train + test archive).
    """
    archive_path = download_root / archive_name
    marker_path = archive_marker(extract_root, archive_name)

    if marker_path.exists():
        print(f"Already extracted, skipping: {archive_name}")
        return

    download_file(url, archive_path)

    print(f"\nExtracting {archive_path.name}...")
    safe_extract_zip(archive_path, extract_root)

    marker_path.write_text(
        f"Extracted from {archive_path}\n",
        encoding="utf-8",
    )

    print(f"Extraction complete: {archive_path.name}")


# =============================================================================
# DATASET-SPECIFIC ENTRY POINTS
# =============================================================================

def ensure_mnist_dataset_available() -> None:
    print("=" * 78)
    print("Checking MNISQ MNIST dataset")
    print(f"Data root: {MNIST_DATA_ROOT.resolve()}")
    print("=" * 78)

    train_url = f"{OFFICIAL_BASE_URL}/{MNIST_TRAIN_ARCHIVE_NAME}"
    test_url = f"{OFFICIAL_BASE_URL}/{MNIST_TEST_ARCHIVE_NAME}"

    ensure_archive_downloaded_and_extracted(
        MNIST_TRAIN_ARCHIVE_NAME,
        train_url,
        MNIST_DOWNLOAD_ROOT,
        MNIST_EXTRACT_ROOT,
    )
    ensure_archive_downloaded_and_extracted(
        MNIST_TEST_ARCHIVE_NAME,
        test_url,
        MNIST_DOWNLOAD_ROOT,
        MNIST_EXTRACT_ROOT,
    )

    print(f"MNIST dataset ready. Extracted under: {MNIST_EXTRACT_ROOT.resolve()}")


def ensure_fashionmnist_dataset_available() -> None:
    print("=" * 78)
    print("Checking MNISQ Fashion-MNIST dataset")
    print(f"Data root: {FASHIONMNIST_DATA_ROOT.resolve()}")
    print("=" * 78)

    train_url = f"{OFFICIAL_BASE_URL}/{FASHIONMNIST_TRAIN_ARCHIVE_NAME}"
    test_url = f"{OFFICIAL_BASE_URL}/{FASHIONMNIST_TEST_ARCHIVE_NAME}"

    ensure_archive_downloaded_and_extracted(
        FASHIONMNIST_TRAIN_ARCHIVE_NAME,
        train_url,
        FASHIONMNIST_DOWNLOAD_ROOT,
        FASHIONMNIST_EXTRACT_ROOT,
    )
    ensure_archive_downloaded_and_extracted(
        FASHIONMNIST_TEST_ARCHIVE_NAME,
        test_url,
        FASHIONMNIST_DOWNLOAD_ROOT,
        FASHIONMNIST_EXTRACT_ROOT,
    )

    print(f"Fashion-MNIST dataset ready. Extracted under: {FASHIONMNIST_EXTRACT_ROOT.resolve()}")


def ensure_kuzushiji_dataset_available() -> None:
    print("=" * 78)
    print("Checking MNISQ Kuzushiji-MNIST dataset")
    print(f"Data root: {KUZUSHIJI_DATA_ROOT.resolve()}")
    print("=" * 78)

    train_url = f"{OFFICIAL_BASE_URL}/{KUZUSHIJI_TRAIN_ARCHIVE_NAME}"
    test_url = f"{OFFICIAL_BASE_URL}/{KUZUSHIJI_TEST_ARCHIVE_NAME}"

    ensure_archive_downloaded_and_extracted(
        KUZUSHIJI_TRAIN_ARCHIVE_NAME,
        train_url,
        KUZUSHIJI_DOWNLOAD_ROOT,
        KUZUSHIJI_EXTRACT_ROOT,
    )
    ensure_archive_downloaded_and_extracted(
        KUZUSHIJI_TEST_ARCHIVE_NAME,
        test_url,
        KUZUSHIJI_DOWNLOAD_ROOT,
        KUZUSHIJI_EXTRACT_ROOT,
    )

    print(f"Kuzushiji-MNIST dataset ready. Extracted under: {KUZUSHIJI_EXTRACT_ROOT.resolve()}")


def ensure_all_mnisq_datasets_available() -> None:
    """Runs all three dataset checks/downloads/extractions in sequence."""
    ensure_mnist_dataset_available()
    print()
    ensure_fashionmnist_dataset_available()
    print()
    ensure_kuzushiji_dataset_available()

    print("\n" + "=" * 78)
    print("All three MNISQ datasets ready.")
    print(f"  MNIST           -> {MNIST_EXTRACT_ROOT.resolve()}")
    print(f"  Fashion-MNIST   -> {FASHIONMNIST_EXTRACT_ROOT.resolve()}")
    print(f"  Kuzushiji-MNIST -> {KUZUSHIJI_EXTRACT_ROOT.resolve()}")
    print("=" * 78)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ensure_all_mnisq_datasets_available()
    main()