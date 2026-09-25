# ============================================================
# PennyLane Quantum Classifier Sweep with CodeCarbon
# Different circuits + different qubits
# Save/load trained models automatically
# Qubits: 1 to 5
# Circuits:
#   1. RY_RZ_LINEAR
#   2. RX_RY_RING
#   3. HARDWARE_EFFICIENT_CZ
#   4. DATA_REUPLOAD
#
# Dataset: sklearn digits, binary classification 0 vs 1
# Output:
#   logs/quantum_circuit_qubit_sweep_results.csv
#   logs/quantum_circuit_qubit_sweep_metrics.json
#   checkpoints/*.npz
# ============================================================

import os
import sys
import time
import json
import hashlib
import socket
import platform
from pathlib import Path

import psutil
import pandas as pd
from tqdm import tqdm

import numpy as np
import pennylane as qml

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ============================================================
# Config
# ============================================================
def get_cpu_model():
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info().get("brand_raw", "Unknown")
    except Exception:
        return platform.processor() or "Unknown"
    
def get_os_full_name():
    system = platform.system()
    arch = platform.machine()

    if system == "Windows":
        return f"Windows {platform.release()} {platform.version()} {arch}"

    if system == "Linux":
        try:
            os_info = {}
            with open("/etc/os-release", "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        os_info[k] = v.strip('"')
            return f"{os_info.get('PRETTY_NAME', 'Linux')} {arch}"
        except Exception:
            return f"Linux {platform.release()} {arch}"

    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]} {arch}"

    return f"{system} {platform.release()} {arch}"

CPU_MODEL_NAME = get_cpu_model()
OS_FULL_NAME = get_os_full_name()
PYTHON_VERSION = sys.version.split()[0]
PENNYLANE_VERSION = qml.__version__
SYSTEM_RAM_TOTAL_GB = round(psutil.virtual_memory().total / (1024 ** 3), 2)
CPU_CORE_COUNT = psutil.cpu_count(logical=False)
CPU_THREAD_COUNT = psutil.cpu_count(logical=True)

def make_stable_device_id():
    raw = f"{socket.gethostname()}-{platform.system()}-{platform.machine()}-{CPU_MODEL_NAME}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


DEVICE_UUID = make_stable_device_id()
DEVICE_SHORT = DEVICE_UUID[:8]

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "checkpoints"
DEVICE_LOG_DIR = ROOT / f"{DEVICE_SHORT}"

LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
DEVICE_LOG_DIR.mkdir(exist_ok=True)

CSV_PATH = DEVICE_LOG_DIR / "quantum_circuit_qubit_sweep_results.csv"
METRICS_PATH = DEVICE_LOG_DIR / "quantum_circuit_qubit_sweep_metrics.json"
ENERGY_LOG_DIR = LOG_DIR / "energy_logs"

ENERGY_LOG_DIR.mkdir(exist_ok=True)

QUBIT_RANGE = [1, 2, 3, 4, 5,6,7,8,9,10]

CIRCUIT_TYPES = [
    "RY_RZ_LINEAR",
     "RX_RY_RING",
     "HARDWARE_EFFICIENT_CZ",
     "DATA_REUPLOAD",
]

N_LAYERS = 4
EPOCHS = 50
TRAIN_SUBSAMPLE = 120
NUM_INFERENCE_SAMPLES = 100
RANDOM_SEED = 42

# Set True only when you want to retrain everything
FORCE_RETRAIN = False

np.random.seed(RANDOM_SEED)


# ============================================================
# Optional CodeCarbon energy tracking
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
# System helpers
# ============================================================













def get_memory_footprint_mb():
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 4)
    except Exception:
        return None


def get_cpu_usage():
    return psutil.cpu_percent(interval=None)


def get_ram_usage():
    return psutil.virtual_memory().percent


def get_cpu_freq():
    try:
        freq = psutil.cpu_freq()
        return round(freq.current, 2) if freq else None
    except Exception:
        return None


def get_cpu_cores_used():
    try:
        return sum(1 for p in psutil.cpu_percent(percpu=True) if p > 1.0)
    except Exception:
        return None


# ============================================================
# Dataset
# ============================================================

def load_binary_digits_dataset():
    digits = load_digits()

    X = digits.data
    y = digits.target

    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]

    # digit 0 -> -1, digit 1 -> +1
    y = np.array([-1 if label == 0 else 1 for label in y], dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def compress_to_n_qubits(X, n_qubits):
    chunks = np.array_split(X, n_qubits, axis=1)
    compressed = [np.mean(chunk, axis=1) for chunk in chunks]
    return np.stack(compressed, axis=1)


# ============================================================
# Model save/load helpers
# ============================================================

def safe_name(text):
    return text.lower().replace(" ", "_").replace("-", "_")


def get_model_path(circuit_type, n_qubits):
    return MODEL_DIR / f"qclassifier_{safe_name(circuit_type)}_{n_qubits}q.npz"


def save_model(model_path, circuit_type, n_qubits, n_layers, weights, bias, best_loss):
    np.savez(
        model_path,
        circuit_type=circuit_type,
        n_qubits=np.array(n_qubits),
        n_layers=np.array(n_layers),
        weights=weights,
        bias=np.array(bias),
        best_loss=np.array(best_loss),
        pennylane_version=PENNYLANE_VERSION,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print(f"Saved model: {model_path}")


def load_model(model_path):
    data = np.load(model_path, allow_pickle=True)

    weights = data["weights"]
    bias = float(data["bias"])
    best_loss = float(data["best_loss"]) if "best_loss" in data.files else None

    saved_info = {
        "circuit_type": str(data["circuit_type"]) if "circuit_type" in data.files else None,
        "n_qubits": int(data["n_qubits"]) if "n_qubits" in data.files else None,
        "n_layers": int(data["n_layers"]) if "n_layers" in data.files else None,
        "best_loss": best_loss,
        "pennylane_version": str(data["pennylane_version"]) if "pennylane_version" in data.files else None,
        "timestamp": str(data["timestamp"]) if "timestamp" in data.files else None,
    }

    return weights, bias, saved_info


# ============================================================
# Circuit definitions
# ============================================================

def apply_circuit(circuit_type, x, weights, n_qubits, n_layers):
    if circuit_type == "RY_RZ_LINEAR":
        for q in range(n_qubits):
            qml.RY(float(x[q]), wires=q)
            qml.RZ(float(x[q]), wires=q)

        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.Rot(
                    float(weights[layer, q, 0]),
                    float(weights[layer, q, 1]),
                    float(weights[layer, q, 2]),
                    wires=q,
                )

            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])

    elif circuit_type == "RX_RY_RING":
        for q in range(n_qubits):
            qml.RX(float(x[q]), wires=q)
            qml.RY(float(x[q]), wires=q)

        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RX(float(weights[layer, q, 0]), wires=q)
                qml.RY(float(weights[layer, q, 1]), wires=q)
                qml.RZ(float(weights[layer, q, 2]), wires=q)

            if n_qubits > 1:
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])

    elif circuit_type == "HARDWARE_EFFICIENT_CZ":
        for q in range(n_qubits):
            qml.Hadamard(wires=q)
            qml.RY(float(x[q]), wires=q)

        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RX(float(weights[layer, q, 0]), wires=q)
                qml.RY(float(weights[layer, q, 1]), wires=q)
                qml.RZ(float(weights[layer, q, 2]), wires=q)

            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    qml.CZ(wires=[q, q + 1])

    elif circuit_type == "DATA_REUPLOAD":
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(float(x[q]), wires=q)
                qml.RZ(float(x[q]), wires=q)

                qml.Rot(
                    float(weights[layer, q, 0]),
                    float(weights[layer, q, 1]),
                    float(weights[layer, q, 2]),
                    wires=q,
                )

            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])

    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")


def create_quantum_model(n_qubits, n_layers, circuit_type):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def quantum_circuit(x, weights):
        apply_circuit(circuit_type, x, weights, n_qubits, n_layers)
        return qml.expval(qml.PauliZ(0))

    def score(x, weights, bias):
        return float(quantum_circuit(x, weights)) + float(bias)

    def predict_label(x, weights, bias):
        s = score(x, weights, bias)
        return 1 if s >= 0 else -1

    return quantum_circuit, score, predict_label


# ============================================================
# Circuit statistics
# ============================================================

def estimate_gate_count(circuit_type, n_qubits, n_layers):
    if circuit_type == "RY_RZ_LINEAR":
        encoding_gates = 2 * n_qubits
        trainable_gates = n_layers * n_qubits
        entangling_gates = n_layers * max(0, n_qubits - 1)
        entangling_type = "CNOT_LINEAR"

    elif circuit_type == "RX_RY_RING":
        encoding_gates = 2 * n_qubits
        trainable_gates = 3 * n_layers * n_qubits
        entangling_gates = n_layers * n_qubits if n_qubits > 1 else 0
        entangling_type = "CNOT_RING"

    elif circuit_type == "HARDWARE_EFFICIENT_CZ":
        encoding_gates = 2 * n_qubits
        trainable_gates = 3 * n_layers * n_qubits
        entangling_gates = n_layers * max(0, n_qubits - 1)
        entangling_type = "CZ_LINEAR"

    elif circuit_type == "DATA_REUPLOAD":
        encoding_gates = 2 * n_layers * n_qubits
        trainable_gates = n_layers * n_qubits
        entangling_gates = n_layers * max(0, n_qubits - 1)
        entangling_type = "CNOT_LINEAR"

    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")

    total_gates = encoding_gates + trainable_gates + entangling_gates

    return {
        "encoding_gates": encoding_gates,
        "trainable_gates": trainable_gates,
        "entangling_gates": entangling_gates,
        "total_gates": total_gates,
        "entangling_type": entangling_type,
    }


def estimate_circuit_depth(circuit_type, n_qubits, n_layers):
    if circuit_type == "RY_RZ_LINEAR":
        return 2 + n_layers * (1 + (1 if n_qubits > 1 else 0))

    if circuit_type == "RX_RY_RING":
        return 2 + n_layers * (3 + (1 if n_qubits > 1 else 0))

    if circuit_type == "HARDWARE_EFFICIENT_CZ":
        return 2 + n_layers * (3 + (1 if n_qubits > 1 else 0))

    if circuit_type == "DATA_REUPLOAD":
        return n_layers * (3 + (1 if n_qubits > 1 else 0))

    return None


# ============================================================
# Gradient-free training
# ============================================================

def mse_loss(X, y, score_fn, weights, bias):
    preds = np.array([score_fn(x, weights, bias) for x in X])
    return float(np.mean((preds - y) ** 2))


def accuracy(X, y, predict_fn, weights, bias):
    preds = np.array([predict_fn(x, weights, bias) for x in X])
    return accuracy_score(y.astype(int), preds.astype(int))


def evaluate_setting(X_test_q, y_test, weights, bias, predict_fn):
    test_preds = np.array([predict_fn(x, weights, bias) for x in X_test_q])

    acc = accuracy_score(y_test.astype(int), test_preds.astype(int))

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_test.astype(int),
        test_preds.astype(int),
        average="weighted",
        zero_division=0,
    )

    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_test.astype(int),
        test_preds.astype(int),
        average="macro",
        zero_division=0,
    )

    return test_preds, {
        "accuracy": float(acc),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(precision_m),
        "recall_macro": float(recall_m),
        "macro_f1": float(f1_m),
    }


def train_or_load_model(
    circuit_type,
    n_qubits,
    X_train,
    y_train,
    X_test,
    y_test,
    n_layers=N_LAYERS,
    epochs=EPOCHS,
):
    print("\n" + "=" * 80)
    print(f"Setting: circuit={circuit_type}, qubits={n_qubits}")
    print("=" * 80)

    X_train_q = compress_to_n_qubits(X_train, n_qubits)
    X_test_q = compress_to_n_qubits(X_test, n_qubits)

    train_limit = min(TRAIN_SUBSAMPLE, len(X_train_q))
    X_train_small = X_train_q[:train_limit]
    y_train_small = y_train[:train_limit]

    _, score_fn, predict_fn = create_quantum_model(
        n_qubits=n_qubits,
        n_layers=n_layers,
        circuit_type=circuit_type,
    )

    model_path = get_model_path(circuit_type, n_qubits)

    if model_path.exists() and not FORCE_RETRAIN:
        print(f"Found trained model. Loading: {model_path}")
        weights, bias, saved_info = load_model(model_path)

        print("Loaded model info:")
        print(saved_info)

    else:
        if FORCE_RETRAIN and model_path.exists():
            print(f"FORCE_RETRAIN=True. Retraining existing model: {model_path}")
        else:
            print(f"No saved model found. Training new model: {model_path}")

        weights = 0.01 * np.random.randn(n_layers, n_qubits, 3)
        bias = 0.0

        best_loss = mse_loss(X_train_small, y_train_small, score_fn, weights, bias)

        step_size = 0.25
        bias_step = 0.05

        print(f"Initial loss: {best_loss:.4f}")

        for epoch in range(epochs):
            candidate_weights = weights + step_size * np.random.randn(*weights.shape)
            candidate_bias = bias + bias_step * np.random.randn()

            candidate_loss = mse_loss(
                X_train_small,
                y_train_small,
                score_fn,
                candidate_weights,
                candidate_bias,
            )

            if candidate_loss < best_loss:
                weights = candidate_weights
                bias = candidate_bias
                best_loss = candidate_loss

            step_size *= 0.985
            bias_step *= 0.985

            if (epoch + 1) % 10 == 0:
                train_acc = accuracy(
                    X_train_small,
                    y_train_small,
                    predict_fn,
                    weights,
                    bias,
                )
                test_acc = accuracy(
                    X_test_q,
                    y_test,
                    predict_fn,
                    weights,
                    bias,
                )

                print(
                    f"Epoch {epoch+1:03d} | "
                    f"Loss: {best_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Test Acc: {test_acc:.4f}"
                )

        save_model(
            model_path=model_path,
            circuit_type=circuit_type,
            n_qubits=n_qubits,
            n_layers=n_layers,
            weights=weights,
            bias=bias,
            best_loss=best_loss,
        )

    test_preds, metrics = evaluate_setting(
        X_test_q,
        y_test,
        weights,
        bias,
        predict_fn,
    )

    model_metrics = {
        "circuit_type": circuit_type,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        **metrics,
    }

    print("\nEvaluation")
    print("Circuit:", circuit_type)
    print("Qubits :", n_qubits)
    print("Accuracy:", model_metrics["accuracy"])
    print("Macro-F1:", model_metrics["macro_f1"])
    print("Confusion Matrix:")
    print(confusion_matrix(y_test.astype(int), test_preds.astype(int)))
    print(
        classification_report(
            y_test.astype(int),
            test_preds.astype(int),
            target_names=["Digit 0", "Digit 1"],
        )
    )

    return X_test_q, y_test, weights, bias, model_metrics, score_fn, predict_fn


# ============================================================
# Prediction quality
# ============================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def prediction_quality(score):
    prob_pos = sigmoid(score)
    prob_neg = 1.0 - prob_pos

    confidence = float(max(prob_pos, prob_neg))
    entropy = float(
        -(prob_pos * np.log(prob_pos + 1e-12) + prob_neg * np.log(prob_neg + 1e-12))
    )
    margin = float(abs(score))

    return round(confidence, 6), round(margin, 6), round(entropy, 6)


# ============================================================
# CodeCarbon energy tracking
# ============================================================

def run_with_energy_tracking(
    inference_fn,
    *args,
    output_dir=None,
    project_name="pennylane_quantum_circuit_qubit_sweep",
    output_file="codecarbon_quantum_sweep.csv",
    **kwargs,
):
    if output_dir is None:
        output_dir = str(ENERGY_LOG_DIR)

    os.makedirs(output_dir, exist_ok=True)

    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=output_dir,
            output_file=output_file,
            log_level="error",
            save_to_file=True,
            measure_power_secs=1,
        )

        tracker.start()

        t0 = time.perf_counter()
        result = inference_fn(*args, **kwargs)
        exec_time = time.perf_counter() - t0

        emissions_value = tracker.stop()

        final_data = getattr(tracker, "final_emissions_data", None)

        cpu_energy = getattr(final_data, "cpu_energy", 0) if final_data else 0
        gpu_energy = getattr(final_data, "gpu_energy", 0) if final_data else 0
        ram_energy = getattr(final_data, "ram_energy", 0) if final_data else 0
        total_energy = getattr(final_data, "energy_consumed", 0) if final_data else 0

        cpu_energy = cpu_energy or 0
        gpu_energy = gpu_energy or 0
        ram_energy = ram_energy or 0
        total_energy = total_energy or 0
        emissions_value = emissions_value or 0

        carbon_intensity = None
        if total_energy > 0:
            carbon_intensity = round(emissions_value / total_energy, 8)

        return {
            "result": result,
            "execution_time_sec": exec_time,
            "cpu_energy_kwh": cpu_energy,
            "gpu_energy_kwh": gpu_energy,
            "ram_energy_kwh": ram_energy,
            "total_energy_kwh": total_energy,
            "total_emissions_kg": emissions_value,
            "carbon_intensity_kgco2_kwh": carbon_intensity,
        }

    # Fallback if CodeCarbon is not installed
    t0 = time.perf_counter()
    result = inference_fn(*args, **kwargs)
    exec_time = time.perf_counter() - t0

    return {
        "result": result,
        "execution_time_sec": exec_time,
        "cpu_energy_kwh": 0,
        "gpu_energy_kwh": 0,
        "ram_energy_kwh": 0,
        "total_energy_kwh": 0,
        "total_emissions_kg": 0,
        "carbon_intensity_kgco2_kwh": None,
    }


# ============================================================
# CSV logging
# ============================================================

def build_row(
    circuit_type,
    n_qubits,
    n_layers,
    sample_index,
    true_label,
    prediction,
    score,
    result_info,
    model_metrics,
):
    confidence, margin, entropy = prediction_quality(score)

    exec_time = result_info["execution_time_sec"]
    gate_info = estimate_gate_count(circuit_type, n_qubits, n_layers)

    input_tokens = n_qubits
    output_tokens = 1
    total_tokens = input_tokens + output_tokens

    total_energy = result_info["total_energy_kwh"] or 0.0

    joules_per_token = 0.0
    energy_per_token_kwh = 0.0
    watts_estimated = 0.0

    if total_energy > 0 and total_tokens > 0:
        energy_per_token_kwh = round(total_energy / total_tokens, 12)
        joules_total = total_energy * 3_600_000
        joules_per_token = round(joules_total / total_tokens, 8)

        if exec_time > 0:
            watts_estimated = round(joules_total / exec_time, 8)

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unique_device_id": DEVICE_UUID,
        "device_short_id": DEVICE_SHORT,
        "pc_name": socket.gethostname(),
        "collection_mode": "quantum_circuit_qubit_sweep_codecarbon",

        "sample_index": sample_index,
        "true_label": int(true_label),
        "prediction": int(prediction),
        "correct": int(prediction) == int(true_label),

        "model_type": f"{circuit_type}_{n_qubits}Q",
        "framework": "PennyLane",
        "pennylane_version": PENNYLANE_VERSION,
        "quantum_device": "default.qubit",

        "circuit_type": circuit_type,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "parameter_count": n_layers * n_qubits * 3 + 1,
        "circuit_depth_estimate": estimate_circuit_depth(circuit_type, n_qubits, n_layers),

        "encoding_gates": gate_info["encoding_gates"],
        "trainable_gates": gate_info["trainable_gates"],
        "entangling_gates": gate_info["entangling_gates"],
        "entangling_type": gate_info["entangling_type"],
        "total_gates": gate_info["total_gates"],

        "raw_score": round(float(score), 8),
        "confidence_score": confidence,
        "score_margin": margin,
        "entropy": entropy,

        "execution_time_sec": round(exec_time, 10),

        # Energy
        "cpu_energy_kwh": result_info["cpu_energy_kwh"],
        "gpu_energy_kwh": result_info["gpu_energy_kwh"],
        "ram_energy_kwh": result_info["ram_energy_kwh"],
        "total_energy_kwh": result_info["total_energy_kwh"],
        "total_emissions_kg": result_info["total_emissions_kg"],
        "carbon_intensity_kgco2_kwh": result_info["carbon_intensity_kgco2_kwh"],
        "codecarbon_version": CODECARBON_VERSION,

        # Efficiency
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": round(total_tokens / exec_time, 4) if exec_time > 0 else None,
        "joules_per_token": joules_per_token,
        "energy_per_token_kwh": energy_per_token_kwh,
        "watts_estimated": watts_estimated,

        # CPU/RAM
        "cpu_model": CPU_MODEL_NAME,
        "cpu_core_count": CPU_CORE_COUNT,
        "cpu_thread_count": CPU_THREAD_COUNT,
        "cpu_usage_pct": get_cpu_usage(),
        "cpu_clock_mhz": get_cpu_freq(),
        "cpu_cores_used": get_cpu_cores_used(),
        "ram_usage_pct": get_ram_usage(),
        "memory_footprint_mb": get_memory_footprint_mb(),
        "system_ram_total_gb": SYSTEM_RAM_TOTAL_GB,

        # OS/environment
        "os_full_name": OS_FULL_NAME,
        "os_name": platform.system(),
        "os_architecture": platform.machine(),
        "python_version": PYTHON_VERSION,

        # Model metrics
        "model_accuracy": model_metrics.get("accuracy"),
        "model_precision_weighted": model_metrics.get("precision_weighted"),
        "model_recall_weighted": model_metrics.get("recall_weighted"),
        "model_f1_weighted": model_metrics.get("f1_weighted"),
        "model_precision_macro": model_metrics.get("precision_macro"),
        "model_recall_macro": model_metrics.get("recall_macro"),
        "model_macro_f1": model_metrics.get("macro_f1"),
    }


def append_rows(rows, path):
    if not rows:
        return

    new_df = pd.DataFrame(rows)

    if path.exists():
        old_df = pd.read_csv(path, on_bad_lines="skip")

        for col in new_df.columns:
            if col not in old_df.columns:
                old_df[col] = None

        for col in old_df.columns:
            if col not in new_df.columns:
                new_df[col] = None

        new_df = new_df[old_df.columns]
        final_df = pd.concat([old_df, new_df], ignore_index=True)
        final_df.to_csv(path, index=False)
    else:
        new_df.to_csv(path, index=False)


# ============================================================
# Inference collection
# ============================================================

def collect_for_setting(
    circuit_type,
    n_qubits,
    X_test_q,
    y_test,
    weights,
    bias,
    model_metrics,
    score_fn,
    predict_fn,
    num_samples=NUM_INFERENCE_SAMPLES,
    flush_every=10,
):
    limit = min(num_samples, len(X_test_q))
    rows = []

    print(f"\nCollecting logs for circuit={circuit_type}, qubits={n_qubits}")
    print(f"Output CSV: {CSV_PATH}")

    def quantum_inference(x):
        s = score_fn(x, weights, bias)
        p = 1 if s >= 0 else -1
        return p, float(s)

    for i in tqdm(
        range(limit),
        desc=f"{circuit_type}-{n_qubits}Q inference",
        unit="sample",
    ):
        x = X_test_q[i]
        true_label = y_test[i]

        result_info = run_with_energy_tracking(
            quantum_inference,
            x,
            output_dir=str(ENERGY_LOG_DIR),
            project_name="pennylane_quantum_circuit_qubit_sweep",
            output_file="codecarbon_quantum_sweep.csv",
        )

        pred, score = result_info["result"]

        row = build_row(
            circuit_type=circuit_type,
            n_qubits=n_qubits,
            n_layers=N_LAYERS,
            sample_index=i,
            true_label=true_label,
            prediction=pred,
            score=score,
            result_info=result_info,
            model_metrics=model_metrics,
        )

        rows.append(row)

        if (i + 1) % flush_every == 0:
            append_rows(rows, CSV_PATH)
            rows = []

    if rows:
        append_rows(rows, CSV_PATH)

    print(f"Finished logs for circuit={circuit_type}, qubits={n_qubits}")


# ============================================================
# Main
# ============================================================

def main():
    print("PennyLane Quantum Circuit + Qubit Sweep with CodeCarbon")
    print("ROOT:", ROOT)
    print("FORCE_RETRAIN:", FORCE_RETRAIN)
    print("CodeCarbon available:", CODECARBON_AVAILABLE)
    print("CodeCarbon version:", CODECARBON_VERSION)
    print("Device UUID:", DEVICE_UUID)
    print("Device short:", DEVICE_SHORT)
    print("OS:", OS_FULL_NAME)
    print("CPU:", CPU_MODEL_NAME)
    print("RAM GB:", SYSTEM_RAM_TOTAL_GB)
    print("PennyLane:", PENNYLANE_VERSION)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = load_binary_digits_dataset()

    print("\nDataset loaded")
    print("Train:", X_train_raw.shape)
    print("Test :", X_test_raw.shape)

    all_metrics = {}

    for circuit_type in CIRCUIT_TYPES:
        for n_qubits in QUBIT_RANGE:
            (
                X_test_q,
                y_test,
                weights,
                bias,
                model_metrics,
                score_fn,
                predict_fn,
            ) = train_or_load_model(
                circuit_type=circuit_type,
                n_qubits=n_qubits,
                X_train=X_train_raw,
                y_train=y_train_raw,
                X_test=X_test_raw,
                y_test=y_test_raw,
            )

            key = f"{circuit_type}_{n_qubits}Q"
            all_metrics[key] = model_metrics

            collect_for_setting(
                circuit_type=circuit_type,
                n_qubits=n_qubits,
                X_test_q=X_test_q,
                y_test=y_test,
                weights=weights,
                bias=bias,
                model_metrics=model_metrics,
                score_fn=score_fn,
                predict_fn=predict_fn,
                num_samples=NUM_INFERENCE_SAMPLES,
                flush_every=10,
            )

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    print("\nAll circuit and qubit settings completed.")
    print("CSV saved to:", CSV_PATH)
    print("Metrics saved to:", METRICS_PATH)
    print("CodeCarbon logs saved to:", ENERGY_LOG_DIR)


main()