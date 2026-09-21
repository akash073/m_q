# -*- coding: utf-8 -*-
"""
Combined Qwen2-VL + Moondream2 telemetry test
Datasets: MNIST, FashionMNIST, KMNIST

Key design:
- Both models use EXACTLY the same query for a given dataset.
- Both models use the same test samples (indices 0..N-1).
- Models are loaded and tested SEQUENTIALLY to reduce RAM/VRAM use.
- One telemetry CSV is saved per model x dataset.
- A combined summary CSV is also saved.

Set sample count from shell:
    NUM_TEST_SAMPLES=100 python qwen2vl_moondream2_three_dataset_telemetry.py

Set NUM_TEST_SAMPLES=-1 for the complete 10,000-image test split.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import os
import platform
import re
import socket
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from torchvision.datasets import MNIST, FashionMNIST, KMNIST
from tqdm.auto import tqdm

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    DynamicCache,
    GenerationConfig,
    GenerationMixin,
    Qwen2VLForConditionalGeneration,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

_raw_num_samples = int(os.getenv("NUM_TEST_SAMPLES", "10000"))
NUM_TEST_SAMPLES = None if _raw_num_samples < 0 else _raw_num_samples

DATA_ROOT = Path("./classical_data")

OUTPUT_ROOT = Path.cwd() / "test_results"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

QWEN_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

MOONDREAM_MODEL_ID = "fal/moondream2-docci-instruct"
MOONDREAM_REVISION = "3ec40c7b6b5d87bc0c51edee45e21f5f29b449d8"

CLASS_NAMES = [str(i) for i in range(10)]

# IMPORTANT:
# For each dataset, BOTH Qwen and Moondream receive the exact same
# query string below. This makes the prompt condition comparable.
DATASET_CONFIGS = {
    "MNIST": {
        "key": "mnist",
        "dataset_class": MNIST,
        "query": (
            "Classify this MNIST handwritten digit image. "
            "Answer with exactly one digit: "
            "0, 1, 2, 3, 4, 5, 6, 7, 8, or 9."
        ),
    },
    "FashionMNIST": {
        "key": "fashionmnist",
        "dataset_class": FashionMNIST,
        "query": (
            "Classify this FashionMNIST image into exactly one class ID. "
            "Use this mapping: "
            "0=T-shirt/top, 1=trouser, 2=pullover, 3=dress, 4=coat, "
            "5=sandal, 6=shirt, 7=sneaker, 8=bag, 9=ankle boot. "
            "Answer with exactly one digit from 0 to 9."
        ),
    },
    "KMNIST": {
        "key": "kmnist",
        "dataset_class": KMNIST,
        "query": (
            "Classify this KMNIST Kuzushiji character image into exactly "
            "one class ID from 0 to 9. "
            "Use this mapping: "
            "0=o, 1=ki, 2=su, 3=tsu, 4=na, "
            "5=ha, 6=ma, 7=ya, 8=re, 9=wo. "
            "Answer with exactly one digit from 0 to 9."
        ),
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    DTYPE = (
        torch.bfloat16
        if hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
        else torch.float16
    )
else:
    DTYPE = torch.float32

torch.set_grad_enabled(False)


# ============================================================
# 2. OPTIONAL TELEMETRY PACKAGES
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
    print("CodeCarbon unavailable. Energy values will be 0.")

try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    NVML_HANDLE = (
        pynvml.nvmlDeviceGetHandleByIndex(0)
        if torch.cuda.is_available()
        else None
    )
except Exception:
    NVML_AVAILABLE = False
    NVML_HANDLE = None
    print("NVML unavailable.")


# ============================================================
# 3. HARDWARE / ENVIRONMENT
# ============================================================

def get_cpu_model():
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info().get("brand_raw", "Unknown")
    except Exception:
        return platform.processor() or "Unknown"


CPU_MODEL_NAME = get_cpu_model()
CPU_ARCH = platform.machine()
CPU_TDP_W = None

PYTHON_VERSION = platform.python_version()
TORCH_VERSION = torch.__version__

OS_NAME = platform.system()
OS_VERSION = platform.version()
OS_ARCHITECTURE = platform.machine()

SYSTEM_RAM_TOTAL_GB = round(
    psutil.virtual_memory().total / (1024 ** 3),
    2,
)

CPU_CORE_COUNT = psutil.cpu_count(logical=False)
CPU_THREAD_COUNT = psutil.cpu_count(logical=True)


def get_os_full_name():
    system = platform.system()

    if system == "Windows":
        return f"Windows {platform.release()} {platform.machine()}"

    if system == "Linux":
        try:
            info = {}
            with open("/etc/os-release", "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        info[key] = value.strip('"')
            return f"{info.get('PRETTY_NAME', 'Linux')} {platform.machine()}"
        except Exception:
            return f"Linux {platform.release()} {platform.machine()}"

    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]} {platform.machine()}"

    return f"{system} {platform.release()} {platform.machine()}"


OS_FULL_NAME = get_os_full_name()


def make_stable_device_id():
    raw = (
        f"{socket.gethostname()}-"
        f"{platform.system()}-"
        f"{platform.machine()}-"
        f"{CPU_MODEL_NAME}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


DEVICE_UUID = make_stable_device_id()
DEVICE_SHORT = DEVICE_UUID[:8]

DEVICE_LOG_DIR = OUTPUT_ROOT / DEVICE_SHORT
DEVICE_LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_hostname():
    return socket.gethostname()


def get_cuda_driver_version():
    if not NVML_AVAILABLE:
        return None
    try:
        version = pynvml.nvmlSystemGetDriverVersion()
        return version.decode("utf-8") if isinstance(version, bytes) else version
    except Exception:
        return None


CUDA_DRIVER_VERSION = get_cuda_driver_version()


def get_gpu_static():
    result = {
        "gpu_driver_version": CUDA_DRIVER_VERSION,
        "gpu_compute_capability": None,
        "gpu_power_limit_w": None,
        "gpu_memory_total_mb": None,
    }

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            result["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        except Exception:
            pass

    if not NVML_AVAILABLE or NVML_HANDLE is None:
        return result

    try:
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(NVML_HANDLE)
        memory = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)

        result["gpu_power_limit_w"] = round(power_limit / 1000.0, 2)
        result["gpu_memory_total_mb"] = round(
            memory.total / (1024 ** 2),
            2,
        )
    except Exception:
        pass

    return result


GPU_STATIC = get_gpu_static()


def get_gpu_core_thread():
    if not torch.cuda.is_available():
        return None, None

    try:
        props = torch.cuda.get_device_properties(0)
        sm_count = props.multi_processor_count

        cores_per_sm = {
            5: 128,
            6: 64,
            7: 64,
            8: 128,
            9: 128,
        }.get(props.major, 64)

        return (
            sm_count * cores_per_sm,
            sm_count * props.max_threads_per_multi_processor,
        )
    except Exception:
        return None, None


GPU_CORE_COUNT, GPU_THREAD_COUNT = get_gpu_core_thread()


def get_gpu_name():
    if not torch.cuda.is_available():
        return "No GPU"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "Unknown GPU"


def get_cpu_usage():
    try:
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None


def get_cpu_freq():
    try:
        freq = psutil.cpu_freq()
        return round(freq.current, 2) if freq else None
    except Exception:
        return None


def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None

        for name in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
            if name in temps:
                values = [
                    x.current for x in temps[name]
                    if x.current is not None
                ]
                if values:
                    return round(sum(values) / len(values), 2)
    except Exception:
        pass

    return None


def get_cpu_power_draw_w():
    # Requires platform-specific package power sensors.
    return None


def get_cpu_cores_used():
    try:
        return sum(
            1
            for value in psutil.cpu_percent(percpu=True)
            if value > 1.0
        )
    except Exception:
        return None


def get_memory_footprint_mb():
    try:
        return round(
            psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2),
            4,
        )
    except Exception:
        return None


def get_gpu_metrics():
    result = {
        "gpu_power_draw_w": None,
        "gpu_utilization_pct": None,
        "gpu_temp_c": None,
        "gpu_memory_used_mb": None,
        "gpu_sm_clock_mhz": None,
        "gpu_memory_clock_mhz": None,
    }

    if not NVML_AVAILABLE or NVML_HANDLE is None:
        return result

    try:
        power = pynvml.nvmlDeviceGetPowerUsage(NVML_HANDLE)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(NVML_HANDLE)
        temperature = pynvml.nvmlDeviceGetTemperature(
            NVML_HANDLE,
            pynvml.NVML_TEMPERATURE_GPU,
        )
        memory = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)
        sm_clock = pynvml.nvmlDeviceGetClockInfo(
            NVML_HANDLE,
            pynvml.NVML_CLOCK_SM,
        )
        memory_clock = pynvml.nvmlDeviceGetClockInfo(
            NVML_HANDLE,
            pynvml.NVML_CLOCK_MEM,
        )

        return {
            "gpu_power_draw_w": round(power / 1000.0, 2),
            "gpu_utilization_pct": utilization.gpu,
            "gpu_temp_c": temperature,
            "gpu_memory_used_mb": round(memory.used / (1024 ** 2), 2),
            "gpu_sm_clock_mhz": sm_clock,
            "gpu_memory_clock_mhz": memory_clock,
        }
    except Exception:
        return result


# ============================================================
# 4. SHARED HELPERS
# ============================================================

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def normalize_prediction(response):
    text = str(response).lower().strip()

    digit_matches = list(
        dict.fromkeys(
            re.findall(r"(?<!\d)[0-9](?!\d)", text)
        )
    )

    word_matches = []
    for word, digit in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            word_matches.append(digit)

    combined = list(
        dict.fromkeys(digit_matches + word_matches)
    )

    return combined[0] if len(combined) == 1 else "invalid"


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def run_with_energy_tracking(
    inference_fn,
    image,
    model_key,
    dataset_key,
):
    energy_dir = OUTPUT_ROOT / "codecarbon"
    energy_dir.mkdir(parents=True, exist_ok=True)

    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name=f"{model_key}_{dataset_key}_test",
            output_dir=str(energy_dir),
            output_file=f"codecarbon_{model_key}_{dataset_key}.csv",
            log_level="error",
            save_to_file=True,
        )

        tracker.start()
        synchronize()
        start = time.perf_counter()

        result = inference_fn(image)

        synchronize()
        exec_time = time.perf_counter() - start

        emissions_value = tracker.stop()
        final_data = getattr(
            tracker,
            "final_emissions_data",
            None,
        )

        cpu_energy = float(
            getattr(final_data, "cpu_energy", 0) or 0
        )
        gpu_energy = float(
            getattr(final_data, "gpu_energy", 0) or 0
        )
        ram_energy = float(
            getattr(final_data, "ram_energy", 0) or 0
        )
        total_energy = float(
            getattr(final_data, "energy_consumed", 0) or 0
        )

        carbon_intensity = (
            float(emissions_value) / total_energy
            if emissions_value is not None and total_energy > 0
            else None
        )

    else:
        synchronize()
        start = time.perf_counter()

        result = inference_fn(image)

        synchronize()
        exec_time = time.perf_counter() - start

        cpu_energy = 0.0
        gpu_energy = 0.0
        ram_energy = 0.0
        total_energy = 0.0
        emissions_value = 0.0
        carbon_intensity = None

    return (
        result,
        exec_time,
        cpu_energy,
        gpu_energy,
        ram_energy,
        total_energy,
        emissions_value,
        carbon_intensity,
        get_gpu_metrics(),
    )


def build_telemetry_row(
    *,
    model_name,
    model_id,
    parameters,
    model_flops,
    dataset_name,
    dataset_key,
    query,
    sample_index,
    true_label,
    result,
    exec_time,
    cpu_energy,
    gpu_energy,
    ram_energy,
    total_energy,
    emissions_value,
    carbon_intensity,
    gpu_metrics,
    error_message="",
):
    """
    IMPORTANT:
    This row schema is intentionally identical to the uploaded
    CNN/DNN telemetry script.

    Do not add VLM-specific columns here, because the goal is
    column-for-column compatibility across CNN, DNN, Qwen2-VL,
    and Moondream2 CSV files.
    """

    prediction = result.get("prediction", "invalid")

    # Keep prediction numeric when valid. Invalid VLM outputs remain -1 so
    # the CSV remains compatible with integer-based classical telemetry.
    try:
        prediction_for_csv = int(prediction)
    except (TypeError, ValueError):
        prediction_for_csv = -1

    correct = (
        int(prediction_for_csv) == int(true_label)
        if true_label is not None
        else None
    )

    input_tokens = int(
        result.get("input_tokens", 0) or 0
    )

    output_tokens = int(
        result.get("output_tokens", 0) or 0
    )

    total_tokens = input_tokens + output_tokens

    total_energy = total_energy or 0.0
    cpu_energy = cpu_energy or 0.0
    gpu_energy = gpu_energy or 0.0
    ram_energy = ram_energy or 0.0

    joules_per_token = 0.0
    energy_per_token_kwh = 0.0
    watts_estimated = 0.0
    gpu_energy_pct = 0.0
    cpu_energy_pct = 0.0

    if total_energy > 0 and total_tokens > 0:
        energy_per_token_kwh = round(
            total_energy / total_tokens,
            12,
        )

        joules_per_token = round(
            (total_energy * 3_600_000)
            / total_tokens,
            6,
        )

        if exec_time > 0:
            watts_estimated = round(
                (total_energy * 3_600_000)
                / exec_time,
                4,
            )

        gpu_energy_pct = round(
            (gpu_energy / total_energy) * 100,
            2,
        )

        cpu_energy_pct = round(
            (cpu_energy / total_energy) * 100,
            2,
        )

    gpu_metrics = gpu_metrics or {}

    return {
        # --- Identity ---
        "timestamp":                    time.strftime("%Y-%m-%d %H:%M:%S"),
        "unique_device_id":             DEVICE_UUID,
        "device_short_id":              DEVICE_SHORT,
        "pc_name":                      get_hostname(),
        "collection_mode":              "automated_edge",

        # --- Sample ---
        "sample_index":                 sample_index,
        "sample_id":                    sample_index,
        "true_label":                   int(true_label) if true_label is not None else None,
        "prediction":                   prediction_for_csv,
        "correct":                      correct,

        # --- Model identity ---
        "dataset":                      dataset_name,
        "model_type":                   model_name,
        "parameters":                   parameters,
        "model_flops":                  model_flops,
        "checkpoint_path":              model_id,

        # --- Prediction quality ---
        "confidence_score":             (
            round(float(result["confidence_score"]), 6)
            if result.get("confidence_score") is not None
            else None
        ),
        "logit_margin":                 (
            round(float(result["logit_margin"]), 6)
            if result.get("logit_margin") is not None
            else None
        ),
        "entropy":                      (
            round(float(result["entropy"]), 6)
            if result.get("entropy") is not None
            else None
        ),

        # --- Timing ---
        "execution_time_sec":           round(exec_time, 10),

        # --- CodeCarbon energy ---
        "cpu_energy_kwh":               cpu_energy,
        "gpu_energy_kwh":               gpu_energy,
        "ram_energy_kwh":               ram_energy,
        "total_energy_kwh":             total_energy,
        "total_emissions_kg":           emissions_value,
        "carbon_intensity_kgco2_kwh":   carbon_intensity,
        "codecarbon_version":           CODECARBON_VERSION,

        # --- Efficiency derived ---
        "input_tokens":                 input_tokens,
        "output_tokens":                output_tokens,
        "total_tokens":                 total_tokens,
        "tokens_per_second":            (
            round(total_tokens / exec_time, 4)
            if exec_time > 0
            else None
        ),
        "joules_per_token":             joules_per_token,
        "energy_per_token_kwh":         energy_per_token_kwh,
        "watts_estimated":              watts_estimated,
        "gpu_energy_pct_of_total":      gpu_energy_pct,
        "cpu_energy_pct_of_total":      cpu_energy_pct,

        # --- CPU hardware ---
        "cpu_model":                    CPU_MODEL_NAME,
        "cpu_architecture":             CPU_ARCH,
        "cpu_core_count":               CPU_CORE_COUNT,
        "cpu_thread_count":             CPU_THREAD_COUNT,
        "cpu_core":                     CPU_CORE_COUNT,
        "cpu_thread":                   CPU_THREAD_COUNT,
        "cpu_tdp_w":                    CPU_TDP_W,
        "cpu_usage_pct":                get_cpu_usage(),
        "cpu_clock_mhz":                get_cpu_freq(),
        "cpu_temp_c":                   get_cpu_temp(),
        "cpu_power_draw_w":             get_cpu_power_draw_w(),
        "cpu_cores_used":               get_cpu_cores_used(),

        # --- GPU hardware ---
        "gpu_model":                    get_gpu_name(),
        "gpu_core":                     GPU_CORE_COUNT,
        "gpu_thread":                   GPU_THREAD_COUNT,
        "gpu_driver_version":           GPU_STATIC["gpu_driver_version"],
        "gpu_compute_capability":       GPU_STATIC["gpu_compute_capability"],
        "gpu_power_limit_w":            GPU_STATIC["gpu_power_limit_w"],
        "gpu_memory_total_mb":          GPU_STATIC["gpu_memory_total_mb"],
        "gpu_power_draw_w":             gpu_metrics.get("gpu_power_draw_w"),
        "gpu_utilization_pct":          gpu_metrics.get("gpu_utilization_pct"),
        "gpu_temp_c":                   gpu_metrics.get("gpu_temp_c"),
        "gpu_memory_used_mb":           gpu_metrics.get("gpu_memory_used_mb"),
        "gpu_sm_clock_mhz":             gpu_metrics.get("gpu_sm_clock_mhz"),
        "gpu_memory_clock_mhz":         gpu_metrics.get("gpu_memory_clock_mhz"),
        "cuda_driver_version":          CUDA_DRIVER_VERSION,
        "cuda_available":               torch.cuda.is_available(),
        "device_type":                  str(DEVICE),

        # --- RAM / memory ---
        "ram_usage_pct":                psutil.virtual_memory().percent,
        "memory_footprint_mb":          get_memory_footprint_mb(),
        "system_ram_total_gb":          SYSTEM_RAM_TOTAL_GB,

        # --- Environment ---
        "os_name":                      OS_NAME,
        "os_version":                   OS_VERSION,
        "os_architecture":              OS_ARCHITECTURE,
        "os_full_name":                 OS_FULL_NAME,
        "python_version":               PYTHON_VERSION,
        "torch_version":                TORCH_VERSION,

        # --- Final model metrics (backfilled after run) ---
        "model_accuracy":               None,
        "model_precision_weighted":     None,
        "model_recall_weighted":        None,
        "model_f1_weighted":            None,

        # --- Custom / attack context ---
        "quantum_computing":            False,
        "model_under_attack":           0,

        # --- Quantum-only fields kept for exact schema parity ---
        "source_qubits":                None,
        "n_qubits":                     None,
        "fidelity":                     None,
        "pennylane_device":             None,
        "pennylane_version":            None,
    }


def get_test_dataset(dataset_class):
    return dataset_class(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=None,
    )


def get_test_limit(dataset):
    if NUM_TEST_SAMPLES is None:
        return len(dataset)

    if NUM_TEST_SAMPLES <= 0:
        raise ValueError(
            "NUM_TEST_SAMPLES must be positive, "
            "or use -1 for the complete test set."
        )

    return min(NUM_TEST_SAMPLES, len(dataset))


def finalize_and_save(
    rows,
    model_name,
    dataset_name,
    dataset_key,
):
    df = pd.DataFrame(rows)

    y_true = df["true_label"].astype(int).to_numpy()

    # Invalid predictions are retained as -1 for metric computation.
    y_pred = pd.to_numeric(
        df["prediction"],
        errors="coerce",
    ).fillna(-1).astype(int).to_numpy()

    accuracy = float(
        accuracy_score(y_true, y_pred)
    )

    precision = float(
        precision_score(
            y_true,
            y_pred,
            labels=list(range(10)),
            average="weighted",
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_true,
            y_pred,
            labels=list(range(10)),
            average="weighted",
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(10)),
            average="weighted",
            zero_division=0,
        )
    )

    df["model_accuracy"] = accuracy
    df["model_precision_weighted"] = precision
    df["model_recall_weighted"] = recall
    df["model_f1_weighted"] = f1

    # Force EXACT column names and order from the uploaded reference test code.
    REFERENCE_COLUMNS = ['timestamp', 'unique_device_id', 'device_short_id', 'pc_name', 'collection_mode', 'sample_index', 'sample_id', 'true_label', 'prediction', 'correct', 'dataset', 'model_type', 'parameters', 'model_flops', 'checkpoint_path', 'confidence_score', 'logit_margin', 'entropy', 'execution_time_sec', 'cpu_energy_kwh', 'gpu_energy_kwh', 'ram_energy_kwh', 'total_energy_kwh', 'total_emissions_kg', 'carbon_intensity_kgco2_kwh', 'codecarbon_version', 'input_tokens', 'output_tokens', 'total_tokens', 'tokens_per_second', 'joules_per_token', 'energy_per_token_kwh', 'watts_estimated', 'gpu_energy_pct_of_total', 'cpu_energy_pct_of_total', 'cpu_model', 'cpu_architecture', 'cpu_core_count', 'cpu_thread_count', 'cpu_core', 'cpu_thread', 'cpu_tdp_w', 'cpu_usage_pct', 'cpu_clock_mhz', 'cpu_temp_c', 'cpu_power_draw_w', 'cpu_cores_used', 'gpu_model', 'gpu_core', 'gpu_thread', 'gpu_driver_version', 'gpu_compute_capability', 'gpu_power_limit_w', 'gpu_memory_total_mb', 'gpu_power_draw_w', 'gpu_utilization_pct', 'gpu_temp_c', 'gpu_memory_used_mb', 'gpu_sm_clock_mhz', 'gpu_memory_clock_mhz', 'cuda_driver_version', 'cuda_available', 'device_type', 'ram_usage_pct', 'memory_footprint_mb', 'system_ram_total_gb', 'os_name', 'os_version', 'os_architecture', 'os_full_name', 'python_version', 'torch_version', 'model_accuracy', 'model_precision_weighted', 'model_recall_weighted', 'model_f1_weighted', 'quantum_computing', 'model_under_attack', 'source_qubits', 'n_qubits', 'fidelity', 'pennylane_device', 'pennylane_version']
    df = df.reindex(columns=REFERENCE_COLUMNS)

    safe_model = (
        model_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )

    output_csv = (
        DEVICE_LOG_DIR
        / f"{safe_model}_{dataset_key}_test_telemetry.csv"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    print("\n" + "=" * 72)
    print(f"{model_name} | {dataset_name}")
    print("=" * 72)
    print(f"Samples   : {len(df)}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"Saved     : {output_csv.resolve()}")

    return {
        "model": model_name,
        "dataset": dataset_name,
        "samples": len(df),
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "telemetry_csv": str(output_csv.resolve()),
    }


# ============================================================
# 5. QWEN2-VL
# ============================================================

def load_qwen():
    print("\n" + "=" * 72)
    print("LOADING QWEN2-VL")
    print("=" * 72)

    processor = AutoProcessor.from_pretrained(
        QWEN_MODEL_ID,
        min_pixels=56 * 56,
        max_pixels=56 * 56,
    )

    if DEVICE.type == "cuda":
        model = (
            Qwen2VLForConditionalGeneration
            .from_pretrained(
                QWEN_MODEL_ID,
                torch_dtype=DTYPE,
                device_map="auto",
            )
        )
    else:
        model = (
            Qwen2VLForConditionalGeneration
            .from_pretrained(
                QWEN_MODEL_ID,
                torch_dtype=torch.float32,
            )
            .to(DEVICE)
        )

    model.eval()

    digit_token_ids = []
    for digit in range(10):
        ids = processor.tokenizer.encode(
            str(digit),
            add_special_tokens=False,
        )

        if len(ids) != 1:
            raise RuntimeError(
                f"Qwen tokenizer represents digit {digit} "
                f"with {len(ids)} tokens: {ids}"
            )

        digit_token_ids.append(ids[0])

    parameters = int(
        sum(p.numel() for p in model.parameters())
    )

    return model, processor, digit_token_ids, parameters


def make_qwen_inference(
    model,
    processor,
    digit_token_ids,
    query,
):
    def run_qwen_inference(image):
        image = image.convert("RGB")

        # IMPORTANT:
        # "query" is the same string passed to Moondream.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        model_device = next(model.parameters()).device

        inputs = {
            key: (
                value.to(model_device)
                if torch.is_tensor(value)
                else value
            )
            for key, value in inputs.items()
        }

        if "attention_mask" in inputs:
            input_tokens = int(
                inputs["attention_mask"].sum().item()
            )
        else:
            input_tokens = int(
                inputs["input_ids"].numel()
            )

        with torch.inference_mode():
            outputs = model(
                **inputs,
                return_dict=True,
            )

        next_token_logits = outputs.logits[0, -1, :]

        candidate_ids = torch.tensor(
            digit_token_ids,
            device=next_token_logits.device,
        )

        digit_logits = next_token_logits[candidate_ids]
        digit_probs = torch.softmax(
            digit_logits.float(),
            dim=0,
        )

        predicted_index = int(
            torch.argmax(digit_probs).item()
        )

        top2_logits = torch.topk(
            digit_logits.float(),
            k=2,
        ).values

        entropy = float(
            -(
                digit_probs
                * torch.log(digit_probs + 1e-12)
            ).sum().item()
        )

        return {
            "prediction": predicted_index,
            "raw_response": str(predicted_index),
            "confidence_score": float(
                digit_probs[predicted_index].item()
            ),
            "logit_margin": float(
                (top2_logits[0] - top2_logits[1]).item()
            ),
            "entropy": entropy,
            "input_tokens": input_tokens,
            "output_tokens": 1,
        }

    return run_qwen_inference


def run_qwen_all_datasets():
    model_name = "Qwen2-VL-2B-Instruct"

    model, processor, digit_ids, parameters = load_qwen()

    summaries = []

    try:
        for dataset_name, cfg in DATASET_CONFIGS.items():
            dataset = get_test_dataset(
                cfg["dataset_class"]
            )
            limit = get_test_limit(dataset)
            query = cfg["query"]

            inference_fn = make_qwen_inference(
                model,
                processor,
                digit_ids,
                query,
            )

            rows = []

            print(
                f"\nTesting {model_name} on {dataset_name} "
                f"({limit} samples)"
            )
            print("Shared query:")
            print(query)

            for i in tqdm(
                range(limit),
                desc=f"Qwen | {dataset_name}",
                unit="sample",
            ):
                image, true_label = dataset[i]

                error_message = ""

                try:
                    (
                        result,
                        exec_time,
                        cpu_energy,
                        gpu_energy,
                        ram_energy,
                        total_energy,
                        emissions_value,
                        carbon_intensity,
                        gpu_metrics,
                    ) = run_with_energy_tracking(
                        inference_fn,
                        image,
                        model_key="qwen2vl",
                        dataset_key=cfg["key"],
                    )

                except Exception as error:
                    error_message = repr(error)
                    result = {
                        "prediction": "invalid",
                        "raw_response": "",
                        "confidence_score": None,
                        "logit_margin": None,
                        "entropy": None,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                    exec_time = 0.0
                    cpu_energy = 0.0
                    gpu_energy = 0.0
                    ram_energy = 0.0
                    total_energy = 0.0
                    emissions_value = 0.0
                    carbon_intensity = None
                    gpu_metrics = get_gpu_metrics()

                rows.append(
                    build_telemetry_row(
                        model_name=model_name,
                        model_id=QWEN_MODEL_ID,
                        parameters=parameters,
                        model_flops=None,
                        dataset_name=dataset_name,
                        dataset_key=cfg["key"],
                        query=query,
                        sample_index=i,
                        true_label=int(true_label),
                        result=result,
                        exec_time=exec_time,
                        cpu_energy=cpu_energy,
                        gpu_energy=gpu_energy,
                        ram_energy=ram_energy,
                        total_energy=total_energy,
                        emissions_value=emissions_value,
                        carbon_intensity=carbon_intensity,
                        gpu_metrics=gpu_metrics,
                        error_message=error_message,
                    )
                )

            summaries.append(
                finalize_and_save(
                    rows,
                    model_name,
                    dataset_name,
                    cfg["key"],
                )
            )

    finally:
        del model
        del processor

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summaries


# ============================================================
# 6. MOONDREAM2 COMPATIBILITY PATCHES
# ============================================================

def patch_dynamic_cache():
    if not hasattr(
        DynamicCache,
        "get_usable_length",
    ):
        def _get_usable_length(
            self,
            new_seq_length: int,
            layer_idx: int = 0,
        ):
            try:
                return int(
                    self.get_seq_length(layer_idx)
                )
            except TypeError:
                return int(
                    self.get_seq_length()
                )

        DynamicCache.get_usable_length = (
            _get_usable_length
        )

    if not hasattr(
        DynamicCache,
        "get_max_length",
    ):
        def _get_max_length(self):
            return None

        DynamicCache.get_max_length = (
            _get_max_length
        )


def patch_generation_support(moondream_model):
    text_model = moondream_model.text_model

    if not callable(
        getattr(text_model, "generate", None)
    ):
        original_class = text_model.__class__

        patched_class = type(
            f"{original_class.__name__}WithGeneration",
            (GenerationMixin, original_class),
            {
                "__module__":
                    original_class.__module__,
            },
        )

        text_model.__class__ = patched_class

    if getattr(
        text_model,
        "generation_config",
        None,
    ) is None:
        text_model.generation_config = (
            GenerationConfig.from_model_config(
                text_model.config
            )
        )

    if not hasattr(
        text_model,
        "_supports_cache_class",
    ):
        text_model._supports_cache_class = False


def patch_phi_forward(moondream_model):
    original_forward = moondream_model.text_model.forward

    supported = set(
        inspect.signature(
            original_forward
        ).parameters.keys()
    )

    def compat_forward(
        self,
        *args,
        **kwargs,
    ):
        for key in (
            "cache_position",
            "num_logits_to_keep",
        ):
            if key in kwargs and key not in supported:
                kwargs.pop(key, None)

        return original_forward(
            *args,
            **kwargs,
        )

    moondream_model.text_model.forward = (
        types.MethodType(
            compat_forward,
            moondream_model.text_model,
        )
    )


def load_moondream():
    print("\n" + "=" * 72)
    print("LOADING MOONDREAM2")
    print("=" * 72)

    patch_dynamic_cache()

    tokenizer = AutoTokenizer.from_pretrained(
        MOONDREAM_MODEL_ID,
        revision=MOONDREAM_REVISION,
        trust_remote_code=True,
    )

    config = AutoConfig.from_pretrained(
        MOONDREAM_MODEL_ID,
        revision=MOONDREAM_REVISION,
        trust_remote_code=True,
    )

    if getattr(
        config,
        "auto_map",
        None,
    ) is None:
        config.auto_map = {}

    # Force the matching historical Moondream class from the
    # pinned DoCCI repository revision.
    config.auto_map[
        "AutoModelForCausalLM"
    ] = "moondream.Moondream"

    model = AutoModelForCausalLM.from_pretrained(
        MOONDREAM_MODEL_ID,
        revision=MOONDREAM_REVISION,
        config=config,
        trust_remote_code=True,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    )

    patch_generation_support(model)
    patch_phi_forward(model)

    model.to(DEVICE)
    model.eval()

    parameters = int(
        sum(p.numel() for p in model.parameters())
    )

    return model, tokenizer, parameters


def get_moondream_embedding_layer(model):
    text_model = model.text_model

    if hasattr(
        text_model,
        "get_input_embeddings",
    ):
        layer = text_model.get_input_embeddings()
        if layer is not None:
            return layer

    try:
        return (
            text_model
            .transformer
            .embd
            .wte
        )
    except Exception:
        pass

    raise RuntimeError(
        "Could not locate Moondream token embedding layer."
    )


def get_moondream_start_token_id(tokenizer):
    if tokenizer.bos_token_id is not None:
        return int(tokenizer.bos_token_id)

    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)

    raise RuntimeError(
        "Moondream tokenizer has no BOS/EOS token."
    )


def make_moondream_inference(
    model,
    tokenizer,
    query,
):
    token_embedding = (
        get_moondream_embedding_layer(model)
    )

    start_token_id = (
        get_moondream_start_token_id(tokenizer)
    )

    def tokenize_text(text):
        return tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(DEVICE)

    def score_candidate_classes(image):
        image = image.convert("RGB")

        with torch.inference_mode():
            image_embeddings = model.encode_image(
                image
            )

        # IMPORTANT:
        # Exact same query string used by Qwen for this dataset.
        prompt_text = (
            f"\n\nQuestion: {query}"
            f"\n\nAnswer:"
        )

        start_ids = torch.tensor(
            [[start_token_id]],
            dtype=torch.long,
            device=DEVICE,
        )

        prompt_ids = tokenize_text(
            prompt_text
        )

        start_embeddings = token_embedding(
            start_ids
        )

        prompt_embeddings = token_embedding(
            prompt_ids
        )

        image_embeddings = image_embeddings.to(
            device=DEVICE,
            dtype=start_embeddings.dtype,
        )

        prefix_embeddings = torch.cat(
            [
                start_embeddings,
                image_embeddings,
                prompt_embeddings,
            ],
            dim=1,
        )

        prefix_length = (
            prefix_embeddings.shape[1]
        )

        class_scores = []

        for class_name in CLASS_NAMES:
            class_ids = tokenize_text(
                class_name
            )

            class_embeddings = token_embedding(
                class_ids
            ).to(
                dtype=prefix_embeddings.dtype
            )

            full_embeddings = torch.cat(
                [
                    prefix_embeddings,
                    class_embeddings,
                ],
                dim=1,
            )

            attention_mask = torch.ones(
                full_embeddings.shape[:2],
                dtype=torch.long,
                device=DEVICE,
            )

            with torch.inference_mode():
                outputs = model.text_model(
                    inputs_embeds=full_embeddings,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )

            logits = outputs.logits

            token_log_probs = []

            for token_offset, token_id in enumerate(
                class_ids[0]
            ):
                position = (
                    prefix_length
                    - 1
                    + token_offset
                )

                log_probs = torch.log_softmax(
                    logits[0, position, :].float(),
                    dim=-1,
                )

                token_log_probs.append(
                    log_probs[
                        int(token_id.item())
                    ]
                )

            class_scores.append(
                torch.stack(
                    token_log_probs
                ).mean()
            )

        scores = torch.stack(
            class_scores
        ).float()

        probabilities = torch.softmax(
            scores,
            dim=0,
        )

        top2_scores = torch.topk(
            scores,
            k=2,
        ).values

        entropy = float(
            -(
                probabilities
                * torch.log(
                    probabilities + 1e-12
                )
            ).sum().item()
        )

        best_index = int(
            torch.argmax(
                probabilities
            ).item()
        )

        input_tokens = int(
            start_ids.shape[1]
            + image_embeddings.shape[1]
            + prompt_ids.shape[1]
        )

        return {
            "class_probabilities":
                probabilities.detach().cpu(),
            "top_class_index":
                best_index,
            "confidence_score":
                float(
                    probabilities[
                        best_index
                    ].item()
                ),
            "logit_margin":
                float(
                    (
                        top2_scores[0]
                        - top2_scores[1]
                    ).item()
                ),
            "entropy":
                entropy,
            "input_tokens":
                input_tokens,
        }

    def run_moondream_inference(image):
        image = image.convert("RGB")

        with torch.inference_mode():
            encoded_image = model.encode_image(
                image
            )

            # IMPORTANT:
            # Same query string that Qwen receives.
            answer = model.answer_question(
                encoded_image,
                query,
                tokenizer,
            )

        raw_response = str(answer).strip()
        prediction = normalize_prediction(
            raw_response
        )

        quality = score_candidate_classes(
            image
        )

        if prediction in CLASS_NAMES:
            prediction_index = int(prediction)

            confidence_score = float(
                quality[
                    "class_probabilities"
                ][prediction_index].item()
            )
        else:
            confidence_score = 0.0

        output_tokens = len(
            tokenizer(
                raw_response,
                add_special_tokens=False,
            ).input_ids
        )

        return {
            "prediction": (
                int(prediction)
                if prediction in CLASS_NAMES
                else "invalid"
            ),
            "raw_response": raw_response,
            "confidence_score": confidence_score,
            "logit_margin": quality[
                "logit_margin"
            ],
            "entropy": quality[
                "entropy"
            ],
            "input_tokens": quality[
                "input_tokens"
            ],
            "output_tokens": max(
                int(output_tokens),
                1,
            ),
        }

    return run_moondream_inference


def run_moondream_all_datasets():
    model_name = "Moondream2-DoCCI-Instruct"

    model, tokenizer, parameters = (
        load_moondream()
    )

    summaries = []

    try:
        for dataset_name, cfg in DATASET_CONFIGS.items():
            dataset = get_test_dataset(
                cfg["dataset_class"]
            )
            limit = get_test_limit(dataset)
            query = cfg["query"]

            inference_fn = (
                make_moondream_inference(
                    model,
                    tokenizer,
                    query,
                )
            )

            rows = []

            print(
                f"\nTesting {model_name} on {dataset_name} "
                f"({limit} samples)"
            )
            print("Shared query:")
            print(query)

            for i in tqdm(
                range(limit),
                desc=f"Moondream2 | {dataset_name}",
                unit="sample",
            ):
                image, true_label = dataset[i]

                error_message = ""

                try:
                    (
                        result,
                        exec_time,
                        cpu_energy,
                        gpu_energy,
                        ram_energy,
                        total_energy,
                        emissions_value,
                        carbon_intensity,
                        gpu_metrics,
                    ) = run_with_energy_tracking(
                        inference_fn,
                        image,
                        model_key="moondream2",
                        dataset_key=cfg["key"],
                    )

                except Exception as error:
                    error_message = repr(error)

                    result = {
                        "prediction": "invalid",
                        "raw_response": "",
                        "confidence_score": None,
                        "logit_margin": None,
                        "entropy": None,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                    exec_time = 0.0
                    cpu_energy = 0.0
                    gpu_energy = 0.0
                    ram_energy = 0.0
                    total_energy = 0.0
                    emissions_value = 0.0
                    carbon_intensity = None
                    gpu_metrics = get_gpu_metrics()

                rows.append(
                    build_telemetry_row(
                        model_name=model_name,
                        model_id=MOONDREAM_MODEL_ID,
                        parameters=parameters,
                        model_flops=None,
                        dataset_name=dataset_name,
                        dataset_key=cfg["key"],
                        query=query,
                        sample_index=i,
                        true_label=int(true_label),
                        result=result,
                        exec_time=exec_time,
                        cpu_energy=cpu_energy,
                        gpu_energy=gpu_energy,
                        ram_energy=ram_energy,
                        total_energy=total_energy,
                        emissions_value=emissions_value,
                        carbon_intensity=carbon_intensity,
                        gpu_metrics=gpu_metrics,
                        error_message=error_message,
                    )
                )

            summaries.append(
                finalize_and_save(
                    rows,
                    model_name,
                    dataset_name,
                    cfg["key"],
                )
            )

    finally:
        del model
        del tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summaries


# ============================================================
# 7. MAIN
# ============================================================

def qwen_moondream_main():
    print("=" * 72)
    print("QWEN2-VL + MOONDREAM2 THREE-DATASET TELEMETRY TEST")
    print("=" * 72)
    print("Device:", DEVICE)
    print("dtype:", DTYPE)
    print("Samples per dataset:", (
        "ALL"
        if NUM_TEST_SAMPLES is None
        else NUM_TEST_SAMPLES
    ))
    print("Output directory:", DEVICE_LOG_DIR.resolve())

    all_summaries = []

    # --------------------------------------------------------
    # Qwen first
    # --------------------------------------------------------
    qwen_summaries = run_qwen_all_datasets()
    all_summaries.extend(qwen_summaries)

    # Model is explicitly deleted before Moondream is loaded.

    # --------------------------------------------------------
    # Moondream second
    # --------------------------------------------------------
    moondream_summaries = (
        run_moondream_all_datasets()
    )
    all_summaries.extend(
        moondream_summaries
    )

    # --------------------------------------------------------
    # Combined summary
    # --------------------------------------------------------
    summary_csv = (
        DEVICE_LOG_DIR
        / "qwen2vl_moondream2_three_dataset_summary.csv"
    )

    pd.DataFrame(
        all_summaries
    ).to_csv(
        summary_csv,
        index=False,
    )

    print("\n" + "=" * 72)
    print("ALL TESTS COMPLETE")
    print("=" * 72)

    for item in all_summaries:
        print(
            f"{item['model']} | "
            f"{item['dataset']} | "
            f"accuracy={item['accuracy']:.4f} | "
            f"F1={item['f1_weighted']:.4f}"
        )

    print("\nCombined summary:")
    print(summary_csv.resolve())


# -*- coding: utf-8 -*-
"""
YOLO classification + MobileNetV2 telemetry test
Datasets: MNIST, FashionMNIST, KMNIST

This script:
1. Loads the PREVIOUSLY SAVED trained models.
2. Tests MNIST, FashionMNIST, and KMNIST.
3. Uses the exact same 83 telemetry columns as the reference CNN/DNN test.
4. Saves one CSV per model/dataset.
5. Backfills Accuracy / Precision / Recall / F1 into every row.

Expected saved models:
    yolo26n_mnist_cpu.pt
    yolo26n_fashionmnist_cpu.pt
    yolo26n_kmnist_cpu.pt

    mobilenet_v2_mnist_cpu.pt
    mobilenet_v2_fashionmnist_cpu.pt
    mobilenet_v2_kmnist_cpu.pt
"""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import sys
import time
from pathlib import Path

import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from torchvision import datasets, transforms
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights,
)
from tqdm.auto import tqdm
from ultralytics import YOLO


# ============================================================
# 1. CONFIGURATION
# ============================================================

NUM_TEST_SAMPLES = int(
    os.getenv("NUM_TEST_SAMPLES", "10000")
)

DEVICE_MODE = os.getenv(
    "DEVICE_MODE",
    "cpu",
).lower()

if DEVICE_MODE == "cuda":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        print(
            "CUDA requested but unavailable. "
            "Falling back to CPU."
        )
        DEVICE = torch.device("cpu")

elif DEVICE_MODE == "auto":
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

else:
    DEVICE = torch.device("cpu")


DATA_ROOT = Path("./classical_data")

OUTPUT_ROOT = Path.cwd() / "test_results"
OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

DATASET_CONFIGS = {
    "MNIST": {
        "dataset_class":
            datasets.MNIST,

        "key":
            "mnist",

        "yolo_checkpoint":
            Path("yolo26n_mnist_cpu.pt"),

        "mobilenet_checkpoint":
            Path("mobilenet_v2_mnist_cpu.pt"),
    },

    "FashionMNIST": {
        "dataset_class":
            datasets.FashionMNIST,

        "key":
            "fashionmnist",

        "yolo_checkpoint":
            Path("yolo26n_fashionmnist_cpu.pt"),

        "mobilenet_checkpoint":
            Path("mobilenet_v2_fashionmnist_cpu.pt"),
    },

    "KMNIST": {
        "dataset_class":
            datasets.KMNIST,

        "key":
            "kmnist",

        "yolo_checkpoint":
            Path("yolo26n_kmnist_cpu.pt"),

        "mobilenet_checkpoint":
            Path("mobilenet_v2_kmnist_cpu.pt"),
    },
}


# ============================================================
# 3. OPTIONAL PACKAGES
# ============================================================

try:
    from codecarbon import EmissionsTracker
    import codecarbon

    CODECARBON_AVAILABLE = True
    CODECARBON_VERSION = (
        codecarbon.__version__
    )

except Exception:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False
    CODECARBON_VERSION = "unavailable"

    print(
        "CodeCarbon not available. "
        "Energy values will be 0."
    )


try:
    import pynvml

    pynvml.nvmlInit()

    NVML_AVAILABLE = True

    NVML_HANDLE = (
        pynvml
        .nvmlDeviceGetHandleByIndex(0)
        if torch.cuda.is_available()
        else None
    )

except Exception:
    NVML_AVAILABLE = False
    NVML_HANDLE = None

    print(
        "pynvml not available."
    )


try:
    import cpuinfo

    _CPU_INFO = (
        cpuinfo.get_cpu_info()
    )

    CPU_MODEL_NAME = (
        _CPU_INFO.get(
            "brand_raw",
            "Unknown",
        )
    )

    CPU_ARCH = (
        _CPU_INFO.get(
            "arch",
            platform.machine(),
        )
    )

except Exception:
    CPU_MODEL_NAME = (
        platform.processor()
        or "Unknown"
    )

    CPU_ARCH = (
        platform.machine()
    )


CPU_TDP_W = None


# ============================================================
# 4. SYSTEM INFORMATION
# ============================================================

TORCH_VERSION = torch.__version__
PYTHON_VERSION = (
    sys.version.split()[0]
)

OS_NAME = platform.system()
OS_VERSION = platform.version()

OS_ARCHITECTURE = (
    platform.machine()
)

SYSTEM_RAM_TOTAL_GB = round(
    psutil
    .virtual_memory()
    .total
    / (1024 ** 3),
    2,
)

CPU_CORE_COUNT = (
    psutil.cpu_count(
        logical=False
    )
)

CPU_THREAD_COUNT = (
    psutil.cpu_count(
        logical=True
    )
)


def get_os_full_name():

    system = (
        platform.system()
    )

    architecture = (
        platform.machine()
    )

    if system == "Windows":

        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                (
                    r"SOFTWARE\Microsoft"
                    r"\Windows NT"
                    r"\CurrentVersion"
                ),
            )

            product_name = (
                winreg.QueryValueEx(
                    key,
                    "ProductName",
                )[0]
            )

            display_version = (
                winreg.QueryValueEx(
                    key,
                    "DisplayVersion",
                )[0]
            )

            current_build = (
                winreg.QueryValueEx(
                    key,
                    "CurrentBuild",
                )[0]
            )

            return (
                f"{product_name} "
                f"{display_version} "
                f"Build {current_build} "
                f"{architecture}"
            )

        except Exception:

            return (
                f"Windows "
                f"{platform.release()} "
                f"{architecture}"
            )

    if system == "Linux":

        info = {}

        try:

            with open(
                "/etc/os-release",
                "r",
                encoding="utf-8",
            ) as f:

                for line in f:

                    if "=" in line:

                        key, value = (
                            line
                            .strip()
                            .split("=", 1)
                        )

                        info[key] = (
                            value.strip('"')
                        )

        except Exception:
            pass

        pretty_name = (
            info.get(
                "PRETTY_NAME"
            )
        )

        if pretty_name:

            return (
                f"{pretty_name} "
                f"{architecture}"
            )

        return (
            f"Linux "
            f"{platform.release()} "
            f"{architecture}"
        )

    if system == "Darwin":

        return (
            f"macOS "
            f"{platform.mac_ver()[0]} "
            f"{architecture}"
        )

    return (
        f"{system} "
        f"{platform.release()} "
        f"{architecture}"
    )


OS_FULL_NAME = (
    get_os_full_name()
)


# ============================================================
# 5. DEVICE ID
# ============================================================

def make_stable_device_id():

    raw = (
        f"{socket.gethostname()}-"
        f"{platform.system()}-"
        f"{platform.machine()}-"
        f"{CPU_MODEL_NAME}"
    )

    return hashlib.sha256(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()


DEVICE_UUID = (
    make_stable_device_id()
)

DEVICE_SHORT = (
    DEVICE_UUID[:8]
)

DEVICE_LOG_DIR = (
    OUTPUT_ROOT
    / DEVICE_SHORT
)

DEVICE_LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def get_hostname():

    return (
        socket.gethostname()
    )


# ============================================================
# 6. GPU STATIC INFORMATION
# ============================================================

def get_cuda_driver_version():

    if not NVML_AVAILABLE:
        return None

    try:

        value = (
            pynvml
            .nvmlSystemGetDriverVersion()
        )

        if isinstance(
            value,
            bytes,
        ):
            return value.decode(
                "utf-8"
            )

        return value

    except Exception:
        return None


CUDA_DRIVER_VERSION = (
    get_cuda_driver_version()
)


def get_gpu_static():

    result = {
        "gpu_driver_version":
            CUDA_DRIVER_VERSION,

        "gpu_compute_capability":
            None,

        "gpu_power_limit_w":
            None,

        "gpu_memory_total_mb":
            None,
    }

    if torch.cuda.is_available():

        try:

            props = (
                torch.cuda
                .get_device_properties(0)
            )

            result[
                "gpu_compute_capability"
            ] = (
                f"{props.major}."
                f"{props.minor}"
            )

        except Exception:
            pass

    if (
        not NVML_AVAILABLE
        or NVML_HANDLE is None
    ):
        return result

    try:

        power_limit = (
            pynvml
            .nvmlDeviceGetPowerManagementLimit(
                NVML_HANDLE
            )
        )

        memory = (
            pynvml
            .nvmlDeviceGetMemoryInfo(
                NVML_HANDLE
            )
        )

        result[
            "gpu_power_limit_w"
        ] = round(
            power_limit / 1000.0,
            2,
        )

        result[
            "gpu_memory_total_mb"
        ] = round(
            memory.total
            / (1024 ** 2),
            2,
        )

    except Exception:
        pass

    return result


GPU_STATIC = (
    get_gpu_static()
)


def get_gpu_core_thread():

    if not torch.cuda.is_available():

        return (
            None,
            None,
        )

    try:

        props = (
            torch.cuda
            .get_device_properties(0)
        )

        sm_count = (
            props.multi_processor_count
        )

        cores_per_sm = {
            2: 32,
            3: 192,
            5: 128,
            6: 64,
            7: 64,
            8: 128,
            9: 128,
        }.get(
            props.major,
            64,
        )

        gpu_core_count = (
            sm_count
            * cores_per_sm
        )

        gpu_thread_count = (
            sm_count
            * props
            .max_threads_per_multi_processor
        )

        return (
            gpu_core_count,
            gpu_thread_count,
        )

    except Exception:

        return (
            None,
            None,
        )


(
    GPU_CORE_COUNT,
    GPU_THREAD_COUNT,
) = get_gpu_core_thread()


# ============================================================
# 7. RUNTIME HARDWARE HELPERS
# ============================================================

def get_gpu_name():

    try:

        if torch.cuda.is_available():

            return (
                torch.cuda
                .get_device_name(0)
            )

        return "No GPU"

    except Exception:

        return "Unknown"


def get_cpu_usage():

    try:

        return (
            psutil.cpu_percent(
                interval=None
            )
        )

    except Exception:

        return None


def get_cpu_freq():

    try:

        freq = (
            psutil.cpu_freq()
        )

        return (
            round(
                freq.current,
                2,
            )
            if freq
            else None
        )

    except Exception:

        return None


def get_memory_footprint_mb():

    try:

        return round(
            psutil
            .Process(
                os.getpid()
            )
            .memory_info()
            .rss
            / (1024 ** 2),
            4,
        )

    except Exception:

        return None


def get_cpu_temp():

    try:

        temps = (
            psutil
            .sensors_temperatures()
        )

        if not temps:

            return None

        for key in (
            "coretemp",
            "k10temp",
            "cpu_thermal",
            "acpitz",
        ):

            if key in temps:

                values = [
                    item.current
                    for item
                    in temps[key]
                    if (
                        item.current
                        is not None
                        and item.current > 0
                    )
                ]

                if values:

                    return round(
                        sum(values)
                        / len(values),
                        1,
                    )

    except Exception:
        pass

    return None


def get_cpu_power_draw_w():

    return None


def get_cpu_cores_used():

    try:

        return sum(
            1
            for value
            in psutil.cpu_percent(
                percpu=True
            )
            if value > 1.0
        )

    except Exception:

        return None


def get_gpu_metrics():

    result = {
        "gpu_power_draw_w":
            None,

        "gpu_utilization_pct":
            None,

        "gpu_temp_c":
            None,

        "gpu_memory_used_mb":
            None,

        "gpu_sm_clock_mhz":
            None,

        "gpu_memory_clock_mhz":
            None,
    }

    if (
        not NVML_AVAILABLE
        or NVML_HANDLE is None
    ):

        return result

    try:

        power = (
            pynvml
            .nvmlDeviceGetPowerUsage(
                NVML_HANDLE
            )
        )

        utilization = (
            pynvml
            .nvmlDeviceGetUtilizationRates(
                NVML_HANDLE
            )
        )

        temperature = (
            pynvml
            .nvmlDeviceGetTemperature(
                NVML_HANDLE,
                pynvml.NVML_TEMPERATURE_GPU,
            )
        )

        memory = (
            pynvml
            .nvmlDeviceGetMemoryInfo(
                NVML_HANDLE
            )
        )

        sm_clock = (
            pynvml
            .nvmlDeviceGetClockInfo(
                NVML_HANDLE,
                pynvml.NVML_CLOCK_SM,
            )
        )

        memory_clock = (
            pynvml
            .nvmlDeviceGetClockInfo(
                NVML_HANDLE,
                pynvml.NVML_CLOCK_MEM,
            )
        )

        return {
            "gpu_power_draw_w":
                round(
                    power / 1000.0,
                    2,
                ),

            "gpu_utilization_pct":
                utilization.gpu,

            "gpu_temp_c":
                temperature,

            "gpu_memory_used_mb":
                round(
                    memory.used
                    / (1024 ** 2),
                    2,
                ),

            "gpu_sm_clock_mhz":
                sm_clock,

            "gpu_memory_clock_mhz":
                memory_clock,
        }

    except Exception:

        return result


# ============================================================
# 8. EXACT REFERENCE COLUMN ORDER
# ============================================================

REFERENCE_COLUMNS = [
    "timestamp",
    "unique_device_id",
    "device_short_id",
    "pc_name",
    "collection_mode",
    "sample_index",
    "sample_id",
    "true_label",
    "prediction",
    "correct",
    "dataset",
    "model_type",
    "parameters",
    "model_flops",
    "checkpoint_path",
    "confidence_score",
    "logit_margin",
    "entropy",
    "execution_time_sec",
    "cpu_energy_kwh",
    "gpu_energy_kwh",
    "ram_energy_kwh",
    "total_energy_kwh",
    "total_emissions_kg",
    "carbon_intensity_kgco2_kwh",
    "codecarbon_version",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "tokens_per_second",
    "joules_per_token",
    "energy_per_token_kwh",
    "watts_estimated",
    "gpu_energy_pct_of_total",
    "cpu_energy_pct_of_total",
    "cpu_model",
    "cpu_architecture",
    "cpu_core_count",
    "cpu_thread_count",
    "cpu_core",
    "cpu_thread",
    "cpu_tdp_w",
    "cpu_usage_pct",
    "cpu_clock_mhz",
    "cpu_temp_c",
    "cpu_power_draw_w",
    "cpu_cores_used",
    "gpu_model",
    "gpu_core",
    "gpu_thread",
    "gpu_driver_version",
    "gpu_compute_capability",
    "gpu_power_limit_w",
    "gpu_memory_total_mb",
    "gpu_power_draw_w",
    "gpu_utilization_pct",
    "gpu_temp_c",
    "gpu_memory_used_mb",
    "gpu_sm_clock_mhz",
    "gpu_memory_clock_mhz",
    "cuda_driver_version",
    "cuda_available",
    "device_type",
    "ram_usage_pct",
    "memory_footprint_mb",
    "system_ram_total_gb",
    "os_name",
    "os_version",
    "os_architecture",
    "os_full_name",
    "python_version",
    "torch_version",
    "model_accuracy",
    "model_precision_weighted",
    "model_recall_weighted",
    "model_f1_weighted",
    "quantum_computing",
    "model_under_attack",
    "source_qubits",
    "n_qubits",
    "fidelity",
    "pennylane_device",
    "pennylane_version",
]


# ============================================================
# 9. PREDICTION QUALITY
# ============================================================

def get_prediction_quality(
    logits,
):

    probs = (
        F.softmax(
            logits.float(),
            dim=-1,
        )
        .squeeze()
    )

    confidence = float(
        probs.max().item()
    )

    top2 = (
        torch.topk(
            logits.float().squeeze(),
            k=2,
        )
        .values
    )

    margin = float(
        (
            top2[0]
            - top2[1]
        ).item()
    )

    entropy = float(
        -(
            probs
            * torch.log(
                probs + 1e-12
            )
        )
        .sum()
        .item()
    )

    return (
        round(
            confidence,
            6,
        ),

        round(
            margin,
            6,
        ),

        round(
            entropy,
            6,
        ),
    )


# ============================================================
# 10. ENERGY TRACKING
# ============================================================

def extract_energy_data(
    tracker,
    emissions_value,
):

    final_data = getattr(
        tracker,
        "final_emissions_data",
        None,
    )

    cpu_energy = (
        getattr(
            final_data,
            "cpu_energy",
            0,
        )
        if final_data
        else 0
    )

    gpu_energy = (
        getattr(
            final_data,
            "gpu_energy",
            0,
        )
        if final_data
        else 0
    )

    ram_energy = (
        getattr(
            final_data,
            "ram_energy",
            0,
        )
        if final_data
        else 0
    )

    total_energy = (
        getattr(
            final_data,
            "energy_consumed",
            0,
        )
        if final_data
        else 0
    )

    carbon_intensity = None

    if (
        emissions_value
        and total_energy
        and total_energy > 0
    ):

        carbon_intensity = (
            emissions_value
            / total_energy
        )

    return (
        float(cpu_energy or 0),
        float(gpu_energy or 0),
        float(ram_energy or 0),
        float(total_energy or 0),
        carbon_intensity,
    )


def run_with_energy_tracking(
    inference_fn,
    *args,
    project_name,
    codecarbon_filename,
    **kwargs,
):

    energy_dir = (
        OUTPUT_ROOT
        / "codecarbon"
    )

    energy_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if CODECARBON_AVAILABLE:

        tracker = (
            EmissionsTracker(
                project_name=
                    project_name,

                output_dir=
                    str(
                        energy_dir
                    ),

                output_file=
                    codecarbon_filename,

                log_level=
                    "error",

                save_to_file=
                    True,
            )
        )

        tracker.start()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = (
            time.perf_counter()
        )

        result = (
            inference_fn(
                *args,
                **kwargs,
            )
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        exec_time = (
            time.perf_counter()
            - t0
        )

        emissions_value = (
            tracker.stop()
        )

        (
            cpu_energy,
            gpu_energy,
            ram_energy,
            total_energy,
            carbon_intensity,
        ) = extract_energy_data(
            tracker,
            emissions_value,
        )

    else:

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = (
            time.perf_counter()
        )

        result = (
            inference_fn(
                *args,
                **kwargs,
            )
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        exec_time = (
            time.perf_counter()
            - t0
        )

        cpu_energy = 0.0
        gpu_energy = 0.0
        ram_energy = 0.0
        total_energy = 0.0
        emissions_value = 0.0
        carbon_intensity = None

    gpu_metrics = (
        get_gpu_metrics()
    )

    return (
        result,
        exec_time,
        cpu_energy,
        gpu_energy,
        ram_energy,
        total_energy,
        emissions_value,
        carbon_intensity,
        gpu_metrics,
    )


# ============================================================
# 11. EXACT TELEMETRY ROW
# ============================================================

def build_row(
    *,
    model_name,
    prediction,
    exec_time,
    parameters,
    true_label,
    sample_index,
    logits,
    cpu_energy,
    gpu_energy,
    ram_energy,
    total_energy,
    emissions_value,
    carbon_intensity,
    gpu_metrics,
    model_flops,
    dataset_name,
    checkpoint_path,
):

    (
        confidence_score,
        logit_margin,
        entropy,
    ) = get_prediction_quality(
        logits
    )

    total_energy = (
        total_energy
        or 0.0
    )

    cpu_energy = (
        cpu_energy
        or 0.0
    )

    gpu_energy = (
        gpu_energy
        or 0.0
    )

    ram_energy = (
        ram_energy
        or 0.0
    )

    # For exact schema compatibility with the user's existing
    # CNN/DNN telemetry, use the same 784-pixel proxy.
    input_tokens = 784
    output_tokens = 1

    total_tokens = (
        input_tokens
        + output_tokens
    )

    joules_per_token = 0.0
    energy_per_token_kwh = 0.0
    watts_estimated = 0.0
    gpu_energy_pct = 0.0
    cpu_energy_pct = 0.0

    if (
        total_energy > 0
        and total_tokens > 0
    ):

        energy_per_token_kwh = round(
            total_energy
            / total_tokens,
            12,
        )

        joules_per_token = round(
            (
                total_energy
                * 3_600_000
            )
            / total_tokens,
            6,
        )

        if exec_time > 0:

            watts_estimated = round(
                (
                    total_energy
                    * 3_600_000
                )
                / exec_time,
                4,
            )

        gpu_energy_pct = round(
            (
                gpu_energy
                / total_energy
            )
            * 100,
            2,
        )

        cpu_energy_pct = round(
            (
                cpu_energy
                / total_energy
            )
            * 100,
            2,
        )

    correct = (
        int(prediction)
        == int(true_label)
    )

    gpu_metrics = (
        gpu_metrics
        or {}
    )

    row = {
        # --- Identity ---
        "timestamp":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "unique_device_id":
            DEVICE_UUID,

        "device_short_id":
            DEVICE_SHORT,

        "pc_name":
            get_hostname(),

        "collection_mode":
            "automated_edge",

        # --- Sample ---
        "sample_index":
            sample_index,

        "sample_id":
            sample_index,

        "true_label":
            int(true_label),

        "prediction":
            int(prediction),

        "correct":
            correct,

        # --- Model identity ---
        "dataset":
            dataset_name,

        "model_type":
            model_name,

        "parameters":
            parameters,

        "model_flops":
            model_flops,

        "checkpoint_path":
            str(
                checkpoint_path
            ),

        # --- Prediction quality ---
        "confidence_score":
            confidence_score,

        "logit_margin":
            logit_margin,

        "entropy":
            entropy,

        # --- Timing ---
        "execution_time_sec":
            round(
                exec_time,
                10,
            ),

        # --- CodeCarbon energy ---
        "cpu_energy_kwh":
            cpu_energy,

        "gpu_energy_kwh":
            gpu_energy,

        "ram_energy_kwh":
            ram_energy,

        "total_energy_kwh":
            total_energy,

        "total_emissions_kg":
            emissions_value,

        "carbon_intensity_kgco2_kwh":
            carbon_intensity,

        "codecarbon_version":
            CODECARBON_VERSION,

        # --- Efficiency derived ---
        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "total_tokens":
            total_tokens,

        "tokens_per_second":
            (
                round(
                    total_tokens
                    / exec_time,
                    4,
                )
                if exec_time > 0
                else None
            ),

        "joules_per_token":
            joules_per_token,

        "energy_per_token_kwh":
            energy_per_token_kwh,

        "watts_estimated":
            watts_estimated,

        "gpu_energy_pct_of_total":
            gpu_energy_pct,

        "cpu_energy_pct_of_total":
            cpu_energy_pct,

        # --- CPU hardware ---
        "cpu_model":
            CPU_MODEL_NAME,

        "cpu_architecture":
            CPU_ARCH,

        "cpu_core_count":
            CPU_CORE_COUNT,

        "cpu_thread_count":
            CPU_THREAD_COUNT,

        "cpu_core":
            CPU_CORE_COUNT,

        "cpu_thread":
            CPU_THREAD_COUNT,

        "cpu_tdp_w":
            CPU_TDP_W,

        "cpu_usage_pct":
            get_cpu_usage(),

        "cpu_clock_mhz":
            get_cpu_freq(),

        "cpu_temp_c":
            get_cpu_temp(),

        "cpu_power_draw_w":
            get_cpu_power_draw_w(),

        "cpu_cores_used":
            get_cpu_cores_used(),

        # --- GPU hardware ---
        "gpu_model":
            get_gpu_name(),

        "gpu_core":
            GPU_CORE_COUNT,

        "gpu_thread":
            GPU_THREAD_COUNT,

        "gpu_driver_version":
            GPU_STATIC[
                "gpu_driver_version"
            ],

        "gpu_compute_capability":
            GPU_STATIC[
                "gpu_compute_capability"
            ],

        "gpu_power_limit_w":
            GPU_STATIC[
                "gpu_power_limit_w"
            ],

        "gpu_memory_total_mb":
            GPU_STATIC[
                "gpu_memory_total_mb"
            ],

        "gpu_power_draw_w":
            gpu_metrics.get(
                "gpu_power_draw_w"
            ),

        "gpu_utilization_pct":
            gpu_metrics.get(
                "gpu_utilization_pct"
            ),

        "gpu_temp_c":
            gpu_metrics.get(
                "gpu_temp_c"
            ),

        "gpu_memory_used_mb":
            gpu_metrics.get(
                "gpu_memory_used_mb"
            ),

        "gpu_sm_clock_mhz":
            gpu_metrics.get(
                "gpu_sm_clock_mhz"
            ),

        "gpu_memory_clock_mhz":
            gpu_metrics.get(
                "gpu_memory_clock_mhz"
            ),

        "cuda_driver_version":
            CUDA_DRIVER_VERSION,

        "cuda_available":
            torch.cuda.is_available(),

        "device_type":
            str(
                DEVICE
            ),

        # --- RAM / memory ---
        "ram_usage_pct":
            psutil
            .virtual_memory()
            .percent,

        "memory_footprint_mb":
            get_memory_footprint_mb(),

        "system_ram_total_gb":
            SYSTEM_RAM_TOTAL_GB,

        # --- Environment ---
        "os_name":
            OS_NAME,

        "os_version":
            OS_VERSION,

        "os_architecture":
            OS_ARCHITECTURE,

        "os_full_name":
            OS_FULL_NAME,

        "python_version":
            PYTHON_VERSION,

        "torch_version":
            TORCH_VERSION,

        # --- Final model metrics ---
        "model_accuracy":
            None,

        "model_precision_weighted":
            None,

        "model_recall_weighted":
            None,

        "model_f1_weighted":
            None,

        # --- Context ---
        "quantum_computing":
            False,

        "model_under_attack":
            0,

        # --- Quantum-only placeholders ---
        "source_qubits":
            None,

        "n_qubits":
            None,

        "fidelity":
            None,

        "pennylane_device":
            None,

        "pennylane_version":
            None,
    }

    # Hard assertion: exact schema/order.
    assert (
        list(row.keys())
        == REFERENCE_COLUMNS
    ), (
        "Telemetry schema does not "
        "match reference columns."
    )

    return row


# ============================================================
# 12. DATASETS
# ============================================================

def load_raw_test_dataset(
    dataset_class,
):

    # Keep PIL images.
    # Each model applies its own preprocessing.
    return dataset_class(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=None,
    )


# ============================================================
# 13. MOBILENETV2
# ============================================================

MOBILENET_TRANSFORM = (
    transforms.Compose([
        transforms.Resize(
            (224, 224)
        ),

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(
                0.485,
                0.456,
                0.406,
            ),
            std=(
                0.229,
                0.224,
                0.225,
            ),
        ),
    ])
)


def load_mobilenet(
    checkpoint_path,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"MobileNetV2 checkpoint "
            f"not found: "
            f"{checkpoint_path}"
        )

    # IMPORTANT:
    # weights=None prevents another ImageNet
    # download. The trained state_dict below
    # contains the saved learned parameters.
    model = (
        mobilenet_v2(
            weights=None
        )
    )

    in_features = (
        model.classifier[1]
        .in_features
    )

    model.classifier[1] = (
        nn.Linear(
            in_features,
            10,
        )
    )

    state = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        state
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    parameters = int(
        sum(
            p.numel()
            for p
            in model.parameters()
        )
    )

    return (
        model,
        parameters,
    )


def run_mobilenet_inference(
    model,
    image,
):

    input_tensor = (
        MOBILENET_TRANSFORM(
            image
        )
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )

    with torch.inference_mode():

        logits = (
            model(
                input_tensor
            )
        )

    prediction = int(
        torch.argmax(
            logits,
            dim=1,
        ).item()
    )

    return (
        prediction,
        logits.detach(),
    )


# ============================================================
# 14. YOLO CLASSIFICATION
# ============================================================

def load_yolo(
    checkpoint_path,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"YOLO checkpoint "
            f"not found: "
            f"{checkpoint_path}"
        )

    model = YOLO(
        str(
            checkpoint_path
        )
    )

    # Underlying PyTorch model.
    torch_model = (
        model.model
    )

    torch_model.to(
        DEVICE
    )

    torch_model.eval()

    parameters = int(
        sum(
            p.numel()
            for p
            in torch_model.parameters()
        )
    )

    return (
        model,
        parameters,
    )


def run_yolo_inference(
    model,
    image,
):

    # Training images were converted to RGB.
    image = image.convert(
        "RGB"
    )

    # Ultralytics classification result.
    results = model.predict(
        source=image,
        imgsz=64,
        device=(
            0
            if DEVICE.type == "cuda"
            else "cpu"
        ),
        verbose=False,
    )

    result = results[0]

    if (
        result.probs is None
        or result.probs.data is None
    ):

        raise RuntimeError(
            "YOLO classification did "
            "not return class probabilities."
        )

    probs = (
        result.probs.data
        .detach()
        .float()
    )

    # Convert probabilities back to log-space
    # so the same confidence/margin/entropy
    # helper can be applied consistently.
    logits = torch.log(
        probs + 1e-12
    ).unsqueeze(0)

    prediction = int(
        torch.argmax(
            probs
        ).item()
    )

    return (
        prediction,
        logits,
    )


# ============================================================
# 15. FINAL METRICS + SAVE
# ============================================================

def finalize_and_save(
    rows,
    output_path,
):

    if not rows:

        raise RuntimeError(
            "No telemetry rows "
            "were generated."
        )

    df = pd.DataFrame(
        rows
    )

    # Force exact 83-column schema.
    df = df.reindex(
        columns=
            REFERENCE_COLUMNS
    )

    y_true = (
        df[
            "true_label"
        ]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        df[
            "prediction"
        ]
        .astype(int)
        .to_numpy()
    )

    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    precision = float(
        precision_score(
            y_true,
            y_pred,
            labels=list(
                range(10)
            ),
            average=
                "weighted",
            zero_division=
                0,
        )
    )

    recall = float(
        recall_score(
            y_true,
            y_pred,
            labels=list(
                range(10)
            ),
            average=
                "weighted",
            zero_division=
                0,
        )
    )

    f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=list(
                range(10)
            ),
            average=
                "weighted",
            zero_division=
                0,
        )
    )

    df[
        "model_accuracy"
    ] = accuracy

    df[
        "model_precision_weighted"
    ] = precision

    df[
        "model_recall_weighted"
    ] = recall

    df[
        "model_f1_weighted"
    ] = f1

    # Reindex again after backfill.
    df = df.reindex(
        columns=
            REFERENCE_COLUMNS
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved telemetry: "
        f"{output_path.resolve()}"
    )

    print(
        f"Accuracy:  "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall:    "
        f"{recall:.4f}"
    )

    print(
        f"F1:        "
        f"{f1:.4f}"
    )

    return {
        "accuracy":
            accuracy,

        "precision_weighted":
            precision,

        "recall_weighted":
            recall,

        "f1_weighted":
            f1,
    }


# ============================================================
# 16. GENERIC MODEL TEST LOOP
# ============================================================

def test_model(
    *,
    dataset_name,
    dataset_key,
    dataset,
    model_name,
    checkpoint_path,
    model,
    parameters,
    inference_fn,
):

    limit = min(
        NUM_TEST_SAMPLES,
        len(dataset),
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"TESTING "
        f"{model_name} "
        f"ON "
        f"{dataset_name}"
    )

    print(
        "=" * 72
    )

    print(
        "Samples:",
        limit,
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    rows = []

    for sample_index in tqdm(
        range(limit),
        desc=(
            f"{model_name} | "
            f"{dataset_name}"
        ),
        unit="sample",
    ):

        image, true_label = (
            dataset[
                sample_index
            ]
        )

        (
            (
                prediction,
                logits,
            ),
            exec_time,
            cpu_energy,
            gpu_energy,
            ram_energy,
            total_energy,
            emissions_value,
            carbon_intensity,
            gpu_metrics,
        ) = run_with_energy_tracking(
            inference_fn,
            model,
            image,
            project_name=(
                f"{model_name}_"
                f"{dataset_key}_test"
            ),
            codecarbon_filename=(
                f"codecarbon_"
                f"{model_name.lower()}_"
                f"{dataset_key}.csv"
            ),
        )

        row = build_row(
            model_name=
                model_name,

            prediction=
                prediction,

            exec_time=
                exec_time,

            parameters=
                parameters,

            true_label=
                int(
                    true_label
                ),

            sample_index=
                sample_index,

            logits=
                logits,

            cpu_energy=
                cpu_energy,

            gpu_energy=
                gpu_energy,

            ram_energy=
                ram_energy,

            total_energy=
                total_energy,

            emissions_value=
                emissions_value,

            carbon_intensity=
                carbon_intensity,

            gpu_metrics=
                gpu_metrics,

            model_flops=
                None,

            dataset_name=
                dataset_name,

            checkpoint_path=
                checkpoint_path,
        )

        rows.append(
            row
        )

    output_path = (
        DEVICE_LOG_DIR
        / (
            f"{model_name.lower()}_"
            f"{dataset_key}_"
            f"test_telemetry.csv"
        )
    )

    return finalize_and_save(
        rows,
        output_path,
    )


# ============================================================
# 17. MAIN
# ============================================================

def yolo_mobilenet_main():

    print(
        "=" * 72
    )

    print(
        "YOLO + MOBILENETV2 "
        "EXACT TELEMETRY TEST"
    )

    print(
        "=" * 72
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Reference column count:",
        len(
            REFERENCE_COLUMNS
        ),
    )

    all_summaries = []

    for (
        dataset_name,
        config,
    ) in DATASET_CONFIGS.items():

        dataset = (
            load_raw_test_dataset(
                config[
                    "dataset_class"
                ]
            )
        )

        # ====================================================
        # YOLO
        # ====================================================

        yolo_model, yolo_parameters = (
            load_yolo(
                config[
                    "yolo_checkpoint"
                ]
            )
        )

        yolo_metrics = test_model(
            dataset_name=
                dataset_name,

            dataset_key=
                config["key"],

            dataset=
                dataset,

            model_name=
                "YOLO26n-CLS",

            checkpoint_path=
                config[
                    "yolo_checkpoint"
                ],

            model=
                yolo_model,

            parameters=
                yolo_parameters,

            inference_fn=
                run_yolo_inference,
        )

        all_summaries.append({
            "model":
                "YOLO26n-CLS",

            "dataset":
                dataset_name,

            **yolo_metrics,
        })

        del yolo_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ====================================================
        # MobileNetV2
        # ====================================================

        (
            mobilenet_model,
            mobilenet_parameters,
        ) = load_mobilenet(
            config[
                "mobilenet_checkpoint"
            ]
        )

        mobilenet_metrics = test_model(
            dataset_name=
                dataset_name,

            dataset_key=
                config["key"],

            dataset=
                dataset,

            model_name=
                "MobileNetV2",

            checkpoint_path=
                config[
                    "mobilenet_checkpoint"
                ],

            model=
                mobilenet_model,

            parameters=
                mobilenet_parameters,

            inference_fn=
                run_mobilenet_inference,
        )

        all_summaries.append({
            "model":
                "MobileNetV2",

            "dataset":
                dataset_name,

            **mobilenet_metrics,
        })

        del mobilenet_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = (
        DEVICE_LOG_DIR
        / (
            "yolo_mobilenetv2_"
            "three_dataset_summary.csv"
        )
    )

    # pd.DataFrame(
    #     all_summaries
    # ).to_csv(
    #     summary_path,
    #     index=False,
    # )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ALL TESTS COMPLETE"
    )


import hashlib
import os
import platform
import socket
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# =============================================================================
# Retrieves env var 'NUM_TEST_SAMPLES', defaults to 10 if not set
NUM_TEST_SAMPLES = int(os.getenv("NUM_TEST_SAMPLES", 10000))

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc1 = nn.Linear(
            64 * 5 * 5,
            128
        )

        self.fc2 = nn.Linear(
            128,
            10
        )

    def forward(self, x):

        x = self.pool(
            F.relu(
                self.conv1(x)
            )
        )

        x = self.pool(
            F.relu(
                self.conv2(x)
            )
        )

        x = torch.flatten(
            x,
            1
        )

        x = F.relu(
            self.fc1(x)
        )

        return self.fc2(x)


# ============================================================
# DNN MODEL
# ============================================================

class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()

        self.fc1 = nn.Linear(
            28 * 28,
            512
        )

        self.fc2 = nn.Linear(
            512,
            256
        )

        self.fc3 = nn.Linear(
            256,
            128
        )

        self.fc4 = nn.Linear(
            128,
            10
        )

    def forward(self, x):

        x = torch.flatten(
            x,
            1
        )

        x = F.relu(
            self.fc1(x)
        )

        x = F.relu(
            self.fc2(x)
        )

        x = F.relu(
            self.fc3(x)
        )

        return self.fc4(x)

#from utils.llm_utils import load_tiny_llm_model, predict_with_tiny_llm
#from utils.vlm_utils import load_tiny_vlm_model, predict_with_tiny_vlm



def get_cpu_model():
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info().get("brand_raw", "Unknown")
    except Exception:
        return platform.processor() or "Unknown"

CPU_MODEL_NAME = get_cpu_model()
def make_stable_device_id():
    raw = f"{socket.gethostname()}-{platform.system()}-{platform.machine()}-{CPU_MODEL_NAME}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

DEVICE_UUID = make_stable_device_id()
DEVICE_SHORT = DEVICE_UUID[:8]

OUTPUT_ROOT = Path.cwd() / "test_results"
OUTPUT_ROOT.mkdir(exist_ok=True)
DEVICE_LOG_DIR = OUTPUT_ROOT / f"{DEVICE_SHORT}"
DEVICE_LOG_DIR.mkdir(exist_ok=True)




def get_cpu_model():
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info().get("brand_raw", "Unknown")
    except Exception:
        return platform.processor() or "Unknown"

CPU_MODEL_NAME = get_cpu_model()
def make_stable_device_id():
    raw = f"{socket.gethostname()}-{platform.system()}-{platform.machine()}-{CPU_MODEL_NAME}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


DEVICE_UUID = make_stable_device_id()
DEVICE_SHORT = DEVICE_UUID[:8]

OUTPUT_ROOT = Path.cwd() / "test_results"
OUTPUT_ROOT.mkdir(exist_ok=True)
DEVICE_LOG_DIR = OUTPUT_ROOT / f"{DEVICE_SHORT}"
DEVICE_LOG_DIR.mkdir(exist_ok=True)

DATA_ROOT = Path("./classical_data")

# VERBOSE_DATASET_PATH = Path(
#     "mnist_fgsm_test_dataset.pt"
# )

# ============================================================
# Dataset configuration: MNIST, FashionMNIST, Kuzushiji-MNIST
# ============================================================
# Each dataset uses its own torchvision class, its own per-channel
# normalization stats, and its own saved model checkpoint prefix
# (produced by the training script as "<name>_cnn.pth" / "<name>_dnn.pth").

DATASET_CONFIGS = {
    "MNIST": {
        "dataset_class": datasets.MNIST,
        "mean": (0.1307,),
        "std": (0.3081,),
        "checkpoint_prefix": "mnist",
    },
    "FashionMNIST": {
        "dataset_class": datasets.FashionMNIST,
        "mean": (0.2860,),
        "std": (0.3530,),
        "checkpoint_prefix": "fashionmnist",
    },
    "KMNIST": {
        "dataset_class": datasets.KMNIST,  # Kuzushiji-MNIST
        "mean": (0.1918,),
        "std": (0.3483,),
        "checkpoint_prefix": "kmnist",
    },
}

# =============================================================================



torch.set_grad_enabled(False)



DEVICE_MODE = os.getenv("DEVICE_MODE", "cpu")  # "cpu" | "cuda" | "auto"


def resolve_device():
    if DEVICE_MODE.lower() == "cpu":
        return torch.device("cpu")
    if DEVICE_MODE.lower() == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = torch.device("cpu")#resolve_device()


# ============================================================
# Safe optional imports
# ============================================================

# ---- CodeCarbon ----
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

# ---- pynvml ----
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0) if torch.cuda.is_available() else None
except Exception:
    NVML_AVAILABLE = False
    NVML_HANDLE = None
    print("pynvml not available.")

# ---- py-cpuinfo ----
try:
    import cpuinfo
    _CPU_INFO = cpuinfo.get_cpu_info()
    CPU_MODEL_NAME = _CPU_INFO.get("brand_raw", "Unknown")
    CPU_ARCH = _CPU_INFO.get("arch", platform.machine())
    CPU_TDP_W = None
except Exception:
    CPU_MODEL_NAME = "Unknown"
    CPU_ARCH = platform.machine()
    CPU_TDP_W = None
    print("cpuinfo not available.")

# ---- fvcore: FLOPs ----
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except Exception:
    FlopCountAnalysis = None
    FVCORE_AVAILABLE = False
    print("fvcore not available.")


# ============================================================
# OS / environment constants
# ============================================================

def get_os_full_name():
    system = platform.system()
    architecture = platform.machine()

    if system == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
            )
            product_name = winreg.QueryValueEx(key, "ProductName")[0]
            display_version = winreg.QueryValueEx(key, "DisplayVersion")[0]
            current_build = winreg.QueryValueEx(key, "CurrentBuild")[0]
            return f"{product_name} {display_version} Build {current_build} {architecture}"
        except Exception:
            return f"Windows {platform.release()} {architecture}"

    if system == "Linux":
        os_info = {}
        try:
            with open("/etc/os-release", "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        os_info[k] = v.strip('"')
        except Exception:
            pass
        pretty_name = os_info.get("PRETTY_NAME")
        name = os_info.get("NAME")
        version = os_info.get("VERSION")
        version_id = os_info.get("VERSION_ID")
        distro_id = os_info.get("ID")
        if pretty_name:
            return f"{pretty_name} {architecture}"
        if name and version:
            return f"{name} {version} {architecture}"
        if name and version_id:
            return f"{name} {version_id} {architecture}"
        if distro_id:
            return f"{distro_id} {platform.release()} {architecture}"
        return f"Linux {platform.release()} {architecture}"

    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]} {architecture}"

    return f"{system} {platform.release()} {architecture}"


TORCH_VERSION = torch.__version__
PYTHON_VERSION = sys.version.split()[0]
OS_NAME = platform.system()
OS_VERSION = platform.version()
OS_ARCHITECTURE = platform.machine()
OS_FULL_NAME = get_os_full_name()
SYSTEM_RAM_TOTAL_GB = round(psutil.virtual_memory().total / (1024 ** 3), 2)
CPU_CORE_COUNT = psutil.cpu_count(logical=False)
CPU_THREAD_COUNT = psutil.cpu_count(logical=True)


# ============================================================
# Stable device ID (SHA256-based, consistent across runs)
# ============================================================

def make_stable_device_id():
    raw = f"{socket.gethostname()}-{platform.system()}-{platform.machine()}-{CPU_MODEL_NAME}"
    print(f"Generating device ID from: {raw}")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


DEVICE_UUID = make_stable_device_id()
DEVICE_SHORT = DEVICE_UUID[:8]


# ============================================================
# GPU static info
# ============================================================

def _get_cuda_driver_version():
    if not NVML_AVAILABLE:
        return None
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        return driver.decode("utf-8") if isinstance(driver, bytes) else driver
    except Exception:
        return None


CUDA_DRIVER_VERSION = _get_cuda_driver_version()


def _get_gpu_static():
    defaults = {
        "gpu_power_limit_w": None,
        "gpu_driver_version": CUDA_DRIVER_VERSION,
        "gpu_memory_total_mb": None,
        "gpu_compute_capability": None,
    }
    if not NVML_AVAILABLE or NVML_HANDLE is None:
        return defaults
    try:
        power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(NVML_HANDLE)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)
        cc_major, cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(NVML_HANDLE)
        return {
            "gpu_power_limit_w": round(power_limit_mw / 1000.0, 1),
            "gpu_driver_version": CUDA_DRIVER_VERSION,
            "gpu_memory_total_mb": round(mem_info.total / (1024 ** 2), 2),
            "gpu_compute_capability": f"{cc_major}.{cc_minor}",
        }
    except Exception:
        return defaults


GPU_STATIC = _get_gpu_static()


def _get_gpu_core_thread():
    """
    Return (gpu_core_count, gpu_thread_count) where:
      - gpu_core_count  = total CUDA cores  (multiprocessor_count * cores_per_sm)
      - gpu_thread_count = max threads per device (gpu_core_count * max_threads_per_block,
                           capped to a sensible ceiling via device properties)
    Falls back to torch.cuda device properties when pynvml SM count is unavailable.
    """
    if not torch.cuda.is_available():
        return None, None

    try:
        props = torch.cuda.get_device_properties(0)
        sm_count = props.multi_processor_count

        # Cores-per-SM lookup by compute capability major version
        cc_major = props.major
        cores_per_sm_map = {
            2: 32,   # Fermi
            3: 192,  # Kepler
            5: 128,  # Maxwell
            6: 64,   # Pascal (GP100=64, GP10x=128 — use 64 as conservative default)
            7: 64,   # Volta / Turing
            8: 128,  # Ampere
            9: 128,  # Ada Lovelace / Hopper
        }
        cores_per_sm = cores_per_sm_map.get(cc_major, 64)
        gpu_core_count = sm_count * cores_per_sm

        # gpu_thread_count = cores * max_threads_per_multiprocessor
        gpu_thread_count = sm_count * props.max_threads_per_multi_processor

        return gpu_core_count, gpu_thread_count

    except Exception:
        return None, None


GPU_CORE_COUNT, GPU_THREAD_COUNT = _get_gpu_core_thread()


# ============================================================
# Per-sample hardware helpers
# ============================================================

def get_hostname():
    return socket.gethostname()


def get_gpu_name():
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "No GPU"
    except Exception:
        return "Unknown"


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


def get_memory_footprint_mb():
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 4)
    except Exception:
        return None


def get_gpu_metrics():
    null = {
        "gpu_power_draw_w": None,
        "gpu_utilization_pct": None,
        "gpu_temp_c": None,
        "gpu_memory_used_mb": None,
        "gpu_sm_clock_mhz": None,
        "gpu_memory_clock_mhz": None,
    }
    if not NVML_AVAILABLE or NVML_HANDLE is None:
        return null
    try:
        power_mw = pynvml.nvmlDeviceGetPowerUsage(NVML_HANDLE)
        util = pynvml.nvmlDeviceGetUtilizationRates(NVML_HANDLE)
        temp = pynvml.nvmlDeviceGetTemperature(NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)
        sm_clock = pynvml.nvmlDeviceGetClockInfo(NVML_HANDLE, pynvml.NVML_CLOCK_SM)
        mem_clock = pynvml.nvmlDeviceGetClockInfo(NVML_HANDLE, pynvml.NVML_CLOCK_MEM)
        return {
            "gpu_power_draw_w": round(power_mw / 1000.0, 2),
            "gpu_utilization_pct": util.gpu,
            "gpu_temp_c": temp,
            "gpu_memory_used_mb": round(mem_info.used / (1024 ** 2), 2),
            "gpu_sm_clock_mhz": sm_clock,
            "gpu_memory_clock_mhz": mem_clock,
        }
    except Exception:
        return null


def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
            if key in temps:
                values = [e.current for e in temps[key] if e.current and e.current > 0]
                if values:
                    return round(sum(values) / len(values), 1)
    except Exception:
        pass
    return None


def get_cpu_power_draw_w():
    """Stub — populate with platform-specific implementation if available."""
    return None


def get_cpu_cores_used():
    try:
        return sum(1 for p in psutil.cpu_percent(percpu=True) if p > 1.0)
    except Exception:
        return None


# ============================================================
# Prediction quality helpers
# ============================================================

def get_prediction_quality(logits):
    """Return (confidence, logit_margin, entropy) from a raw logits tensor."""
    probs = F.softmax(logits, dim=-1).squeeze()
    confidence = float(probs.max().item())
    top2 = torch.topk(logits.squeeze(), k=2).values
    margin = float((top2[0] - top2[1]).item())
    entropy = float(-(probs * torch.log(probs + 1e-12)).sum().item())
    return round(confidence, 6), round(margin, 6), round(entropy, 6)


# ============================================================
# FLOPs helper
# ============================================================

def compute_model_flops(model, device, input_shape=(1, 1, 28, 28)):
    if not FVCORE_AVAILABLE:
        return None
    try:
        dummy = torch.ones(input_shape, dtype=torch.float32, device=device)
        fc = FlopCountAnalysis(model, dummy)
        fc.unsupported_ops_warnings(False)
        fc.uncalled_modules_warnings(False)
        return int(fc.total())
    except Exception as e:
        print(f"[FLOPs unavailable] {e}")
        return None


# ============================================================
# Energy tracking
# ============================================================

def _extract_energy_data(tracker, emissions_value):
    fd = getattr(tracker, "final_emissions_data", None)
    cpu_energy = getattr(fd, "cpu_energy", 0) if fd else 0
    gpu_energy = getattr(fd, "gpu_energy", 0) if fd else 0
    ram_energy = getattr(fd, "ram_energy", 0) if fd else 0
    total_energy = getattr(fd, "energy_consumed", 0) if fd else 0
    carbon_intensity = None
    if emissions_value and total_energy and total_energy > 0:
        carbon_intensity = round(emissions_value / total_energy, 8)
    return cpu_energy, gpu_energy, ram_energy, total_energy, carbon_intensity


def run_with_energy_tracking(inference_fn, *args, output_dir="./test_results/codecarbon", **kwargs):
    os.makedirs(output_dir, exist_ok=True)

    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name="mnist_edge_inference",
            output_dir=output_dir,
            output_file="codecarbon_dnn_cnn_edge.csv",
            log_level="error",
            save_to_file=True,
        )
        tracker.start()
        t0 = time.perf_counter()
        result = inference_fn(*args, **kwargs)
        exec_time = time.perf_counter() - t0
        emissions_value = tracker.stop()
        gpu_snap = get_gpu_metrics()
        cpu_energy, gpu_energy, ram_energy, total_energy, carbon_intensity = \
            _extract_energy_data(tracker, emissions_value)
        return result, exec_time, cpu_energy, gpu_energy, ram_energy, total_energy, emissions_value, carbon_intensity, gpu_snap

    t0 = time.perf_counter()
    result = inference_fn(*args, **kwargs)
    exec_time = time.perf_counter() - t0
    gpu_snap = get_gpu_metrics()
    return result, exec_time, 0, 0, 0, 0, 0, None, gpu_snap


# ============================================================
# Dataset / CSV helpers
# ============================================================

def get_edge_dataset_path(file_name, dataset_name="MNIST"):
    # raw_dir = DEVICE_LOG_DIR 
    # raw_dir.mkdir(parents=True, exist_ok=True)
    return DEVICE_LOG_DIR / f"dnn_cnn_dataset_{DEVICE_SHORT}_{dataset_name.lower()}_{file_name}.csv"


def append_rows(rows, file_path):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if file_path.exists():
        try:
            existing_df = pd.read_csv(file_path, on_bad_lines="skip")
            for col in new_df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = None
            for col in existing_df.columns:
                if col not in new_df.columns:
                    new_df[col] = None
            new_df = new_df[existing_df.columns]
            pd.concat([existing_df, new_df], ignore_index=True).to_csv(file_path, index=False)
            return
        except Exception:
            pass
    new_df.to_csv(file_path, index=False)


def get_existing_count(file_path, model_name):
    if not file_path.exists():
        return 0
    try:
        df = pd.read_csv(file_path, on_bad_lines="skip")
        if "model_type" not in df.columns:
            return 0
        if "collection_mode" in df.columns:
            mask = (
                (df["model_type"].astype(str).str.strip() == model_name)
                & (df["collection_mode"].astype(str).str.strip() == "automated_edge")
            )
            return int(mask.sum())
        return int((df["model_type"].astype(str).str.strip() == model_name).sum())
    except Exception:
        return 0


# ============================================================
# Inference runners (unified)
# ============================================================

def _run_torch_model(model, image_tensor, mean=(0.1307,), std=(0.3081,)):
    """Shared inference path for CNN and DNN. `mean`/`std` are the
    dataset-specific normalization stats (MNIST, FashionMNIST, KMNIST each
    have their own)."""
    preprocess = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    input_tensor = preprocess(image_tensor).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(input_tensor)
        pred = torch.argmax(logits, 1).item()
    return pred, logits


def tensor_to_canvas_array(image_tensor):
    img = image_tensor.squeeze(0).cpu().numpy() * 255.0
    return img.clip(0, 255).astype("uint8")


# ============================================================
# Row builder
# ============================================================
#
# Column schema is kept aligned with the MNISQ quantum-kernel test script's
# build_row(), so classical (CNN/DNN) and quantum telemetry CSVs share the
# same shape and can be concatenated / compared directly:
#
#   quantum script field   -> classical equivalent here
#   ---------------------------------------------------
#   sample_id               -> sample_id (mirrors sample_index; torchvision
#                               datasets don't expose a separate id)
#   qasm_path                -> N/A for classical models -> None
#   svm_model_path            -> checkpoint_path (the loaded .pth file)
#   train_states_path         -> N/A for classical models -> None
#   model_under_attack         -> model_under_attack (0 = clean, 1 = adversarial;
#                                 set via the `under_attack` argument, useful
#                                 once FGSM/PGD-perturbed inputs are added)
#   source_qubits, n_qubits,
#   fidelity, pennylane_device,
#   pennylane_version          -> quantum-only fields, kept as None here for
#                                 schema parity so both CSVs have identical
#                                 columns even though these don't apply
#                                 to a classical CNN/DNN.

def build_row(
    model_name,
    prediction,
    exec_time,
    parameters,
    true_label,
    sample_index,
    logits=None,
    cpu_energy=0,
    gpu_energy=0,
    ram_energy=0,
    total_energy=0,
    emissions_value=0,
    carbon_intensity=None,
    gpu_metrics=None,
    model_flops=None,
    quantum_computing=False,
    dataset_name="MNIST",
    sample_id=None,
    checkpoint_path=None,
    under_attack=False,
):
    gpu_metrics = gpu_metrics or {}

    # Prediction quality
    confidence_score, logit_margin, entropy = (None, None, None)
    if logits is not None:
        confidence_score, logit_margin, entropy = get_prediction_quality(logits)

    # Energy-derived efficiency metrics
    total_energy = total_energy or 0.0
    cpu_energy = cpu_energy or 0.0
    gpu_energy = gpu_energy or 0.0
    ram_energy = ram_energy or 0.0

    input_tokens = 784  # 28x28 pixel proxy
    output_tokens = 1
    total_tokens = input_tokens + output_tokens

    joules_per_token = 0.0
    energy_per_token_kwh = 0.0
    watts_estimated = 0.0
    gpu_energy_pct = 0.0
    cpu_energy_pct = 0.0

    if total_energy > 0 and total_tokens > 0:
        energy_per_token_kwh = round(total_energy / total_tokens, 12)
        joules_per_token = round((total_energy * 3_600_000) / total_tokens, 6)
        if exec_time > 0:
            watts_estimated = round((total_energy * 3_600_000) / exec_time, 4)
        gpu_energy_pct = round((gpu_energy / total_energy) * 100, 2)
        cpu_energy_pct = round((cpu_energy / total_energy) * 100, 2)

    correct = None
    if true_label is not None and prediction is not None:
        correct = int(prediction) == int(true_label)

    return {
        # --- Identity ---
        "timestamp":                    time.strftime("%Y-%m-%d %H:%M:%S"),
        "unique_device_id":             DEVICE_UUID,
        "device_short_id":              DEVICE_SHORT,
        "pc_name":                      get_hostname(),
        "collection_mode":              "automated_edge",

        # --- Sample ---
        "sample_index":                 sample_index,
        "sample_id":                    sample_id if sample_id is not None else sample_index,
        "true_label":                   true_label,
        "prediction":                   prediction,
        "correct":                      correct,

        # --- Model identity ---
        "dataset":                      dataset_name,
        "model_type":                   model_name,
        "parameters":                   parameters,
        "model_flops":                  model_flops,
        "checkpoint_path":              str(checkpoint_path) if checkpoint_path is not None else None,

        # --- Prediction quality ---
        "confidence_score":             confidence_score,
        "logit_margin":                 logit_margin,
        "entropy":                      entropy,

        # --- Timing ---
        "execution_time_sec":           round(exec_time, 10),

        # --- CodeCarbon energy ---
        "cpu_energy_kwh":               cpu_energy,
        "gpu_energy_kwh":               gpu_energy,
        "ram_energy_kwh":               ram_energy,
        "total_energy_kwh":             total_energy,
        "total_emissions_kg":           emissions_value,
        "carbon_intensity_kgco2_kwh":   carbon_intensity,
        "codecarbon_version":           CODECARBON_VERSION,

        # --- Efficiency derived ---
        "input_tokens":                 input_tokens,
        "output_tokens":                output_tokens,
        "total_tokens":                 total_tokens,
        "tokens_per_second":            round(total_tokens / exec_time, 4) if exec_time > 0 else None,
        "joules_per_token":             joules_per_token,
        "energy_per_token_kwh":         energy_per_token_kwh,
        "watts_estimated":              watts_estimated,
        "gpu_energy_pct_of_total":      gpu_energy_pct,
        "cpu_energy_pct_of_total":      cpu_energy_pct,

        # --- CPU hardware ---
        "cpu_model":                    CPU_MODEL_NAME,
        "cpu_architecture":             CPU_ARCH,
        "cpu_core_count":               CPU_CORE_COUNT,
        "cpu_thread_count":             CPU_THREAD_COUNT,
        "cpu_core":                     CPU_CORE_COUNT,
        "cpu_thread":                   CPU_THREAD_COUNT,
        "cpu_tdp_w":                    CPU_TDP_W,
        "cpu_usage_pct":                get_cpu_usage(),
        "cpu_clock_mhz":                get_cpu_freq(),
        "cpu_temp_c":                   get_cpu_temp(),
        "cpu_power_draw_w":             get_cpu_power_draw_w(),
        "cpu_cores_used":               get_cpu_cores_used(),

        # --- GPU hardware ---
        "gpu_model":                    get_gpu_name(),
        "gpu_core":                     GPU_CORE_COUNT,
        "gpu_thread":                   GPU_THREAD_COUNT,
        "gpu_driver_version":           GPU_STATIC["gpu_driver_version"],
        "gpu_compute_capability":       GPU_STATIC["gpu_compute_capability"],
        "gpu_power_limit_w":            GPU_STATIC["gpu_power_limit_w"],
        "gpu_memory_total_mb":          GPU_STATIC["gpu_memory_total_mb"],
        "gpu_power_draw_w":             gpu_metrics.get("gpu_power_draw_w"),
        "gpu_utilization_pct":          gpu_metrics.get("gpu_utilization_pct"),
        "gpu_temp_c":                   gpu_metrics.get("gpu_temp_c"),
        "gpu_memory_used_mb":           gpu_metrics.get("gpu_memory_used_mb"),
        "gpu_sm_clock_mhz":             gpu_metrics.get("gpu_sm_clock_mhz"),
        "gpu_memory_clock_mhz":         gpu_metrics.get("gpu_memory_clock_mhz"),
        "cuda_driver_version":          CUDA_DRIVER_VERSION,
        "cuda_available":               torch.cuda.is_available(),
        "device_type":                  str(DEVICE),

        # --- RAM / memory ---
        "ram_usage_pct":                psutil.virtual_memory().percent,
        "memory_footprint_mb":          get_memory_footprint_mb(),
        "system_ram_total_gb":          SYSTEM_RAM_TOTAL_GB,

        # --- Environment ---
        "os_name":                      OS_NAME,
        "os_version":                   OS_VERSION,
        "os_architecture":              OS_ARCHITECTURE,
        "os_full_name":                 OS_FULL_NAME,
        "python_version":               PYTHON_VERSION,
        "torch_version":                TORCH_VERSION,

        # --- Final model metrics (backfilled after run) ---
        "model_accuracy":               None,
        "model_precision_weighted":     None,
        "model_recall_weighted":        None,
        "model_f1_weighted":            None,

        # --- Custom / attack context (mirrors quantum script's fields) ---
        "quantum_computing":            quantum_computing,
        "model_under_attack":           int(bool(under_attack)),

        # --- Quantum-only fields, kept as None here for schema parity with
        #     the MNISQ quantum-kernel telemetry CSV (not applicable to a
        #     classical CNN/DNN, but present so both CSVs share columns) ---
        "source_qubits":                None,
        "n_qubits":                     None,
        "fidelity":                     None,
        "pennylane_device":             None,
        "pennylane_version":            None,
    }



def backfill_model_metrics(file_path, model_name):
    """
    Compute model metrics from the prediction rows already saved in the
    telemetry CSV, then write those metrics back into the same CSV.
    No separate model_metrics.json file is used.
    """
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path, on_bad_lines="skip")

    if "model_type" not in df.columns:
        return None

    mask = (
        df["model_type"]
        .astype(str)
        .str.strip()
        == model_name
    )

    model_df = df.loc[mask].copy()

    if model_df.empty:
        return None

    # Remove incomplete rows, if any.
    model_df = model_df.dropna(
        subset=["true_label", "prediction"]
    )

    if model_df.empty:
        return None

    y_true = model_df["true_label"].astype(int).tolist()
    y_pred = model_df["prediction"].astype(int).tolist()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # Backfill all rows belonging to this model in the SAME CSV.
    df.loc[
        mask,
        "model_accuracy",
    ] = float(accuracy)

    df.loc[
        mask,
        "model_precision_weighted",
    ] = float(precision_weighted)

    df.loc[
        mask,
        "model_recall_weighted",
    ] = float(recall_weighted)

    df.loc[
        mask,
        "model_f1_weighted",
    ] = float(f1_weighted)

    df.to_csv(
        file_path,
        index=False,
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }

    print(
        f"{model_name} metrics -> "
        f"Accuracy: {accuracy:.4f}, "
        f"Precision(weighted): {precision_weighted:.4f}, "
        f"Recall(weighted): {recall_weighted:.4f}, "
        f"F1(weighted): {f1_weighted:.4f}"
    )

    return metrics


# ============================================================
# Main collection loop
# ============================================================

def collect_for_model(base_dataset, model_name, dataset_name="MNIST", mean=(0.1307,), std=(0.3081,), num_samples=250, flush_every=25, file_name ='vanilla'):
    print(f"\nCollecting {num_samples} edge samples for {dataset_name} / {model_name} on {get_hostname()}")
    print(f"Device UUID : {DEVICE_UUID}")
    print(f"Device short: {DEVICE_SHORT}")

    output_path = get_edge_dataset_path(file_name, dataset_name=dataset_name)

    rows = []

    checkpoint_prefix = DATASET_CONFIGS[dataset_name]["checkpoint_prefix"]

    if model_name == "CNN":
        checkpoint_path = f"{checkpoint_prefix}_cnn.pth"
        model = SimpleCNN().to(DEVICE)
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()
        model_flops = compute_model_flops(model, DEVICE)

    elif model_name == "DNN":
        checkpoint_path = f"{checkpoint_prefix}_dnn.pth"
        model = SimpleDNN().to(DEVICE)
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()
        model_flops = compute_model_flops(model, DEVICE)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Compute parameter count directly from the loaded model.
    # No separate model-metrics file is needed.
    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    limit = min(num_samples, len(base_dataset))
    existing_count = get_existing_count(output_path, model_name)

    if existing_count >= limit:
        print(f"{model_name}: already complete ({existing_count}/{limit})")
        backfill_model_metrics(
            output_path,
            model_name,
        )
        return

    print(f"{model_name}: resuming from {existing_count}/{limit}")

    progress_bar = tqdm(
        range(existing_count, limit),
        desc=model_name,
        unit="sample",
        dynamic_ncols=True,
    )

    for i in progress_bar:
        image_tensor, true_label = base_dataset[i]

        logits = None

        if model_name in ("CNN", "DNN"):
            (pred, logits), exec_time, cpu_energy, gpu_energy, ram_energy, \
                total_energy, emissions_value, carbon_intensity, gpu_snap = \
                run_with_energy_tracking(
                    _run_torch_model, model, image_tensor, mean=mean, std=std
                )

        row = build_row(
            model_name=model_name,
            prediction=pred,
            exec_time=exec_time,
            parameters=parameters,
            true_label=int(true_label),
            sample_index=i,
            logits=logits,
            cpu_energy=cpu_energy,
            gpu_energy=gpu_energy,
            ram_energy=ram_energy,
            total_energy=total_energy,
            emissions_value=emissions_value,
            carbon_intensity=carbon_intensity,
            gpu_metrics=gpu_snap,
            model_flops=model_flops,
            quantum_computing = file_name != 'vanilla',
            dataset_name=dataset_name,
            sample_id=i,
            checkpoint_path=checkpoint_path,
            under_attack=False,
        )

        rows.append(row)

        progress_bar.set_postfix({
            "pred": pred,
            "true": int(true_label),
            "time_s": round(exec_time, 3),
            "done": i + 1,
        })

        if (i + 1) % flush_every == 0:
            append_rows(rows, output_path)
            rows = []

    if rows:
        append_rows(rows, output_path)

    # Compute Accuracy / Precision / Recall / F1 from the saved predictions
    # and write them back into the SAME telemetry CSV.
    backfill_model_metrics(
        output_path,
        model_name,
    )

    print(f"{model_name}: finished {limit}/{limit} -> {output_path}")


def dnn_cnn_main():

    # Loop over all three datasets (MNIST, FashionMNIST, KMNIST) and both
    # architectures (CNN, DNN). Each combination loads its own checkpoint
    # (e.g. "fashionmnist_cnn.pth") and writes to its own CSV, so results
    # never overwrite or mix across datasets.

    for dataset_name, config in DATASET_CONFIGS.items():

        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        # Raw ToTensor only here — normalization is applied per-sample
        # inside _run_torch_model using this dataset's own mean/std.
        base_dataset = config["dataset_class"](
            root="./classical_data",
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )

        for model_name in ("CNN", "DNN"):
            collect_for_model(
                base_dataset,
                model_name,
                dataset_name=dataset_name,
                mean=config["mean"],
                std=config["std"],
                num_samples=NUM_TEST_SAMPLES,
                flush_every=25,
                file_name='vanilla'
            )

    print("\nDone.")


# -*- coding: utf-8 -*-
"""
TinyCLIP zero-shot classification + telemetry
Datasets: MNIST, FashionMNIST, KMNIST

Model:
    wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M

The CSV schema is kept identical to the existing 83-column
CNN/DNN/YOLO/MobileNet/Qwen/Moondream telemetry schema.
"""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import sys
import time
from pathlib import Path

import pandas as pd
import psutil
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from torchvision import datasets
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotImageClassification,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
MODEL_NAME = "TinyCLIP-ViT-8M-16-Text-3M"

NUM_TEST_SAMPLES = int(
    os.getenv("NUM_TEST_SAMPLES", "10000")
)

DEVICE_MODE = os.getenv("DEVICE_MODE", "cpu").lower()

if DEVICE_MODE == "cuda":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        print("CUDA unavailable. Falling back to CPU.")
        DEVICE = torch.device("cpu")
elif DEVICE_MODE == "auto":
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
else:
    DEVICE = torch.device("cpu")

DATA_ROOT = Path("./classical_data")
OUTPUT_ROOT = Path.cwd() / "test_results"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASETS + SEMANTIC CLASS PROMPTS
# ============================================================

DATASET_CONFIGS = {
    "MNIST": {
        "dataset_class": datasets.MNIST,
        "key": "mnist",
        "class_prompts": [
            "a handwritten digit zero",
            "a handwritten digit one",
            "a handwritten digit two",
            "a handwritten digit three",
            "a handwritten digit four",
            "a handwritten digit five",
            "a handwritten digit six",
            "a handwritten digit seven",
            "a handwritten digit eight",
            "a handwritten digit nine",
        ],
    },
    "FashionMNIST": {
        "dataset_class": datasets.FashionMNIST,
        "key": "fashionmnist",
        "class_prompts": [
            "a photo of a T-shirt or top",
            "a photo of trousers",
            "a photo of a pullover",
            "a photo of a dress",
            "a photo of a coat",
            "a photo of a sandal",
            "a photo of a shirt",
            "a photo of a sneaker",
            "a photo of a bag",
            "a photo of an ankle boot",
        ],
    },
    "KMNIST": {
        "dataset_class": datasets.KMNIST,
        "key": "kmnist",
        "class_prompts": [
            "a handwritten Japanese Kuzushiji character o",
            "a handwritten Japanese Kuzushiji character ki",
            "a handwritten Japanese Kuzushiji character su",
            "a handwritten Japanese Kuzushiji character tsu",
            "a handwritten Japanese Kuzushiji character na",
            "a handwritten Japanese Kuzushiji character ha",
            "a handwritten Japanese Kuzushiji character ma",
            "a handwritten Japanese Kuzushiji character ya",
            "a handwritten Japanese Kuzushiji character re",
            "a handwritten Japanese Kuzushiji character wo",
        ],
    },
}


# ============================================================
# OPTIONAL TELEMETRY PACKAGES
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
    print("CodeCarbon unavailable. Energy values will be 0.")

try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    NVML_HANDLE = (
        pynvml.nvmlDeviceGetHandleByIndex(0)
        if torch.cuda.is_available()
        else None
    )
except Exception:
    NVML_AVAILABLE = False
    NVML_HANDLE = None
    print("pynvml unavailable.")

try:
    import cpuinfo

    _CPU_INFO = cpuinfo.get_cpu_info()
    CPU_MODEL_NAME = _CPU_INFO.get("brand_raw", "Unknown")
    CPU_ARCH = _CPU_INFO.get("arch", platform.machine())
except Exception:
    CPU_MODEL_NAME = platform.processor() or "Unknown"
    CPU_ARCH = platform.machine()

CPU_TDP_W = None


# ============================================================
# SYSTEM / DEVICE METADATA
# ============================================================

TORCH_VERSION = torch.__version__
PYTHON_VERSION = sys.version.split()[0]
OS_NAME = platform.system()
OS_VERSION = platform.version()
OS_ARCHITECTURE = platform.machine()
SYSTEM_RAM_TOTAL_GB = round(
    psutil.virtual_memory().total / (1024 ** 3), 2
)
CPU_CORE_COUNT = psutil.cpu_count(logical=False)
CPU_THREAD_COUNT = psutil.cpu_count(logical=True)


def get_os_full_name():
    system = platform.system()
    architecture = platform.machine()

    if system == "Windows":
        return f"Windows {platform.release()} {architecture}"

    if system == "Linux":
        try:
            info = {}
            with open("/etc/os-release", "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        info[k] = v.strip('"')
            return f"{info.get('PRETTY_NAME', 'Linux')} {architecture}"
        except Exception:
            return f"Linux {platform.release()} {architecture}"

    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]} {architecture}"

    return f"{system} {platform.release()} {architecture}"


OS_FULL_NAME = get_os_full_name()


def make_stable_device_id():
    raw = (
        f"{socket.gethostname()}-"
        f"{platform.system()}-"
        f"{platform.machine()}-"
        f"{CPU_MODEL_NAME}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


DEVICE_UUID = make_stable_device_id()
DEVICE_SHORT = DEVICE_UUID[:8]

DEVICE_LOG_DIR = OUTPUT_ROOT / DEVICE_SHORT
DEVICE_LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_hostname():
    return socket.gethostname()


def get_cuda_driver_version():
    if not NVML_AVAILABLE:
        return None
    try:
        value = pynvml.nvmlSystemGetDriverVersion()
        return value.decode("utf-8") if isinstance(value, bytes) else value
    except Exception:
        return None


CUDA_DRIVER_VERSION = get_cuda_driver_version()


def get_gpu_static():
    result = {
        "gpu_driver_version": CUDA_DRIVER_VERSION,
        "gpu_compute_capability": None,
        "gpu_power_limit_w": None,
        "gpu_memory_total_mb": None,
    }

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            result["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        except Exception:
            pass

    if not NVML_AVAILABLE or NVML_HANDLE is None:
        return result

    try:
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(NVML_HANDLE)
        memory = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)
        result["gpu_power_limit_w"] = round(power_limit / 1000.0, 2)
        result["gpu_memory_total_mb"] = round(
            memory.total / (1024 ** 2), 2
        )
    except Exception:
        pass

    return result


GPU_STATIC = get_gpu_static()


def get_gpu_core_thread():
    if not torch.cuda.is_available():
        return None, None

    try:
        props = torch.cuda.get_device_properties(0)
        sm_count = props.multi_processor_count
        cores_per_sm = {
            2: 32,
            3: 192,
            5: 128,
            6: 64,
            7: 64,
            8: 128,
            9: 128,
        }.get(props.major, 64)

        return (
            sm_count * cores_per_sm,
            sm_count * props.max_threads_per_multi_processor,
        )
    except Exception:
        return None, None


GPU_CORE_COUNT, GPU_THREAD_COUNT = get_gpu_core_thread()


def get_gpu_name():
    if not torch.cuda.is_available():
        return "No GPU"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "Unknown"


def get_cpu_usage():
    try:
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None


def get_cpu_freq():
    try:
        freq = psutil.cpu_freq()
        return round(freq.current, 2) if freq else None
    except Exception:
        return None


def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None

        for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
            if key in temps:
                vals = [
                    x.current
                    for x in temps[key]
                    if x.current is not None and x.current > 0
                ]
                if vals:
                    return round(sum(vals) / len(vals), 1)
    except Exception:
        pass

    return None


def get_cpu_power_draw_w():
    return None


def get_cpu_cores_used():
    try:
        return sum(
            1
            for x in psutil.cpu_percent(percpu=True)
            if x > 1.0
        )
    except Exception:
        return None


def get_memory_footprint_mb():
    try:
        return round(
            psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2),
            4,
        )
    except Exception:
        return None


def get_gpu_metrics():
    result = {
        "gpu_power_draw_w": None,
        "gpu_utilization_pct": None,
        "gpu_temp_c": None,
        "gpu_memory_used_mb": None,
        "gpu_sm_clock_mhz": None,
        "gpu_memory_clock_mhz": None,
    }

    if not NVML_AVAILABLE or NVML_HANDLE is None:
        return result

    try:
        power = pynvml.nvmlDeviceGetPowerUsage(NVML_HANDLE)
        util = pynvml.nvmlDeviceGetUtilizationRates(NVML_HANDLE)
        temp = pynvml.nvmlDeviceGetTemperature(
            NVML_HANDLE,
            pynvml.NVML_TEMPERATURE_GPU,
        )
        mem = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)
        sm_clock = pynvml.nvmlDeviceGetClockInfo(
            NVML_HANDLE,
            pynvml.NVML_CLOCK_SM,
        )
        mem_clock = pynvml.nvmlDeviceGetClockInfo(
            NVML_HANDLE,
            pynvml.NVML_CLOCK_MEM,
        )

        return {
            "gpu_power_draw_w": round(power / 1000.0, 2),
            "gpu_utilization_pct": util.gpu,
            "gpu_temp_c": temp,
            "gpu_memory_used_mb": round(mem.used / (1024 ** 2), 2),
            "gpu_sm_clock_mhz": sm_clock,
            "gpu_memory_clock_mhz": mem_clock,
        }
    except Exception:
        return result


# ============================================================
# EXACT 83-COLUMN REFERENCE SCHEMA
# ============================================================

REFERENCE_COLUMNS = [
    "timestamp",
    "unique_device_id",
    "device_short_id",
    "pc_name",
    "collection_mode",
    "sample_index",
    "sample_id",
    "true_label",
    "prediction",
    "correct",
    "dataset",
    "model_type",
    "parameters",
    "model_flops",
    "checkpoint_path",
    "confidence_score",
    "logit_margin",
    "entropy",
    "execution_time_sec",
    "cpu_energy_kwh",
    "gpu_energy_kwh",
    "ram_energy_kwh",
    "total_energy_kwh",
    "total_emissions_kg",
    "carbon_intensity_kgco2_kwh",
    "codecarbon_version",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "tokens_per_second",
    "joules_per_token",
    "energy_per_token_kwh",
    "watts_estimated",
    "gpu_energy_pct_of_total",
    "cpu_energy_pct_of_total",
    "cpu_model",
    "cpu_architecture",
    "cpu_core_count",
    "cpu_thread_count",
    "cpu_core",
    "cpu_thread",
    "cpu_tdp_w",
    "cpu_usage_pct",
    "cpu_clock_mhz",
    "cpu_temp_c",
    "cpu_power_draw_w",
    "cpu_cores_used",
    "gpu_model",
    "gpu_core",
    "gpu_thread",
    "gpu_driver_version",
    "gpu_compute_capability",
    "gpu_power_limit_w",
    "gpu_memory_total_mb",
    "gpu_power_draw_w",
    "gpu_utilization_pct",
    "gpu_temp_c",
    "gpu_memory_used_mb",
    "gpu_sm_clock_mhz",
    "gpu_memory_clock_mhz",
    "cuda_driver_version",
    "cuda_available",
    "device_type",
    "ram_usage_pct",
    "memory_footprint_mb",
    "system_ram_total_gb",
    "os_name",
    "os_version",
    "os_architecture",
    "os_full_name",
    "python_version",
    "torch_version",
    "model_accuracy",
    "model_precision_weighted",
    "model_recall_weighted",
    "model_f1_weighted",
    "quantum_computing",
    "model_under_attack",
    "source_qubits",
    "n_qubits",
    "fidelity",
    "pennylane_device",
    "pennylane_version",
]


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading TinyCLIP...")
print(MODEL_ID)

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = (
    AutoModelForZeroShotImageClassification
    .from_pretrained(MODEL_ID)
    .to(DEVICE)
)

model.eval()

MODEL_PARAMETERS = int(
    sum(p.numel() for p in model.parameters())
)

# Leave None rather than report an incomplete image-only/text-only FLOPs value.
MODEL_FLOPS = None

print("Device:", DEVICE)
print("Parameters:", f"{MODEL_PARAMETERS:,}")


# ============================================================
# ENERGY TRACKING
# ============================================================

def run_with_energy_tracking(
    inference_fn,
    image,
    class_prompts,
    dataset_key,
):
    energy_dir = OUTPUT_ROOT / "codecarbon"
    energy_dir.mkdir(parents=True, exist_ok=True)

    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name=f"tinyclip_{dataset_key}_test",
            output_dir=str(energy_dir),
            output_file=f"codecarbon_tinyclip_{dataset_key}.csv",
            log_level="error",
            save_to_file=True,
        )

        tracker.start()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        result = inference_fn(
            image,
            class_prompts,
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        exec_time = time.perf_counter() - t0

        emissions_value = tracker.stop()
        fd = getattr(
            tracker,
            "final_emissions_data",
            None,
        )

        cpu_energy = float(
            getattr(fd, "cpu_energy", 0) or 0
        )
        gpu_energy = float(
            getattr(fd, "gpu_energy", 0) or 0
        )
        ram_energy = float(
            getattr(fd, "ram_energy", 0) or 0
        )
        total_energy = float(
            getattr(fd, "energy_consumed", 0) or 0
        )

        carbon_intensity = (
            float(emissions_value) / total_energy
            if emissions_value is not None and total_energy > 0
            else None
        )

    else:
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        result = inference_fn(
            image,
            class_prompts,
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        exec_time = time.perf_counter() - t0

        cpu_energy = 0.0
        gpu_energy = 0.0
        ram_energy = 0.0
        total_energy = 0.0
        emissions_value = 0.0
        carbon_intensity = None

    return (
        result,
        exec_time,
        cpu_energy,
        gpu_energy,
        ram_energy,
        total_energy,
        emissions_value,
        carbon_intensity,
        get_gpu_metrics(),
    )


# ============================================================
# TINYCLIP ZERO-SHOT INFERENCE
# ============================================================

def run_tinyclip_inference(
    image,
    class_prompts,
):
    image = image.convert("RGB")

    inputs = processor(
        text=class_prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: (
            value.to(DEVICE)
            if torch.is_tensor(value)
            else value
        )
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(
            **inputs
        )

    logits = outputs.logits_per_image[0].float()

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    prediction = int(
        torch.argmax(probabilities).item()
    )

    confidence_score = float(
        probabilities[prediction].item()
    )

    top2_logits = torch.topk(
        logits,
        k=2,
    ).values

    logit_margin = float(
        (top2_logits[0] - top2_logits[1]).item()
    )

    entropy = float(
        -(
            probabilities
            * torch.log(probabilities + 1e-12)
        ).sum().item()
    )

    # Keep same proxy as prior CNN/DNN/YOLO/MobileNet telemetry.
    input_tokens = 784
    output_tokens = 1

    return {
        "prediction": prediction,
        "confidence_score": confidence_score,
        "logit_margin": logit_margin,
        "entropy": entropy,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# ============================================================
# TELEMETRY ROW
# ============================================================

def build_row(
    *,
    dataset_name,
    sample_index,
    true_label,
    result,
    exec_time,
    cpu_energy,
    gpu_energy,
    ram_energy,
    total_energy,
    emissions_value,
    carbon_intensity,
    gpu_metrics,
):
    prediction = int(result["prediction"])

    input_tokens = int(result["input_tokens"])
    output_tokens = int(result["output_tokens"])
    total_tokens = input_tokens + output_tokens

    cpu_energy = float(cpu_energy or 0.0)
    gpu_energy = float(gpu_energy or 0.0)
    ram_energy = float(ram_energy or 0.0)
    total_energy = float(total_energy or 0.0)

    energy_per_token_kwh = 0.0
    joules_per_token = 0.0
    watts_estimated = 0.0
    gpu_energy_pct = 0.0
    cpu_energy_pct = 0.0

    if total_energy > 0 and total_tokens > 0:
        energy_per_token_kwh = round(
            total_energy / total_tokens,
            12,
        )
        joules_per_token = round(
            total_energy * 3_600_000 / total_tokens,
            6,
        )

        if exec_time > 0:
            watts_estimated = round(
                total_energy * 3_600_000 / exec_time,
                4,
            )

        gpu_energy_pct = round(
            gpu_energy / total_energy * 100,
            2,
        )
        cpu_energy_pct = round(
            cpu_energy / total_energy * 100,
            2,
        )

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unique_device_id": DEVICE_UUID,
        "device_short_id": DEVICE_SHORT,
        "pc_name": get_hostname(),
        "collection_mode": "automated_edge",

        "sample_index": sample_index,
        "sample_id": sample_index,
        "true_label": int(true_label),
        "prediction": prediction,
        "correct": prediction == int(true_label),

        "dataset": dataset_name,
        "model_type": MODEL_NAME,
        "parameters": MODEL_PARAMETERS,
        "model_flops": MODEL_FLOPS,
        "checkpoint_path": MODEL_ID,

        "confidence_score": round(
            float(result["confidence_score"]), 6
        ),
        "logit_margin": round(
            float(result["logit_margin"]), 6
        ),
        "entropy": round(
            float(result["entropy"]), 6
        ),

        "execution_time_sec": round(exec_time, 10),

        "cpu_energy_kwh": cpu_energy,
        "gpu_energy_kwh": gpu_energy,
        "ram_energy_kwh": ram_energy,
        "total_energy_kwh": total_energy,
        "total_emissions_kg": emissions_value,
        "carbon_intensity_kgco2_kwh": carbon_intensity,
        "codecarbon_version": CODECARBON_VERSION,

        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": (
            round(total_tokens / exec_time, 4)
            if exec_time > 0
            else None
        ),
        "joules_per_token": joules_per_token,
        "energy_per_token_kwh": energy_per_token_kwh,
        "watts_estimated": watts_estimated,
        "gpu_energy_pct_of_total": gpu_energy_pct,
        "cpu_energy_pct_of_total": cpu_energy_pct,

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

        "gpu_model": get_gpu_name(),
        "gpu_core": GPU_CORE_COUNT,
        "gpu_thread": GPU_THREAD_COUNT,
        "gpu_driver_version": GPU_STATIC["gpu_driver_version"],
        "gpu_compute_capability": GPU_STATIC[
            "gpu_compute_capability"
        ],
        "gpu_power_limit_w": GPU_STATIC["gpu_power_limit_w"],
        "gpu_memory_total_mb": GPU_STATIC["gpu_memory_total_mb"],
        "gpu_power_draw_w": gpu_metrics.get("gpu_power_draw_w"),
        "gpu_utilization_pct": gpu_metrics.get(
            "gpu_utilization_pct"
        ),
        "gpu_temp_c": gpu_metrics.get("gpu_temp_c"),
        "gpu_memory_used_mb": gpu_metrics.get(
            "gpu_memory_used_mb"
        ),
        "gpu_sm_clock_mhz": gpu_metrics.get("gpu_sm_clock_mhz"),
        "gpu_memory_clock_mhz": gpu_metrics.get(
            "gpu_memory_clock_mhz"
        ),
        "cuda_driver_version": CUDA_DRIVER_VERSION,
        "cuda_available": torch.cuda.is_available(),
        "device_type": str(DEVICE),

        "ram_usage_pct": psutil.virtual_memory().percent,
        "memory_footprint_mb": get_memory_footprint_mb(),
        "system_ram_total_gb": SYSTEM_RAM_TOTAL_GB,

        "os_name": OS_NAME,
        "os_version": OS_VERSION,
        "os_architecture": OS_ARCHITECTURE,
        "os_full_name": OS_FULL_NAME,
        "python_version": PYTHON_VERSION,
        "torch_version": TORCH_VERSION,

        "model_accuracy": None,
        "model_precision_weighted": None,
        "model_recall_weighted": None,
        "model_f1_weighted": None,

        "quantum_computing": False,
        "model_under_attack": 0,

        "source_qubits": None,
        "n_qubits": None,
        "fidelity": None,
        "pennylane_device": None,
        "pennylane_version": None,
    }

    assert list(row.keys()) == REFERENCE_COLUMNS

    return row


# ============================================================
# METRICS + SAVE
# ============================================================

def finalize_and_save(
    rows,
    output_path,
):
    df = pd.DataFrame(rows).reindex(
        columns=REFERENCE_COLUMNS
    )

    y_true = df["true_label"].astype(int).to_numpy()
    y_pred = df["prediction"].astype(int).to_numpy()

    accuracy = float(
        accuracy_score(y_true, y_pred)
    )
    precision = float(
        precision_score(
            y_true,
            y_pred,
            labels=list(range(10)),
            average="weighted",
            zero_division=0,
        )
    )
    recall = float(
        recall_score(
            y_true,
            y_pred,
            labels=list(range(10)),
            average="weighted",
            zero_division=0,
        )
    )
    f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(10)),
            average="weighted",
            zero_division=0,
        )
    )

    df["model_accuracy"] = accuracy
    df["model_precision_weighted"] = precision
    df["model_recall_weighted"] = recall
    df["model_f1_weighted"] = f1

    df = df.reindex(
        columns=REFERENCE_COLUMNS
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved: {output_path.resolve()}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    return {
        "model": MODEL_NAME,
        "dataset": str(df["dataset"].iloc[0]),
        "samples": len(df),
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }


# ============================================================
# TEST ONE DATASET
# ============================================================

def test_dataset(
    dataset_name,
    config,
):
    dataset = config["dataset_class"](
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=None,
    )

    limit = min(
        NUM_TEST_SAMPLES,
        len(dataset),
    )

    class_prompts = config[
        "class_prompts"
    ]

    print("\n" + "=" * 72)
    print(f"TINYCLIP TEST: {dataset_name}")
    print("=" * 72)
    print("Samples:", limit)

    for i, text in enumerate(
        class_prompts
    ):
        print(f"{i}: {text}")

    rows = []

    for sample_index in tqdm(
        range(limit),
        desc=f"TinyCLIP | {dataset_name}",
        unit="sample",
    ):
        image, true_label = dataset[
            sample_index
        ]

        (
            result,
            exec_time,
            cpu_energy,
            gpu_energy,
            ram_energy,
            total_energy,
            emissions_value,
            carbon_intensity,
            gpu_metrics,
        ) = run_with_energy_tracking(
            run_tinyclip_inference,
            image,
            class_prompts,
            config["key"],
        )

        rows.append(
            build_row(
                dataset_name=dataset_name,
                sample_index=sample_index,
                true_label=int(true_label),
                result=result,
                exec_time=exec_time,
                cpu_energy=cpu_energy,
                gpu_energy=gpu_energy,
                ram_energy=ram_energy,
                total_energy=total_energy,
                emissions_value=emissions_value,
                carbon_intensity=carbon_intensity,
                gpu_metrics=gpu_metrics,
            )
        )

    output_path = (
        DEVICE_LOG_DIR
        / f"tinyclip_{config['key']}_test_telemetry.csv"
    )

    return finalize_and_save(
        rows,
        output_path,
    )


# ============================================================
# MAIN
# ============================================================

def tinyclip_main():
    print("=" * 72)
    print("TINYCLIP THREE-DATASET EXACT TELEMETRY TEST")
    print("=" * 72)
    print("Model:", MODEL_ID)
    print("Device:", DEVICE)
    print("Parameters:", MODEL_PARAMETERS)
    # print("Telemetry columns:", len(REFERENCE_COLUMNS))

    # summaries = []

    # for dataset_name, config in DATASET_CONFIGS.items():
    #     summaries.append(
    #         test_dataset(
    #             dataset_name,
    #             config,
    #         )
    #     )

    # summary_path = (
    #     DEVICE_LOG_DIR
    #     / "tinyclip_three_dataset_summary.csv"
    # )

    # pd.DataFrame(
    #     summaries
    # ).to_csv(
    #     summary_path,
    #     index=False,
    # )

    print("\nAll TinyCLIP tests complete.")
    #print("Summary:", summary_path.resolve())



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
import requests
import joblib
import numpy as np
import pandas as pd
import pennylane as qml
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm.auto import tqdm
import zipfile
# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
SOURCE_QUBITS = 10
N_QUBITS = 1
N_CLASSES = 10
FIDELITY = "f90"

# Official MNISQ test-set size per dataset (used only for the diagnostic
# shortfall warning below -- does not affect selection behavior).
EXPECTED_TEST_TOTAL = 10000

# --- Test-set size control -----------------------------------------------
# USE_ALL_TEST_SAMPLES = True:  use every discovered test sample per dataset
#     (dataset test counts are allowed to differ from each other).
# USE_ALL_TEST_SAMPLES = False: use a balanced "10 set" instead — exactly
#     TEST_SAMPLES_PER_CLASS samples from EACH of the N_CLASSES classes,
#     for TEST_SAMPLES_PER_CLASS * N_CLASSES total samples per dataset
#     (10 per class x 10 classes = 100 total, by default).
USE_ALL_TEST_SAMPLES = True
TEST_SAMPLES_PER_CLASS = 200  # 10

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
        "archive_tokens": ["base_test_mnist_784"],
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
        "archive_tokens": ["base_test_fashion-mnist"],
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
        "archive_tokens": ["base_test_kuzushiji-mnist"],
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
    used_fallback = False
    if not qasm_files:
        # Last-resort fallback: all QASM files containing test.
        qasm_files = [p for p in qasm_all if "test" in str(p).lower()]
        used_fallback = True
    if not qasm_files:
        raise FileNotFoundError(f"No test QASM files found under {data_root}")

    # DIAGNOSTIC: makes it visible when strict token matching found nothing
    # and the script fell back to a looser "test" substring match -- this
    # is a common silent cause of a smaller-than-expected discovered count.
    if used_fallback:
        print(
            f"  [diagnostic] Strict archive/dataset token match found 0 files "
            f"under {data_root}; fell back to a loose 'test' substring match "
            f"({len(qasm_files):,} files). Check archive_tokens/dataset_tokens "
            f"for this dataset if the final count looks too low."
        )

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

    # DIAGNOSTIC: this is the actual source of a smaller-than-10,000 count
    # for most cases -- QASM files whose id didn't have a matching label
    # file (or vice versa) are silently excluded above, so surface the
    # number explicitly here instead of just a generic warning.
    print(
        f"  [diagnostic] {data_root.name}: {len(qasm_files):,} test QASM files found, "
        f"{len(label_files):,} test label files found, "
        f"{len(df):,} QASM-label pairs matched, "
        f"{unmatched:,} QASM files unmatched."
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

        # DIAGNOSTIC: explicit shortfall warning against the official
        # MNISQ test-set size, so a gap is never silent.
        if len(selected) < EXPECTED_TEST_TOTAL:
            shortfall = EXPECTED_TEST_TOTAL - len(selected)
            print(
                f"  [diagnostic] {dataset_name}: discovered {len(selected):,} test "
                f"samples, which is {shortfall:,} short of the expected "
                f"{EXPECTED_TEST_TOTAL:,} for the official MNISQ test set. "
                f"See the QASM/label match counts printed above for where "
                f"samples were dropped."
            )
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
    """
    Predict one sample and measure prediction latency only.

    Energy is measured once around the complete dataset inference loop.
    """
    t0 = time.perf_counter()

    prediction_output = classifier.predict(kernel_row)
    pred = int(np.asarray(prediction_output).reshape(-1)[0])

    exec_time = time.perf_counter() - t0
    return pred, exec_time


def start_dataset_energy_tracker(dataset_name):
    """Start CodeCarbon once for the complete dataset inference workload."""
    if not CODECARBON_AVAILABLE:
        print("[CodeCarbon] Not available; energy values will remain 0.")
        return None

    try:
        codecarbon_dir = OUTPUT_ROOT / "codecarbon"
        codecarbon_dir.mkdir(parents=True, exist_ok=True)

        tracker = EmissionsTracker(
            project_name=f"mnisq_{dataset_name}_q1_fingerprinting",
            output_dir=str(codecarbon_dir),
            output_file=f"{dataset_name.lower().replace('-', '_')}_dataset.csv",
            log_level="error",
            save_to_file=True,
        )
        tracker.start()
        return tracker
    except Exception as exc:
        print(f"[CodeCarbon warning] Could not start tracker: {exc}")
        return None


def stop_dataset_energy_tracker(tracker):
    """Stop CodeCarbon and return dataset-level energy totals."""
    if tracker is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, None

    try:
        emissions = tracker.stop()
        fd = getattr(tracker, "final_emissions_data", None)

        if fd is None:
            print("[CodeCarbon warning] final_emissions_data is unavailable.")
            return 0.0, 0.0, 0.0, 0.0, float(emissions or 0.0), None

        cpu_e = float(getattr(fd, "cpu_energy", 0) or 0)
        gpu_e = float(getattr(fd, "gpu_energy", 0) or 0)
        ram_e = float(getattr(fd, "ram_energy", 0) or 0)
        total_e = float(getattr(fd, "energy_consumed", 0) or 0)
        emissions_val = float(emissions or 0)

        carbon_intensity = None
        if total_e > 0:
            carbon_intensity = emissions_val / total_e

        print("\nCodeCarbon dataset totals:")
        print(f"  CPU energy   : {cpu_e:.12f} kWh")
        print(f"  GPU energy   : {gpu_e:.12f} kWh")
        print(f"  RAM energy   : {ram_e:.12f} kWh")
        print(f"  Total energy : {total_e:.12f} kWh")
        print(f"  Emissions    : {emissions_val:.12f} kgCO2")
        if carbon_intensity is not None:
            print(f"  Carbon int.  : {carbon_intensity:.6f} kgCO2/kWh")

        return cpu_e, gpu_e, ram_e, total_e, emissions_val, carbon_intensity

    except Exception as exc:
        print(f"[CodeCarbon warning] Could not stop/read tracker: {exc}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, None


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

    # DIAGNOSTIC: explicit pre-selection count, so you can see the true
    # discovered total before select_test_samples() applies any subsetting
    # (USE_ALL_TEST_SAMPLES vs. balanced '10 set').
    print(
        f"  [diagnostic] {dataset_name}: {len(discovered):,} test samples "
        f"discovered on disk before selection."
    )

    selected = select_test_samples(discovered, dataset_name)
    print(f"Total selected test samples: {len(selected):,}")

    tag = "10set" if not USE_ALL_TEST_SAMPLES else "all"
    # selected.to_csv(
    #     OUTPUT_ROOT / f"selected_{dataset_name.lower().replace('-', '_')}_{tag}.csv",
    #     index=False,
    # )

    cache_root = result_root / "fingerprinting_test_state_cache_q1"
    rows = []

    # Start CodeCarbon once for the complete dataset inference workload.
    dataset_tracker = start_dataset_energy_tracker(dataset_name)
    full_inference_start = time.perf_counter()

    for i, sample in tqdm(
        selected.iterrows(),
        total=len(selected),
        desc=f"{dataset_name} fingerprinting",
        unit="sample",
    ):
        qasm_path = Path(sample["qasm_path"])

        # Full inference workload:
        # QASM execution -> q=1 reduction -> fidelity kernel -> SVM prediction
        q1_state = execute_qasm_to_q1(qasm_path, cache_root)
        kernel_row = fidelity_kernel_row(q1_state, train_states)

        confidence, margin, entropy = prediction_quality(classifier, kernel_row)

        pred, exec_time = predict_with_energy(
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
            cpu_energy=0.0,
            gpu_energy=0.0,
            ram_energy=0.0,
            total_energy=0.0,
            emissions=0.0,
            carbon_intensity=None,
            svm_path=svm_path,
            train_states_path=train_states_path,
        ))

    full_inference_time = time.perf_counter() - full_inference_start

    (
        dataset_cpu_e,
        dataset_gpu_e,
        dataset_ram_e,
        dataset_total_e,
        dataset_emissions,
        dataset_ci,
    ) = stop_dataset_energy_tracker(dataset_tracker)

    df = pd.DataFrame(rows)

    # Allocate the measured dataset totals equally across sample rows.
    # Summing the columns reproduces the exact dataset-level totals.
    n_rows = len(df)

    if n_rows > 0:
        df["cpu_energy_kwh"] = dataset_cpu_e / n_rows
        df["gpu_energy_kwh"] = dataset_gpu_e / n_rows
        df["ram_energy_kwh"] = dataset_ram_e / n_rows
        df["total_energy_kwh"] = dataset_total_e / n_rows
        df["total_emissions_kg"] = dataset_emissions / n_rows
        df["carbon_intensity_kgco2_kwh"] = dataset_ci

        df["energy_per_token_kwh"] = np.where(
            df["total_tokens"] > 0,
            df["total_energy_kwh"] / df["total_tokens"],
            0.0,
        )

        df["joules_per_token"] = df["energy_per_token_kwh"] * 3_600_000

        df["gpu_energy_pct_of_total"] = np.where(
            df["total_energy_kwh"] > 0,
            (df["gpu_energy_kwh"] / df["total_energy_kwh"]) * 100,
            0.0,
        )

        df["cpu_energy_pct_of_total"] = np.where(
            df["total_energy_kwh"] > 0,
            (df["cpu_energy_kwh"] / df["total_energy_kwh"]) * 100,
            0.0,
        )

    # Keep dataset-level totals explicitly for analysis.
    df["dataset_total_cpu_energy_kwh"] = dataset_cpu_e
    df["dataset_total_gpu_energy_kwh"] = dataset_gpu_e
    df["dataset_total_ram_energy_kwh"] = dataset_ram_e
    df["dataset_total_energy_kwh"] = dataset_total_e
    df["dataset_total_emissions_kg"] = dataset_emissions
    df["dataset_inference_time_sec"] = full_inference_time

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

def quantum_main():
    print("=" * 78)
    print("MNISQ q=1 MULTI-DATASET FINGERPRINTING TEST")
    print("Datasets: MNIST, FashionMNIST, Kuzushiji-MNIST")
    if USE_ALL_TEST_SAMPLES:
        print("Mode: ALL discovered test samples per dataset")
        print("Dataset test counts are allowed to differ.")
        print(f"Expected official test-set size per dataset: {EXPECTED_TEST_TOTAL:,}")
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





# =============================================================================
# DATASET-SPECIFIC ENTRY POINTS
# =============================================================================
OFFICIAL_BASE_URL = (
    "https://qulacs-quantum-datasets.s3.us-west-1.amazonaws.com"
)

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
def human_size(number_of_bytes: int | float) -> str:
    value = float(number_of_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} TB"

def archive_marker(extract_root: Path, archive_name: str) -> Path:
    """
    A small marker file dropped after successful extraction. Its presence
    means "already extracted" so re-running this script is a fast no-op
    instead of re-unzipping thousands of files every time.
    """
    return extract_root / f".{archive_name}.extracted"
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
    function below calls this for the test archive only.
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


def ensure_mnist_dataset_available() -> None:
    print("=" * 78)
    print("Checking MNISQ MNIST TEST dataset")
    print(f"Data root: {MNIST_DATA_ROOT.resolve()}")
    print("=" * 78)

    test_url = f"{OFFICIAL_BASE_URL}/{MNIST_TEST_ARCHIVE_NAME}"

    ensure_archive_downloaded_and_extracted(
        MNIST_TEST_ARCHIVE_NAME,
        test_url,
        MNIST_DOWNLOAD_ROOT,
        MNIST_EXTRACT_ROOT,
    )

    print(f"MNIST test dataset ready. Extracted under: {MNIST_EXTRACT_ROOT.resolve()}")


def ensure_fashionmnist_dataset_available() -> None:
    print("=" * 78)
    print("Checking MNISQ Fashion-MNIST TEST dataset")
    print(f"Data root: {FASHIONMNIST_DATA_ROOT.resolve()}")
    print("=" * 78)

    test_url = f"{OFFICIAL_BASE_URL}/{FASHIONMNIST_TEST_ARCHIVE_NAME}"

    ensure_archive_downloaded_and_extracted(
        FASHIONMNIST_TEST_ARCHIVE_NAME,
        test_url,
        FASHIONMNIST_DOWNLOAD_ROOT,
        FASHIONMNIST_EXTRACT_ROOT,
    )

    print(f"Fashion-MNIST test dataset ready. Extracted under: {FASHIONMNIST_EXTRACT_ROOT.resolve()}")


def ensure_kuzushiji_dataset_available() -> None:
    print("=" * 78)
    print("Checking MNISQ Kuzushiji-MNIST TEST dataset")
    print(f"Data root: {KUZUSHIJI_DATA_ROOT.resolve()}")
    print("=" * 78)

    test_url = f"{OFFICIAL_BASE_URL}/{KUZUSHIJI_TEST_ARCHIVE_NAME}"

    ensure_archive_downloaded_and_extracted(
        KUZUSHIJI_TEST_ARCHIVE_NAME,
        test_url,
        KUZUSHIJI_DOWNLOAD_ROOT,
        KUZUSHIJI_EXTRACT_ROOT,
    )

    print(f"Kuzushiji-MNIST test dataset ready. Extracted under: {KUZUSHIJI_EXTRACT_ROOT.resolve()}")


def ensure_all_mnisq_datasets_available() -> None:
    """Runs all three dataset checks/downloads/extractions in sequence."""
    ensure_mnist_dataset_available()
    print()
    ensure_fashionmnist_dataset_available()
    print()
    ensure_kuzushiji_dataset_available()

    print("\n" + "=" * 78)
    print("All three MNISQ TEST datasets ready.")
    print(f"  MNIST           -> {MNIST_EXTRACT_ROOT.resolve()}")
    print(f"  Fashion-MNIST   -> {FASHIONMNIST_EXTRACT_ROOT.resolve()}")
    print(f"  Kuzushiji-MNIST -> {KUZUSHIJI_EXTRACT_ROOT.resolve()}")
    print("=" * 78)

if __name__ == "__main__":
    qwen_moondream_main()
    yolo_mobilenet_main()
    dnn_cnn_main()
    tinyclip_main()
    ensure_all_mnisq_datasets_available()
    quantum_main()
