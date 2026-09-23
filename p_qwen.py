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

def main():
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
    # moondream_summaries = (
    #     run_moondream_all_datasets()
    # )
    # all_summaries.extend(
    #     moondream_summaries
    # )

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


if __name__ == "__main__":
    main()
