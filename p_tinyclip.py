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

def main():
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


if __name__ == "__main__":
    main()
