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

def main():

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


if __name__ == "__main__":
    main()
