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
NUM_TEST_SAMPLES = int(os.getenv("NUM_TEST_SAMPLES", 10))

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


def main():

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


if __name__ == "__main__":
    main()