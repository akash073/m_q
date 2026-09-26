# ============================================================
# 10-QUBIT HYBRID YOLO CLASSIFIER
# MNIST / FashionMNIST / KMNIST
#
# Behavior:
#   1. For each dataset, look for a saved hybrid YOLO checkpoint.
#   2. If the checkpoint exists, load it.
#   3. If it does not exist, build YOLO26n-CLS with a 10-qubit
#      PennyLane classification head, train for 10 epochs, and save it.
#   4. Evaluate the saved/loaded model on the test set.
#
# IMPORTANT:
# This is a REAL hybrid quantum-classical YOLO classifier.
# It is different from the uploaded QYOLO paper, which is
# "quantum-inspired" and does not use physical/simulated qubits.
#
# Install:
#   pip install torch torchvision ultralytics pennylane \
#               pennylane-lightning scikit-learn tqdm
#
# Run:
#   python yolo_10qubit_train_or_load.py
#
# Optional environment variables:
#   DEVICE_MODE=cpu
#   TRAIN_BATCH_SIZE=8
#   TEST_BATCH_SIZE=16
#   TRAIN_SAMPLES=0        # 0 = use all training samples
#   TEST_SAMPLES=10000     # 0 = use all test samples
#   FORCE_RETRAIN=0
#   PENNYLANE_DEVICE=default.qubit
# ============================================================

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("./classical_data")
CHECKPOINT_DIR = Path("./quantum_checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EPOCHS = 10
IMG_SIZE = 64

N_QUBITS = 1
N_Q_LAYERS = 4
N_CLASSES = 10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4

TRAIN_BATCH_SIZE = int(
    os.getenv("TRAIN_BATCH_SIZE", "16")
)

TEST_BATCH_SIZE = int(
    os.getenv("TEST_BATCH_SIZE", "16")
)

# 0 means use the full dataset.
TRAIN_SAMPLES = int(
    os.getenv("TRAIN_SAMPLES", "0")
)

TEST_SAMPLES = int(
    os.getenv("NUM_TEST_SAMPLES", "1")
)

FORCE_RETRAIN = (
    os.getenv("FORCE_RETRAIN", "0").strip() == "1"
)

PENNYLANE_DEVICE = os.getenv(
    "PENNYLANE_DEVICE",
    "default.qubit",
)

DEVICE_MODE = os.getenv(
    "DEVICE_MODE",
    "cpu",
).lower()

if DEVICE_MODE == "cuda" and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    # PennyLane default.qubit is CPU-based, so CPU is the safest default.
    DEVICE = torch.device("cpu")


DATASET_CONFIGS = {
    "MNIST": {
        "dataset_class": datasets.MNIST,
        "key": "mnist",
    },
    "FashionMNIST": {
        "dataset_class": datasets.FashionMNIST,
        "key": "fashionmnist",
    },
    "KMNIST": {
        "dataset_class": datasets.KMNIST,
        "key": "kmnist",
    },
}


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

# YOLO classification expects a 3-channel image.
# MNIST-family datasets are grayscale, so replicate to RGB.
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


# ============================================================
# 10-QUBIT QUANTUM CLASSIFICATION HEAD
# ============================================================

def build_quantum_circuit(
    n_qubits: int = N_QUBITS,
    n_layers: int = N_Q_LAYERS,
    device_name: str = PENNYLANE_DEVICE,
):
    """
    10-qubit variational quantum circuit.

    Input:
        x       -> n_qubits encoded angles
        weights -> trainable rotations

    Output:
        one Pauli-Z expectation value per qubit

    Since N_QUBITS = 10 and MNIST-family datasets have 10 classes,
    the ten expectation values are used as ten class features.
    """

    dev = qml.device(
        device_name,
        wires=n_qubits,
    )

    @qml.qnode(
        dev,
        interface="torch",
        diff_method="best",
    )
    def circuit(x, weights):

        # -------------------------------
        # Angle encoding
        # -------------------------------
        for q in range(n_qubits):
            qml.RY(
                x[q],
                wires=q,
            )

            qml.RZ(
                x[q],
                wires=q,
            )

        # -------------------------------
        # Trainable variational layers
        # -------------------------------
        for layer in range(n_layers):

            for q in range(n_qubits):
                qml.Rot(
                    weights[layer, q, 0],
                    weights[layer, q, 1],
                    weights[layer, q, 2],
                    wires=q,
                )

            # Ring entanglement
            if n_qubits > 1:
                for q in range(n_qubits):
                    qml.CNOT(
                        wires=[
                            q,
                            (q + 1) % n_qubits,
                        ]
                    )

        # 10 expectation values
        return [
            qml.expval(
                qml.PauliZ(q)
            )
            for q in range(n_qubits)
        ]

    return circuit


class QuantumYOLOHead(nn.Module):
    """
    Replaces YOLO's final classical Linear classifier.

    Pipeline:

        YOLO feature vector
              |
              v
        Linear(in_features -> 10)
              |
            tanh*pi
              |
              v
         10-qubit VQC
              |
              v
        10 expectation values
              |
              v
        trainable scale + bias
              |
              v
        10 class logits
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_Q_LAYERS,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)

        # Compress YOLO's high-dimensional feature vector
        # to one angle per qubit.
        self.feature_to_qubits = nn.Linear(
            self.in_features,
            self.n_qubits,
        )

        # Trainable VQC parameters.
        self.q_weights = nn.Parameter(
            0.01 * torch.randn(
                self.n_layers,
                self.n_qubits,
                3,
            )
        )

        # Trainable affine transformation after quantum measurement.
        self.output_scale = nn.Parameter(
            torch.ones(
                self.n_qubits
            )
        )

        self.output_bias = nn.Parameter(
            torch.zeros(
                self.n_qubits
            )
        )

        self.qnode = build_quantum_circuit(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            device_name=PENNYLANE_DEVICE,
        )

    def _forward_single(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        # Compress features to exactly 10 values.
        angles = self.feature_to_qubits(
            features
        )

        # Bound encoding angles.
        angles = (
            torch.tanh(angles)
            * torch.pi
        )

        q_out = self.qnode(
            angles,
            self.q_weights,
        )

        # PennyLane can return a tuple/list of scalar tensors.
        if isinstance(q_out, torch.Tensor):
            q_features = q_out
        else:
            q_features = torch.stack(
                list(q_out)
            )

        q_features = q_features.to(
            dtype=torch.float32
        )

        logits = (
            q_features
            * self.output_scale
            + self.output_bias
        )

        return logits

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # YOLO classification head passes shape:
        #     (batch, in_features)

        if x.ndim == 1:
            return self._forward_single(
                x
            ).unsqueeze(0)

        # PennyLane simulation is evaluated sample-by-sample.
        outputs = [
            self._forward_single(
                sample
            )
            for sample in x
        ]

        return torch.stack(
            outputs,
            dim=0,
        )


# ============================================================
# FIND / REPLACE YOLO FINAL LINEAR CLASSIFIER
# ============================================================

def find_last_linear(
    module: nn.Module,
):
    """
    Return:
        parent_module,
        child_name,
        linear_module

    for the last nn.Linear layer in the model.
    """

    last = None

    for full_name, child in module.named_modules():
        if isinstance(
            child,
            nn.Linear,
        ):
            last = (
                full_name,
                child,
            )

    if last is None:
        raise RuntimeError(
            "Could not find a final nn.Linear "
            "layer in the YOLO classification model."
        )

    full_name, linear = last

    parts = full_name.split(".")

    parent = module

    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[
                int(part)
            ]
        else:
            parent = getattr(
                parent,
                part,
            )

    child_name = parts[-1]

    return (
        parent,
        child_name,
        linear,
    )


def set_child_module(
    parent: nn.Module,
    child_name: str,
    new_module: nn.Module,
):
    if child_name.isdigit():
        parent[
            int(child_name)
        ] = new_module
    else:
        setattr(
            parent,
            child_name,
            new_module,
        )


def build_quantum_yolo():
    """
    Build YOLO26n classification model and replace only its
    final classical linear classifier with the 10-qubit head.
    """

    yolo = YOLO(
        "yolo26n-cls.yaml"
    )

    torch_model = yolo.model

    (
        parent,
        child_name,
        old_linear,
    ) = find_last_linear(
        torch_model
    )

    in_features = int(
        old_linear.in_features
    )

    print(
        "Replacing YOLO final classifier:"
    )

    print(
        f"  Linear({in_features}, "
        f"{old_linear.out_features})"
    )

    print(
        f"  -> 10-qubit quantum head"
    )

    quantum_head = QuantumYOLOHead(
        in_features=in_features,
        n_qubits=N_QUBITS,
        n_layers=N_Q_LAYERS,
    )

    set_child_module(
        parent,
        child_name,
        quantum_head,
    )

    torch_model = torch_model.to(
        DEVICE
    )

    return torch_model


# ============================================================
# CHECKPOINTS
# ============================================================

def get_checkpoint_path(
    dataset_key: str,
) -> Path:

    return (
        CHECKPOINT_DIR
        / (
            f"yolo26n_{dataset_key}_"
            f"{N_QUBITS}qubit_cpu.pth"
        )
    )


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    dataset_name: str,
    epoch: int,
    best_val_accuracy: float,
):
    checkpoint = {
        "dataset_name": str(
            dataset_name
        ),

        "model_name":
            "YOLO26n-CLS-10Qubit",

        "n_qubits": int(
            N_QUBITS
        ),

        "n_q_layers": int(
            N_Q_LAYERS
        ),

        "epoch": int(
            epoch
        ),

        "best_val_accuracy": float(
            best_val_accuracy
        ),

        "pennylane_device": str(
            PENNYLANE_DEVICE
        ),

        "pennylane_version": str(
            qml.__version__
        ),

        "torch_version": str(
            torch.__version__
        ),

        "state_dict":
            model.state_dict(),
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print(
        f"Saved checkpoint: "
        f"{checkpoint_path}"
    )


def load_checkpoint(
    checkpoint_path: Path,
):
    """
    Rebuild the architecture first, then load the trained weights.
    """

    model = build_quantum_yolo()

    # PyTorch 2.6+:
    # This checkpoint is generated by this script,
    # so it is a trusted local file.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    saved_qubits = int(
        checkpoint.get(
            "n_qubits",
            N_QUBITS,
        )
    )

    if saved_qubits != N_QUBITS:
        raise RuntimeError(
            f"Checkpoint uses {saved_qubits} qubits, "
            f"but current code uses {N_QUBITS}."
        )

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        f"Loaded checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Saved epoch: "
        f"{checkpoint.get('epoch', '?')}"
    )

    print(
        "Saved best validation accuracy: "
        f"{checkpoint.get('best_val_accuracy', '?')}"
    )

    return model


# ============================================================
# DATASETS
# ============================================================

def make_dataset(
    dataset_class,
    train: bool,
):
    return dataset_class(
        root=str(
            BASE_DIR
        ),
        train=train,
        download=True,
        transform=IMAGE_TRANSFORM,
    )


def maybe_subset(
    dataset,
    n_samples: int,
    seed: int,
):
    if (
        n_samples <= 0
        or n_samples >= len(dataset)
    ):
        return dataset

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:n_samples]

    return Subset(
        dataset,
        indices.tolist(),
    )


def make_loaders(
    dataset_class,
):
    full_train = make_dataset(
        dataset_class,
        train=True,
    )

    test_dataset = make_dataset(
        dataset_class,
        train=False,
    )

    full_train = maybe_subset(
        full_train,
        TRAIN_SAMPLES,
        SEED,
    )

    test_dataset = maybe_subset(
        test_dataset,
        TEST_SAMPLES,
        SEED + 1,
    )

    # Use 90/10 train/validation split.
    total = len(
        full_train
    )

    n_val = max(
        1,
        int(
            0.10 * total
        ),
    )

    n_train = (
        total - n_val
    )

    train_dataset, val_dataset = (
        torch.utils.data.random_split(
            full_train,
            [
                n_train,
                n_val,
            ],
            generator=torch.Generator()
            .manual_seed(
                SEED
            ),
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# OUTPUT NORMALIZATION
# ============================================================

def extract_logits(
    output,
):
    """
    Ultralytics model output can vary slightly by version.
    This helper extracts the classification logits tensor.
    """

    if isinstance(
        output,
        torch.Tensor,
    ):
        return output

    if isinstance(
        output,
        (list, tuple),
    ):
        # Prefer a B x 10 tensor.
        for value in output:
            if (
                isinstance(
                    value,
                    torch.Tensor,
                )
                and value.ndim == 2
                and value.shape[-1] == N_CLASSES
            ):
                return value

        # Fallback to first tensor.
        for value in output:
            if isinstance(
                value,
                torch.Tensor,
            ):
                return value

    if isinstance(
        output,
        dict,
    ):
        for key in (
            "logits",
            "preds",
            "output",
        ):
            value = output.get(
                key
            )

            if isinstance(
                value,
                torch.Tensor,
            ):
                return value

    raise RuntimeError(
        "Could not extract classification logits "
        "from YOLO model output."
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in tqdm(
        loader,
        desc="Validation",
        leave=False,
    ):
        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        output = model(
            images
        )

        logits = extract_logits(
            output
        )

        loss = criterion(
            logits,
            labels,
        )

        total_loss += (
            float(
                loss.item()
            )
            * labels.size(0)
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        total_correct += int(
            (
                predictions
                == labels
            )
            .sum()
            .item()
        )

        total_count += (
            labels.size(0)
        )

    return (
        total_loss
        / max(
            1,
            total_count,
        ),

        total_correct
        / max(
            1,
            total_count,
        ),
    )


# ============================================================
# TRAINING
# ============================================================

def train_model(
    dataset_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    checkpoint_path: Path,
):
    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_accuracy = -1.0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        progress = tqdm(
            train_loader,
            desc=(
                f"{dataset_name} "
                f"epoch {epoch}/{EPOCHS}"
            ),
            unit="batch",
        )

        for images, labels in progress:
            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                images
            )

            logits = extract_logits(
                output
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            batch_size = (
                labels.size(0)
            )

            total_loss += (
                float(
                    loss.item()
                )
                * batch_size
            )

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            total_correct += int(
                (
                    predictions
                    == labels
                )
                .sum()
                .item()
            )

            total_count += (
                batch_size
            )

            progress.set_postfix(
                loss=round(
                    total_loss
                    / max(
                        1,
                        total_count,
                    ),
                    4,
                ),
                acc=round(
                    total_correct
                    / max(
                        1,
                        total_count,
                    ),
                    4,
                ),
            )

        train_loss = (
            total_loss
            / max(
                1,
                total_count,
            )
        )

        train_accuracy = (
            total_correct
            / max(
                1,
                total_count,
            )
        )

        (
            val_loss,
            val_accuracy,
        ) = evaluate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"\nEpoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_accuracy:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_accuracy:.4f}"
        )

        # Save best checkpoint.
        if (
            val_accuracy
            > best_val_accuracy
        ):
            best_val_accuracy = (
                val_accuracy
            )

            save_checkpoint(
                model=model,
                checkpoint_path=(
                    checkpoint_path
                ),
                dataset_name=(
                    dataset_name
                ),
                epoch=epoch,
                best_val_accuracy=(
                    best_val_accuracy
                ),
            )

    # Return the best model, not just the last epoch.
    return load_checkpoint(
        checkpoint_path
    )


# ============================================================
# TRAIN IF MISSING / LOAD IF AVAILABLE
# ============================================================

def get_or_train_model(
    dataset_name: str,
    dataset_key: str,
    train_loader,
    val_loader,
):
    checkpoint_path = (
        get_checkpoint_path(
            dataset_key
        )
    )

    if (
        checkpoint_path.exists()
        and not FORCE_RETRAIN
    ):
        print(
            "\nSaved 10-qubit YOLO model found."
        )

        model = load_checkpoint(
            checkpoint_path
        )

        return (
            model,
            checkpoint_path,
        )

    print(
        "\nNo saved 10-qubit YOLO model found."
    )

    print(
        f"Training for {EPOCHS} epochs "
        f"with {N_QUBITS} qubits..."
    )

    model = build_quantum_yolo()

    model = train_model(
        dataset_name=dataset_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=(
            checkpoint_path
        ),
    )

    return (
        model,
        checkpoint_path,
    )


# ============================================================
# TEST MODEL
# ============================================================

@torch.no_grad()
def test_model(
    dataset_name,
    model,
    test_loader,
):
    model.eval()

    y_true = []
    y_pred = []

    for images, labels in tqdm(
        test_loader,
        desc=f"Testing {dataset_name}",
        unit="batch",
    ):
        images = images.to(
            DEVICE
        )

        output = model(
            images
        )

        logits = extract_logits(
            output
        )

        predictions = (
            logits.argmax(
                dim=1
            )
            .cpu()
            .numpy()
        )

        y_pred.extend(
            predictions.tolist()
        )

        y_true.extend(
            labels.numpy().tolist()
        )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        labels=list(
            range(N_CLASSES)
        ),
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=list(
            range(N_CLASSES)
        ),
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=list(
            range(N_CLASSES)
        ),
        average="weighted",
        zero_division=0,
    )

    print(
        f"\n{dataset_name} TEST RESULTS"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "precision_weighted": float(
            precision
        ),
        "recall_weighted": float(
            recall
        ),
        "f1_weighted": float(
            f1
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "YOLO26n-CLS + 1-QUBIT "
        "PENNYLANE CLASSIFIER"
    )

    print(
        "=" * 72
    )

    print(
        f"PyTorch device   : {DEVICE}"
    )

    print(
        f"PennyLane device : "
        f"{PENNYLANE_DEVICE}"
    )

    print(
        f"Qubits           : "
        f"{N_QUBITS}"
    )

    print(
        f"Quantum layers   : "
        f"{N_Q_LAYERS}"
    )

    print(
        f"Epochs           : "
        f"{EPOCHS}"
    )

    print(
        f"Train batch      : "
        f"{TRAIN_BATCH_SIZE}"
    )

    print(
        f"Train samples    : "
        f"{TRAIN_SAMPLES if TRAIN_SAMPLES > 0 else 'ALL'}"
    )

    for (
        dataset_name,
        config,
    ) in DATASET_CONFIGS.items():

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"DATASET: "
            f"{dataset_name}"
        )

        print(
            "=" * 72
        )

        (
            train_loader,
            val_loader,
            test_loader,
        ) = make_loaders(
            config[
                "dataset_class"
            ]
        )

        (
            model,
            checkpoint_path,
        ) = get_or_train_model(
            dataset_name=(
                dataset_name
            ),
            dataset_key=(
                config["key"]
            ),
            train_loader=(
                train_loader
            ),
            val_loader=(
                val_loader
            ),
        )

        print(
            f"Using checkpoint: "
            f"{checkpoint_path}"
        )

        test_model(
            dataset_name=(
                dataset_name
            ),
            model=model,
            test_loader=(
                test_loader
            ),
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        "\nAll YOLO 10-qubit "
        "models complete."
    )


if __name__ == "__main__":
    main()
