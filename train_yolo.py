import os
import random
import torch

from torchvision import datasets
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "./classical_data"

EPOCHS = 10
DEVICE = "cpu"
IMG_SIZE = 64
BATCH_SIZE = 64
VAL_RATIO = 0.10

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


DATASETS = {
    "MNIST": datasets.MNIST,
    "FashionMNIST": datasets.FashionMNIST,
    "KMNIST": datasets.KMNIST,
}


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset(dataset_name, dataset_class):

    root = os.path.join(
        BASE_DIR,
        dataset_name
    )

    # If already prepared, reuse it
    if os.path.exists(
        os.path.join(root, "train")
    ):
        print(f"{dataset_name} already prepared.")
        return root

    print(f"\nPreparing {dataset_name}...")

    train_dataset = dataset_class(
        root="./classical_data",
        train=True,
        download=True
    )

    # Create only train and val folders
    for split in ["train", "val"]:
        for class_id in range(10):
            os.makedirs(
                os.path.join(
                    root,
                    split,
                    str(class_id)
                ),
                exist_ok=True
            )

    # -----------------------------------------
    # Stratified train/validation split
    # -----------------------------------------

    class_indices = {
        i: [] for i in range(10)
    }

    for idx in range(len(train_dataset)):
        _, label = train_dataset[idx]
        class_indices[int(label)].append(idx)

    train_indices = []
    val_indices = []

    for class_id, indices in class_indices.items():

        random.shuffle(indices)

        val_size = int(
            len(indices) * VAL_RATIO
        )

        val_indices.extend(
            indices[:val_size]
        )

        train_indices.extend(
            indices[val_size:]
        )

    # -----------------------------------------
    # Save training images
    # -----------------------------------------

    print("Saving training images...")

    for idx in train_indices:

        image, label = train_dataset[idx]

        # Convert grayscale to RGB
        image = image.convert("RGB")

        path = os.path.join(
            root,
            "train",
            str(label),
            f"{dataset_name}_{idx}.png"
        )

        image.save(path)

    # -----------------------------------------
    # Save validation images
    # -----------------------------------------

    print("Saving validation images...")

    for idx in val_indices:

        image, label = train_dataset[idx]

        image = image.convert("RGB")

        path = os.path.join(
            root,
            "val",
            str(label),
            f"{dataset_name}_{idx}.png"
        )

        image.save(path)

    print(
        f"{dataset_name} dataset preparation complete."
    )

    return root


# ============================================================
# TRAIN AND SAVE
# ============================================================

def train_and_save(dataset_name, dataset_path):

    print("\n" + "=" * 60)
    print(f"Training YOLO on {dataset_name}")
    print("=" * 60)

    # Fresh model for each dataset
    model = YOLO(
        "yolo26n-cls.yaml"
    )

    model.train(
        data=dataset_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        project="./yolo_runs",
        name=f"yolo26n_{dataset_name.lower()}",
        exist_ok=True
    )

    # Save final model
    output_file = (
        f"yolo26n_{dataset_name.lower()}_cpu.pt"
    )

    model.save(output_file)

    print(
        f"{dataset_name} model saved as: "
        f"{output_file}"
    )


# ============================================================
# RUN ALL THREE DATASETS
# ============================================================

for dataset_name, dataset_class in DATASETS.items():

    dataset_path = prepare_dataset(
        dataset_name,
        dataset_class
    )

    train_and_save(
        dataset_name,
        dataset_path
    )


print("\nAll models trained and saved.")