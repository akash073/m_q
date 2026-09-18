import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# ============================================================
# DATASETS
# ============================================================

DATASETS = {
    "MNIST": torchvision.datasets.MNIST,
    "FashionMNIST": torchvision.datasets.FashionMNIST,
    "KMNIST": torchvision.datasets.KMNIST,
}


# ============================================================
# PREPROCESSING
# ============================================================

# All three datasets are:
# 1 x 28 x 28 grayscale images
#
# MobileNetV2 pretrained on ImageNet expects:
# 3 x 224 x 224
#
# Therefore:
# grayscale -> 3 channels
# resize -> 224 x 224
# normalize with ImageNet mean/std

transform = transforms.Compose([

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
            0.406
        ),
        std=(
            0.229,
            0.224,
            0.225
        )
    )
])


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_mobilenet(
    dataset_name,
    dataset_class
):

    print("\n" + "=" * 70)
    print(f"Training MobileNetV2 on {dataset_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load training dataset only
    # --------------------------------------------------------

    trainset = dataset_class(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # --------------------------------------------------------
    # Check input shape
    # --------------------------------------------------------

    sample_images, sample_labels = next(
        iter(trainloader)
    )

    print(
        "Input batch shape:",
        sample_images.shape
    )

    print(
        "Label batch shape:",
        sample_labels.shape
    )

    assert sample_images.shape[1:] == (
        3,
        224,
        224
    )

    # --------------------------------------------------------
    # Load fresh pretrained MobileNetV2
    # --------------------------------------------------------

    model = mobilenet_v2(
        weights=MobileNet_V2_Weights.DEFAULT
    )

    # --------------------------------------------------------
    # Replace classifier
    # --------------------------------------------------------

    in_features = (
        model.classifier[1]
        .in_features
    )

    model.classifier[1] = nn.Linear(
        in_features,
        10
    )

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    for epoch in range(NUM_EPOCHS):

        running_loss = 0.0
        correct = 0
        total = 0

        for i, (
            inputs,
            labels
        ) in enumerate(trainloader):

            inputs = inputs.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            # Reset gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(
                inputs
            )

            # Loss
            loss = criterion(
                outputs,
                labels
            )

            # Backward
            loss.backward()

            # Optimize
            optimizer.step()

            running_loss += (
                loss.item()
            )

            # Accuracy
            predicted = torch.argmax(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

            # Progress
            if i % 100 == 99:

                accuracy = (
                    100.0
                    * correct
                    / total
                )

                print(
                    f"[{dataset_name}] "
                    f"Epoch {epoch + 1}, "
                    f"Batch {i + 1} | "
                    f"Loss: "
                    f"{running_loss / 100:.3f} | "
                    f"Accuracy: "
                    f"{accuracy:.2f}%"
                )

                running_loss = 0.0

        # ----------------------------------------------------
        # Epoch result
        # ----------------------------------------------------

        epoch_accuracy = (
            100.0
            * correct
            / total
        )

        print(
            f"{dataset_name} "
            f"Epoch {epoch + 1} "
            f"Training Accuracy: "
            f"{epoch_accuracy:.2f}%"
        )

    # --------------------------------------------------------
    # Move to CPU before saving
    # --------------------------------------------------------

    model.to("cpu")

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    filename = (
        f"mobilenet_v2_"
        f"{dataset_name.lower()}_cpu.pt"
    )

    torch.save(
        model.state_dict(),
        filename
    )

    print(
        f"{dataset_name} model saved to "
        f"{filename}"
    )

    # Free memory before next dataset
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# TRAIN ALL THREE DATASETS
# ============================================================

for dataset_name, dataset_class in DATASETS.items():

    train_mobilenet(
        dataset_name,
        dataset_class
    )


print("\nAll three MobileNetV2 models trained and saved.")