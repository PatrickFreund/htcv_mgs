import sys
from pathlib import Path
from typing import Dict

from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from datamodule.dataset import ImageCSVDataset, TransformSubset
from datamodule.transforms import get_train_transforms, get_val_transforms
from utils.model import get_model

DATASET_MEAN = 0.36995071172714233
DATASET_STD = 0.21818380057811737


def debug_visualize_transform(transform_subset:TransformSubset, config: dict, sample_idx: int = 3):
    """
    Visualisiert ein Originalbild und seine transformierte Version
    aus einem TransformSubset.

    Args:
        transform_subset (TransformSubset): Subset mit definierter transform-Funktion.
        config (dict): Dict mit "mean" und "std" (für Denormalisierung).
        sample_idx (int): Index im Subset, der visualisiert werden soll.
    """
    # Index und zugrundeliegender Datensatz
    original_dataset = transform_subset.base_dataset
    original_idx = transform_subset.indices[sample_idx]

    # Bildinformationen abrufen
    row = original_dataset.labels.iloc[original_idx]
    img_name = row["filename"]
    img_path = original_dataset.img_dir / img_name

    if not img_path.exists():
        img_stem = Path(img_name).stem
        possible_files = list(original_dataset.img_dir.glob(f"{img_stem}.*"))
        if not possible_files:
            raise FileNotFoundError(f"Bild {img_name} nicht gefunden.")
        img_path = possible_files[0]

    # Lade Originalbild
    original_img = Image.open(img_path)
    if original_img.mode == "L":
        original_img = original_img.convert("L")
        cmap = "gray"
    else:
        original_img = original_img.convert("RGB")
        cmap = None

    # Transformiertes Bild abrufen
    transformed_img, label = transform_subset[sample_idx]

    # Denormalisierung
    def denormalize(tensor, mean, std):
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        return (tensor * std + mean).clamp(0, 1)

    mean = config.get("mean", [0.5])
    std = config.get("std", [0.5])
    denorm_img = denormalize(transformed_img.clone(), mean, std)

    # Plot erstellen
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original_img, cmap=cmap)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Transformiert (Denormalisiert)")
    if denorm_img.shape[0] == 1:
        plt.imshow(denorm_img.squeeze(), cmap="gray")
    else:
        plt.imshow(denorm_img.permute(1, 2, 0))  # CHW -> HWC
    plt.axis("off")

    plt.suptitle(f"Label: {label} | Datei: {img_name}")
    plt.tight_layout()
    plt.show()

def test_image_is_grayscale(data_dir: str):
    dataset = ImageCSVDataset(data_dir=data_dir)
    train_transform = get_train_transforms(mean = DATASET_MEAN, std = DATASET_STD)
    for or_img, label in dataset:
        img = train_transform(or_img)
        assert isinstance(img, torch.Tensor), "Bild sollte ein Tensor sein"
        assert img.dim() == 3, "Bild sollte 3 Dimensionen haben (Kanal, Höhe, Breite)"
        assert img.shape[0] == 1, "Bild sollte 1 Kanal haben (Grayscale)"
        assert img.shape[1] == 224 and img.shape[2] == 224, "Bildgröße sollte 224x224 sein"
        print(f"Bildgröße: {or_img.size}, Label: {label}")
        print(f"Transformiertes Bildgröße: {img.shape}")
        print(f"Transformiertes Bild dtype: {img.dtype}")
    
    or_img, label = dataset[0]
    print(f"Bildgröße: {or_img.size}, Label: {label}")
    print(f"Transformiertes Bildgröße: {img.shape}")
    print(f"Transformiertes Bild dtype: {img.dtype}")

    #plot original and transformed image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(or_img, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Transformiert")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def test_resnet18_input_output():
    config = {
        "model_name": "resnet18",
        "pretrained": False,
        "num_classes": 2,
    }
    model = get_model(config)
    
    # Eingabe: Batch aus 1-Kanal-Bildern mit 224x224
    assert isinstance(model, nn.Module), "Modell sollte eine Instanz von nn.Module sein"
    print(f"Modellarchitektur:\n{model}")
    dummy_input = torch.randn(4, 1, 224, 224)
    output = model(dummy_input)
    print(f"Modellausgabeform: {output.shape}")

    assert output.shape == (4, 2), "Modell muss 2 Outputneuronen liefern für binary classification"
