import csv
import sys
import subprocess
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from datamodule.transforms import get_val_transforms
from utils.model import get_model

# === CONFIG ===
MEAN = 0.37203550954887965
STD = 0.21801310757916936

PROJECT_ROOT = Path(__file__).resolve().parent#.parent
print(f"Project root: {PROJECT_ROOT}")
ONESHOT_PATH = PROJECT_ROOT / "explainability" / "LRP-for-ResNet" / "oneshot.py"
CONFIG_LRP_PATH = PROJECT_ROOT / "explainability" / "LRP-for-ResNet" / "configs" / "MGS_resnet18.json"
LABEL_CSV = PROJECT_ROOT / "data" / "MGS_data" / "labels" / "labels.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "MGS_data" / "data"
MODEL_PATH = PROJECT_ROOT / "best_model.pth"


OUTPUT_DIR = PROJECT_ROOT / "explainability" / "results" / "run5"
sign_modes = ["all", "positive"]

# === PREP OUTPUT ===
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === COPY MODEL TO RESULT DIR ===
shutil.copy(MODEL_PATH, OUTPUT_DIR / "best_model.pth")
#shutil.copy(CONFIG_PATH, OUTPUT_DIR / "config.yaml")
#shutil.copy(TENSORBOARD_LOG_PATH, OUTPUT_DIR / "events.out.tfevents.1749531828.PCPatrick.33924.140")


device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model({"pretrained": False, "num_classes": 2, "model_name": "resnet18"})
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
model.eval()
transform = get_val_transforms(mean=MEAN, std=STD)

# === RUN ONESHOT FOR EACH IMAGE ===
results = []
with open(LABEL_CSV, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filename = row["filename"]
        label = row["label"]
        image_path = IMAGE_DIR / filename

        if not image_path.exists():
            print(f"[WARN] Missing image: {image_path}")
            continue

        # make prediction
        img = Image.open(image_path).convert("L")
        img = transform(img)
        img = img.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            result = model(img.to(device))
            probs = F.softmax(result, dim=1)[0]  # probs for pain and no pain
            pred = torch.argmax(probs).item()  # Get the predicted class index
        prob_nopain = probs[0].item()  # Probability of no pain
        prob_pain = probs[1].item()  # Probability of pain
        results.append({
            "filename": filename,
            "true_label": label,
            "predicted": pred,
            #"prob_pain": probs[1].item(),
            #"prob_nopain": probs[0].item()
        })

pred_df = pd.DataFrame(results)
pred_csv_path = PROJECT_ROOT / "predictions.csv"
pred_df.to_csv(pred_csv_path, index=False)
print(f"[INFO] Saved predictions to: {pred_csv_path}")

def get_subdataset(filename):
    if filename.startswith("mr_"):
        return "mr"
    elif filename.startswith("lw_"):
        return "lw"
    elif filename.startswith("aw_"):
        return "aw"
    elif filename.startswith("jw_"):
        return "jw"
    elif filename[0].isdigit():
        return "numeric"
    else:
        return "unknown"

def calculate_f1_by_subdataset(pred_csv, meta_csv):
    pred_df = pd.read_csv(pred_csv)
    meta_df = pd.read_csv(meta_csv, names=["filename", "id", "time", "experiment", "control"])
    pred_df["filename"] = pred_df["filename"].astype(str).str.strip()
    meta_df["filename"] = meta_df["filename"].astype(str).str.strip()
    df = pd.merge(pred_df, meta_df, on="filename", how="inner")
    df["experiment"] = df["experiment"].fillna("unknown")
    df["true_label"] = df["true_label"].astype(int)
    df["predicted"] = df["predicted"].astype(int)

    print("\n=== F1 Score by Experiment ===")
    for exp in df["experiment"].unique():
        subset = df[df["experiment"] == exp]
        if len(subset) == 0:
            continue
        f1 = f1_score(subset["true_label"], subset["predicted"], average="weighted")
        print(f"Experiment '{exp}': F1 score = {f1:.4f}")

calculate_f1_by_subdataset("predictions.csv","v3_main.csv")



