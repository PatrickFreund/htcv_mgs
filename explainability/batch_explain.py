import csv
import sys
import subprocess
import shutil
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
sys.path.append(str(Path(__file__).resolve().parent.parent))
from datamodule.transforms import get_val_transforms
from utils.model import get_model

# === CONFIG ===
MEAN = 0.37203550954887965
STD = 0.21801310757916936

PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(f"Project root: {PROJECT_ROOT}")
ONESHOT_PATH = PROJECT_ROOT / "explainability" / "LRP-for-ResNet" / "oneshot.py"
CONFIG_LRP_PATH = PROJECT_ROOT / "explainability" / "LRP-for-ResNet" / "configs" / "MGS_resnet18.json"
LABEL_CSV = PROJECT_ROOT / "data" / "MGS_data" / "labels" / "labels.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "MGS_data" / "data"
MODEL_PATH = PROJECT_ROOT / "notebooks" / "first_hyperparameter_search_res" / "raw_data" / "config_20250610_070348" / "fold_0" / "best_model.pth"
CONFIG_PATH = PROJECT_ROOT / "notebooks" / "first_hyperparameter_search_res" / "raw_data" / "config_20250610_070348" / "config.yaml"
TENSORBOARD_LOG_PATH = PROJECT_ROOT / "notebooks" / "first_hyperparameter_search_res" / "raw_data" / "config_20250610_070348" / "fold_0" / "events.out.tfevents.1749531828.PCPatrick.33924.140"

OUTPUT_DIR = PROJECT_ROOT / "explainability" / "results" / "run5"
sign_modes = ["all", "positive"]

# === PREP OUTPUT ===
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === COPY MODEL TO RESULT DIR ===
shutil.copy(MODEL_PATH, OUTPUT_DIR / "best_model.pth")
shutil.copy(CONFIG_PATH, OUTPUT_DIR / "config.yaml")
shutil.copy(TENSORBOARD_LOG_PATH, OUTPUT_DIR / "events.out.tfevents.1749531828.PCPatrick.33924.140")


device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model({"pretrained": False, "num_classes": 2, "model_name": "resnet18"})
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
model.eval()
transform = get_val_transforms(mean = MEAN, std = STD)

# === RUN ONESHOT FOR EACH IMAGE ===
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
            probs = F.softmax(result, dim=1)[0] # probs for pain and no pain
            pred = torch.argmax(probs).item()  # Get the predicted class index
        prob_nopain = probs[0].item()  # Probability of no pain
        prob_pain = probs[1].item()  # Probability of pain
        


        
        image_path = IMAGE_DIR / filename

        for sign in sign_modes:
            output_dir = OUTPUT_DIR / f"sign_{sign}"
            output_dir.mkdir(parents=True, exist_ok=True)
            filename_stem = Path(filename).stem
            save_path = output_dir / (
                f"{filename_stem}_lrp_true{label}_painpred{prob_pain:.3f}_"
                f"nopainpred{prob_nopain:.3f}_sign{sign}.png"
            )
            print(f"Saving to: {save_path}")

            cmd = [
                "python", str(ONESHOT_PATH),
                "-c", str(CONFIG_LRP_PATH),
                "--method", "lrp",
                "--batch_size", "1",
                "--base_pretrained", str(MODEL_PATH),
                "--image-path", str(image_path),
                "--skip-connection-prop-type", "flows_skip",
                "--heat-quantization",
                "--label", str(label),
                "--sign", sign,
                "--save-path", str(save_path)
            ]

            print(f"[{sign.upper()}] Processing: {filename} with label {label}")
            subprocess.run(cmd)