import json
import re
import shutil
import subprocess
import sys
import tempfile
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ==== Imports from project structure ====
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.constants import DATASET_MEAN, DATASET_STD
from datamodule.dataset import ImageCSVDataset
from datamodule.transforms import get_val_transforms
from datamodule.splitter import KFoldSplit, RandomSplit
from utils.model import get_model
from utils.utility import set_seed


class EvaluationPipeline:
    """
    EvaluationPipeline class for running evaluation tasks on a trained model using LRP 
    (Layer-wise Relevance Propagation) and classification accuracy metrics.
    """
    def __init__(
        self,
        project_root: Path,
        image_dir: Path,
        model_path: Path,
        config_path: Path,
        tensorboard_log_path: Path,
        output_dir: Path,
        strain_csv_path: Optional[Path] = None,
        masks_path: Optional[Path] = None,
        lrp_masks_path: Optional[Path] = None,
        split_strategy: str = "kfold",
        fold: int = 0,
        fold_num: int = 3,
        seed: int = 42,
        fold_seed: Optional[int] = None,
    ):
        """
        Initializes the evaluation pipeline with paths and configuration settings.
        
        Args:
            project_root (Path): Root path of the project.
            image_dir (Path): Directory containing image data.
            model_path (Path): Path to the trained model.
            config_path (Path): Path to the model/config YAML file.
            tensorboard_log_path (Path): Path to TensorBoard log file.
            output_dir (Path): Directory for evaluation output.
            strain_csv_path (Optional[Path]): CSV file mapping filenames to strain IDs.
            masks_path (Optional[Path]): Directory containing ground truth ROI masks.
            lrp_masks_path (Optional[Path]): Directory containing LRP maps.
            split_strategy (str): Split method ('kfold' or 'random').
            fold (int): Fold index to evaluate.
            fold_num (int): Number of folds (for k-fold split).
            seed (int): Random seed for reproducibility.
            fold_seed (Optional[int]): Optional override for fold-specific seed.
        """
        self.project_root = project_root
        self.image_dir = image_dir
        self.model_path = model_path
        self.config_path = config_path
        self.tensorboard_log_path = tensorboard_log_path
        self.output_dir = output_dir
        self.strain_csv_path = strain_csv_path
        self.masks_path = masks_path
        self.lrp_masks_path = lrp_masks_path
        self.split_strategy = split_strategy
        self.fold = fold
        self.fold_num = fold_num
        self.seed = seed
        self.fold_seed = fold_seed if fold_seed is not None else seed
        
        self.oneshot_path = self.project_root / "explainability" / "LRP-for-ResNet" / "oneshot.py"
        self.config_lrp_path = self.project_root / "explainability" / "LRP-for-ResNet" / "configs" / "MGS_resnet18.json"
        self.sign_modes = ["positive"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = get_val_transforms(mean=DATASET_MEAN, std=DATASET_STD)

        self._prepare_output()
        self.model = self._load_model()
        self.dataset, self.split, self.filenames = self._load_split()

    def _lrp_already_generated(self, sign_modes: List[str]) -> Dict[str, bool]:
        """
        Checks whether LRP maps for specified sign modes already exist.

        Args:
            sign_modes (List[str]): List of LRP sign modes to check.

        Returns:
            Dict[str, bool]: Dictionary mapping sign modes to presence of LRP output.
        """
        existing_dirs = {}
        for sign in sign_modes:
            lrp_dir = self.output_dir / f"sign_{sign}"
            attn_map_dir = self.output_dir / f"sign_{sign}_attention_maps"

            # LRP-Ordner muss existieren und Dateien enthalten
            if not lrp_dir.exists() or not any(lrp_dir.iterdir()):
                existing_dirs[sign] = False
                continue

            # Wenn auch ROI-Auswertung gemacht wird, prüfen ob attention maps existieren
            if self.lrp_masks_path is None:
                if not attn_map_dir.exists() or not any(attn_map_dir.iterdir()):
                    existing_dirs[sign] = False
                    continue
            existing_dirs[sign] = True
        return existing_dirs

    def _prepare_output(self):
        """
        Prepares the output directory by creating it and copying relevant files.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.model_path, self.output_dir / self.model_path.name)
        shutil.copy(self.config_path, self.output_dir / self.config_path.name)
        shutil.copy(self.tensorboard_log_path, self.output_dir / self.tensorboard_log_path.name)

    def _load_model(self) -> torch.nn.Module:
        """
        Loads the trained model from disk.

        Returns:
            torch.nn.Module: Loaded model set to evaluation mode.
        """
        model = get_model({"pretrained": False, "num_classes": 2, "model_name": "resnet18"})
        model.load_state_dict(torch.load(self.model_path))
        model.to(self.device).eval()
        return model

    def _load_split(self) -> Tuple[ImageCSVDataset, Tuple[List[int], List[int]], List[str]]:
        """
        Loads dataset and splits based on the configured split strategy.

        Returns:
            Tuple[ImageCSVDataset, Tuple[List[int], List[int]], List[str]]:
                Dataset, train/val indices, and filenames.
        """
        dataset = ImageCSVDataset(data_dir=self.image_dir)
        filenames = dataset.filenames

        if self.split_strategy == "kfold":
            splitter = KFoldSplit(k=self.fold_num, seed=self.seed)
            splits = splitter.get_splits(dataset, seed=self.seed)
            split = splits[self.fold]
        elif self.split_strategy == "random":
            splitter = RandomSplit(seed=self.seed)
            splits = splitter.get_splits(dataset, seed=self.seed)
            split = splits
        else:
            raise ValueError(f"Unknown splitting strategy: {self.split_strategy}")

        set_seed(self.fold_seed)
        return dataset, split, filenames

    def get_validation_data(self) -> List[str]:
        """
        Returns the list of validation filenames based on the split strategy.

        Returns:
            List[str]: List of filenames in the validation split.
        """
        return [self.filenames[i] for i in self.split[1]]

    def evaluate_all(self, per_strain: bool = False, do_class_eval: bool = True, do_lrp: bool = False, do_lrp_roi_eval: bool = False):
        """
        Runs evaluation pipeline including classification, LRP generation, and ROI-based evaluation.

        Args:
            per_strain (bool): Whether to evaluate per strain ID.
            do_class_eval (bool): Whether to perform classification evaluation.
            do_lrp (bool): Whether to generate LRP maps.
            do_lrp_roi_eval (bool): Whether to perform LRP ROI-based evaluation.
        """
        if self.device == "cuda":
            if do_lrp_roi_eval:
                do_lrp = True

            if do_lrp:
                existing_dirs = self._lrp_already_generated(self.sign_modes)
                if all(existing_dirs.values()):
                    print("LRP results already generated for all sign modes, skipping LRP evaluation.")
                else:
                    sign_modes_to_run = [sign for sign in self.sign_modes if not existing_dirs.get(sign, False)]
                    self._run_lrp_explainability(sign_modes=sign_modes_to_run)
        else:
            print("Skipping LRP evaluation on CPU, as it requires GPU support.")
            do_lrp = False
            do_lrp_roi_eval = False
        
        if per_strain:
            if self.strain_csv_path is None:
                raise ValueError("strain_csv_path is required for per-strain evaluation")
            strains = pd.read_csv(self.strain_csv_path)
            val_filenames = set(self.get_validation_data())
            for strain_id in strains["strain_id"].unique():
                output_dir = self.output_dir / f"strain_{strain_id}"
                strain_filenames = set(strains[strains["strain_id"] == strain_id]["filename"])
                relevant_filenames = strain_filenames & val_filenames
                relevant_filenames = val_filenames & relevant_filenames
                image_paths = [self.image_dir / "data" / f for f in relevant_filenames]
                
                if do_class_eval:
                    self._evaluate_class_predictions(image_paths, output_dir / "class_eval")

                if do_lrp_roi_eval:
                    self._evaluate_lrp_roi(strain_id, relevant_filenames, output_dir / "lrp_roi_eval")

        else:
            filenames = self.get_validation_data()
            image_paths = [self.image_dir / "data" / f for f in filenames]

            output_dir = self.output_dir / "overall"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            if do_class_eval:
                self._evaluate_class_predictions(image_paths, output_dir / "overall_class_eval")

            if do_lrp_roi_eval:
                self._evaluate_lrp_roi("overall", filenames, output_dir / "overall_lrp_roi_eval")

    def _evaluate_class_predictions(self, image_paths: List[Path], output_dir: Path):
        """
        Evaluates classification accuracy and prediction confidence per image.

        Args:
            image_paths (List[Path]): List of image file paths to evaluate.
            output_dir (Path): Directory to save evaluation results.
        """
        output_dir.mkdir(exist_ok=True, parents=True)
        results = []

        for filename in image_paths:
            img_name = filename.name
            if img_name not in self.filenames:
                print(f"[WARN] {img_name} not found in dataset filenames, skipping.")
                continue
            idx = self.filenames.index(img_name)
            img, label = self.dataset[idx]
            img = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img)
                probs = F.softmax(output, dim=1)[0]
                pred = torch.argmax(probs).item()
                correct_prob = probs[label].item()

            results.append({
                "filename": img_name,
                "true_label": label,
                "pred_label": pred,
                "correct_prob": correct_prob,
            })

        df = pd.DataFrame(results)
        summary = df.groupby("true_label")["correct_prob"].agg(["mean", "std"])
        summary["num_samples"] = df.groupby("true_label")["correct_prob"].count()
        accuracy_per_class = df.groupby("true_label").apply(lambda x: (x["true_label"] == x["pred_label"]).mean())

        df.boxplot(column="correct_prob", by="true_label")
        plt.title("Distribution of Correct Prediction Probabilities")
        plt.suptitle("")
        plt.savefig(output_dir / "correct_probs_boxplot.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.bar(accuracy_per_class.index, accuracy_per_class.values, color='skyblue')
        plt.title("Accuracy per True Label")
        plt.xlabel("True Label")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_per_class.png")
        plt.close()
        
        results = {
            "probability_stats": summary.to_dict(orient="index"),
            "accuracy_per_class": accuracy_per_class.to_dict(),
            "raw_df": df.to_dict(orient="records"),
        }

        with open(output_dir / "class_eval.json", "w") as f:
            json.dump(results, f, indent=4)

    def _evaluate_lrp_roi(self, strain_id: str, filenames: List[str], output_dir: Optional[Path] = None):
        """
        Evaluates LRP relevance concentration within ROI / segmentation masks.

        Args:
            strain_id (str): ID of the strain for evaluation.
            filenames (List[str]): Filenames corresponding to this strain.
            output_dir (Optional[Path]): Directory to store results.
        """
        
        def extract_base_key(fname):
            return re.split(r'_lrp|_mask', fname)[0]

        def collect_files(folder):
            suffixes = [".png", ".jpg", ".jpeg"]
            return list(chain.from_iterable(folder.glob(f"*{s}") for s in suffixes))

        masks = collect_files(self.masks_path)
        lrp_maps = collect_files(self.lrp_masks_path)

        valid_basenames = {Path(f).stem for f in filenames}

        masks = [m for m in masks if any(m.stem.startswith(v) for v in valid_basenames)]
        lrp_maps = [l for l in lrp_maps if any(l.stem.startswith(v) for v in valid_basenames)]
        
        mask_dict = {extract_base_key(p.name): p for p in masks}
        lrp_dict = {extract_base_key(p.name): p for p in lrp_maps}

        common = sorted(set(mask_dict.keys()) & set(lrp_dict.keys()))
        ratios = {}

        for fname in common:
            mask = cv2.imread(str(mask_dict[fname]), cv2.IMREAD_GRAYSCALE)
            lrp = cv2.imread(str(lrp_dict[fname]), cv2.IMREAD_GRAYSCALE)

            if mask.shape != lrp.shape:
                mask = cv2.resize(mask, (lrp.shape[1], lrp.shape[0]), interpolation=cv2.INTER_CUBIC)
            if lrp.sum() == 0:
                print(f"[WARN] No relevance in LRP map for {fname}, skipping.")
                continue
            roi_ratio = lrp[mask > 0].sum() / lrp.sum()
            ratios[fname] = roi_ratio


        values = np.array(list(ratios.values()))
        result = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "n": int(len(values)),
            "ratios": ratios
        }
        
        if output_dir is None:
            output_dir = self.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)
        with open(output_dir / f"strain_{strain_id}_roi_lrp_explainability.json", "w") as f:
            json.dump(result, f, indent=4)
    
    def _run_lrp_explainability(
        self,
        output_dir: Optional[Path] = None,
        sign_modes: Optional[List[str]] = None
    ):
        """
        Runs LRP generation via subprocess using oneshot.py for specified sign modes.

        Args:
            output_dir (Optional[Path]): Directory where LRP maps will be saved.
            sign_modes (Optional[List[str]]): LRP sign modes to generate (e.g., ['positive']).
        """
        if output_dir is None:
            output_dir = self.output_dir 
        
        if sign_modes is None:
            sign_modes = self.sign_modes

        image_paths = []
        save_paths = {sign: [] for sign in sign_modes}
        labels = []
        for dataset_idx in self.split[1]:  # Use validation indices
            filename = self.filenames[dataset_idx]
            img_path = self.image_dir / "data" / filename
            image_paths.append(img_path)
            img, label = self.dataset[dataset_idx] 
            img = self.transform(img).unsqueeze(0).to(self.device) 
            
            with torch.no_grad():
                result = self.model(img)
                probs = F.softmax(result, dim=1)[0]
                prob_nopain = probs[0].item()
                prob_pain = probs[1].item()
                labels.append(torch.argmax(probs).item())  # Get the predicted class index

            for sign in sign_modes:
                sign_output_dir = output_dir / f"sign_{sign}"
                sign_output_dir.mkdir(parents=True, exist_ok=True)
                filename_stem = Path(filename).stem
                save_paths[sign].append(sign_output_dir / (
                    f"{filename_stem}_lrp_true{label}_painpred{prob_pain:.3f}_"
                    f"nopainpred{prob_nopain:.3f}_sign{sign}.png"
                ))

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as img_file:
            for path in image_paths:
                img_file.write(f"{path}\n")
            img_file_path = img_file.name

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as label_file:
            for label in labels:
                label_file.write(f"{label}\n")
            label_file_path = label_file.name
            
        save_path_files = {}  
        for sign in sign_modes:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as spf:
                for path in save_paths[sign]:
                    spf.write(f"{path}\n")
                save_path_files[sign] = spf.name
                

        for sign in sign_modes:
            cmd = [
                "python", str(self.oneshot_path),
                "-c", str(self.config_lrp_path),
                "--method", "lrp",
                "--batch_size", "1",
                "--base_pretrained", str(self.model_path),
                "--image-paths-file", img_file_path,
                "--labels-file", label_file_path,
                "--save-paths-file", save_path_files[sign],
                "--skip-connection-prop-type", "flows_skip",
                "--heat-quantization",
                "--hq-level", "12",
                "--normalize",
                "--sign", sign,
            ]
            _ = subprocess.run(cmd, capture_output=True, text=True)
            Path(save_path_files[sign]).unlink() 
        Path(img_file_path).unlink()
        Path(label_file_path).unlink()
