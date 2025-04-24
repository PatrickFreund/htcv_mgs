# Hot Topic Computer Vision - Mouse Grimace Scale
This repository provides a modular training pipeline for Convolutional Neural Networks (CNNs) using PyTorch. The setup is designed for **image classification** tasks with CSV-labeled datasets.

## Directory Structure 

### 📂 Data
The `data/` folder contains the datasets used for training and testing. The training pipeline supports **two options** possible structures of the `data/` folder in order to load the dataset correctly:


#### 1️⃣ Single Folder (Automatic Split)

If your dataset is **not pre-split** into training and testing sets, simply provide the dataset folder:

```bash
--data_folder_train your_dataset_folder
```

The dataset will be **automatically split** into **80% Training** and **20% Testing** data. The data used for training and testing is stored into `train_labels.csv` and `test_labels.csv` in the `models/[model_name]_[run]/` directory so that later on the test data can be used for further exploration of the model performance.

**Example folder structure:**
```
data/
├── dataset/
│   ├── data/               # contains image files (e.g., .jpg, .png)
│   │   ├── img0.jpg
│   │   ├── img1.png
│   │   └── ...            # more images
│   └── labels/
│       └── labels.csv      # CSV file in 'filename,label' format
```

**Example `labels.csv`:**

| filename   | label |
|------------|-------|
| img0.jpg   | 0     |
| img1.png   | 1     |
| ...        | ...   |

---

#### 2️⃣ Pre-split Dataset (Separate Train and Test Folders)

If your dataset is **already split** into separate training and testing folders, use:

```bash
--data_folder_train train_folder --data_folder_test test_folder
```

In this case, the pipeline will load **both datasets independently** and **no automatic splitting** will be applied.

**Example folder structure:**
```
data/
└── dataset/
    ├── train/
    │   ├── data/               # training images
    │   │   ├── img0.jpg
    │   │   ├── img1.png
    │   │   └── ...
    │   └── labels/
    │       └── labels.csv      # 'filename,label' format
    └── test/
        ├── data/               # testing images
        │   ├── img100.jpg
        │   ├── img101.png
        │   └── ...
        └── labels/
            └── labels.csv      # 'filename,label' format
```

---

> **Note:**  
> - The argument `--data_folder_train` is **always required**  
> - The argument `--data_folder_test` is **optional** if omitted, the training dataset will be automatically split (80/20).

---

### 📂 Explainability *(to be implemented)*
This folder will contain scripts for **model explainability and interpretability**:
- Layer-wise Relevance Propagation (LRP)

---
### 📂 Models
All trained model checkpoints are saved here. Checkpoints are organized by model name and run ID:

```
models/
├── resnet18_0/                         
│   ├── checkpoint_0_resnet18_best.pth  # First best checkpoint (e.g., after epoch 2)
│   └── checkpoint_1_resnet18_best.pth  # Later found better checkpoint (e.g., after epoch 5)
├── resnet18_1/                         
│   └── checkpoint_0_resnet18_best.pth  # First best checkpoint (no later improvement)
├── resnet18_2/
│   ├── checkpoint_0_resnet18_best.pth  # First best checkpoint
│   ├── checkpoint_1_resnet18_best.pth  # Improved checkpoint
│   └── checkpoint_2_resnet18_best.pth  # Further improved checkpoint
└── ...                                 # Further training runs
```

---

### Notebooks *(to be implemented)*
This folder is planned for **Jupyter notebooks** to:
- Visualize training results (accuracy/loss curves)
- Plot confusion matrices
- Analyze explainability outputs

---

### Training
Contains the main training script `train.py`, which handles:
- Dataset loading
- Model initialization
- Training and validation loops
- Checkpoint saving


## Example Usage
Go into the root directory `htcv_mgs/` and execute the following command:

```
python train.py --epochs 50 --batch_size 32 --model resnet18 --device cuda --data_folder [foldername_in_data_dir] --pretrained --shuffle
```
