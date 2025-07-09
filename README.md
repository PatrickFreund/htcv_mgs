# HTCV - Fully Supervised MGS (Mouse Grimace Scale)
This repository provides a modular and configurable training and evaluation pipeline for image classification using Convolutional Neural Networks (CNNs) in PyTorch.
While originally developed for pain classification based on the Mouse Grimace Scale (MGS) using ResNet18, the pipeline is easily adaptable to other grayscale image classification tasks.

The pipeline supports:
- Configurable model training
- Grid search and Optuna-based hyperparameter optimization
- Explainability via LRP (Layer-wise Relevance Propagation)
- Group-wise evaluation (e.g., per strain or subject)



## Directory Structure 

```
htcv_mgs/

├── configs/                     # Static configs and constants
├── datamodule/                  # Datasets, transforms, splits
├── training/                    # Trainer, evaluator, balancing, logging
├── search/                      # GridSearch and Optuna search logic
├── utils/                       # Model and optimizer builders, helpers
├── explainability/              # LRP module (external integration)
├── results/                     # Training logs, metrics, model checkpoints
├── data/                        # Input datasets and label files
|
├── requirements-cpu.txt         # CPU requirements for the repository
├── requirements-gpu.txt         # GPU requirements for the repository
├── GridSearchDemo.ipynb         # Example notebook for grid search
└── OptunaSearchDemo.ipynb       # Example notebook for Optuna hyperparameter optimization
```


## Getting Started 

### Installation 
Set up a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Then isntall the requirements
pip install -r requirements-cpu.txt  # For CPU-only usage
# or
pip install -r requirements-gpu.txt  # For GPU usage
```

### Example Usage
Instead of a monolithic CLI script, usage is demonstrated through Jupyter notebooks:
Notebook | Description
--- | ---
notebooks/TrainDemo.ipynb | Run a training session with a ResNet18 model on MGS data
notebooks/ExplainabilityDemo.ipynb | Run explainability analysis using LRP on trained models
<br>
Each notebook walks through:
1. Definition of the hyperparameter search space
2. Configuration of training specifics (e.g., balancing, early stopping, etc.)
3. Preparation of the split strategy for the dataset
4. Preparation of the dataset and transforms 
5. Execution of the hyperparameter search 
6. Evaluation of the best model via
    - Group-wise evaluation
    - Explainability analysis (LRP)
    - ...


## Data structure
Detailed information about the data structure can be found in the demo notebooks or in datamodule/dataset.py however generally the data structure is as follows:

```
data/
└── dataset_name/
    ├── data/                    # grayscale image files (e.g. .jpg)
    └── labels/
        └── labels.csv           # must contain: filename,label
```

Example labels.csv:
```
filename,label
img1.jpg,0
img2.jpg,1
img3.jpg,0
img4.jpg,1
``` 

Presplitting is not possible, since the pipeline will handle it automatically. The dataset will be split into training and validation sets based on the provided split strategy (e.g., KFold, RandomSplit) and the defined number of splits.


## Results & Logs
All training runs are saved in:
```
results/
└── experiment_name/
    ├─ run_name/
    |    ├── fold_0/
    |    │   ├── best_model.pth
    |    │   └── events.out.tfevents...
    |    ├── fold_1/
    |    │   ├── best_model.pth
    |    │   └── events.out.tfevents...
    |    ├── fold_2/
    |    │   ├── best_model.pth
    |    │   └── events.out.tfevents...
    |    ├── config.yaml
    |    └── hparams_summary.csv
    |
    └── config.yaml
```
The `best_model.pth` files contain the best model weights for each fold, while the `events.out.tfevents...` files contain TensorBoard logs for visualization of training metrics. The `config.yaml` file contains the configuration used in the run, and `hparams_summary.csv` contains a summary of hyperparameters and their corresponding performance metrics. The `config.yaml` on the top level contains the overall search space and training configuration used for the experiment.


## Explainability
Integrated Layer-wise Relevance Propagation (LRP) for resnet18 with grayscale inputs.
Current limitations:
- Only resnet18 supported (due to external LRP structure)
- Only grayscale images (single channel)
- Requires ROI /segmentation masks for relevance-in-ROI analysis