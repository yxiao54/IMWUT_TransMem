# TransMemNet: Resilience-Guided Physiological Craving Detection

This repository contains the implementation for **TransMemNet**, a resilience-guided framework for physiological craving detection under subject-independent evaluation. The code supports training and evaluation of the proposed model, ablation variants, and baseline methods using a unified training pipeline.

This anonymized repository is provided to support reproducibility during double-blind review.

---

## Repository Structure

```
├── engine/           # Training and evaluation engine
│   └── engine.py
├── losses/           # Loss functions (standard and DG objectives)
│   ├── factory.py
│   └── *.py
├── models/           # Model definitions (proposed, ablations, baselines)
│   ├── factory.py
│   └── *.py
├── runners/          # Cross-validation and experiment runners
│   └── run_cv.py   
├── feature/
│   └── feature.py    # Feature extraction code
├── Stat_Analysis/ # Statistical analysis scripts
│   ├── RQ1.py        # Analysis code for Research Question 1
│   └── RQ2.py        # Analysis code for Research Question 2
└── README.md
└── main.py           # Main entry point for training and evaluation
```
**Note:** The `dataloaders/` and `data/` directories are not included in this anonymized repository due to data privacy and human-subject protection requirements.

---

## Main Entry Point

All experiments are executed via:

python main.py

The main script performs the following steps:

- Sets global random seeds for reproducibility  
- Loads physiological features and participant-level language embeddings  
- Instantiates the selected model architecture  
- Runs one fold of subject-independent cross-validation  
- Logs metrics and predictions  

---

## Models

The `models/` directory includes implementations of:

- **Proposed model**
  - TransMemNet (resilience-guided physiological inference)

- **Ablation models**
  - Variants that selectively remove or replace components (e.g., feature gating, decision sensitivity, stress anchoring)

- **Baseline models**
  - Physiology-only models  
  - Multimodal fusion baselines
  - Robustness and domain generalization baselines  

All models are instantiated through a unified factory interface (`models/factory.py`) to ensure consistent training and evaluation.

---

## Statistical Analysis

The `Stat_Analysis/` directory contains standalone scripts used to reproduce the statistical analyses reported for Research Questions 1 and 2 in the paper. These scripts implement the feature-level and system-level physiological analyses described in the manuscript and are independent of the model training pipeline.

---

## Training and Evaluation

Training and evaluation are handled by a shared `Engine` class (`engine/engine.py`), which provides:

- Unified training loops  
- Gradient clipping and optimization  
- Consistent metric computation (accuracy, balanced accuracy, F1, AUC)  
- Support for environment-aware losses using subject identifiers  

All experiments follow **subject-independent cross-validation**, ensuring no overlap of participants between training and testing.

---

## Command-Line Arguments

Key arguments supported by `scripts/main.py` include:

- `--backbone` : model type (proposed, ablation, or baseline)  
- `--label_type` : prediction target (`stress` or `craving`)  
- `--embedding_mode` : language embedding variant  
- `--fold_idx` : cross-validation fold index  
- `--normalizer` : feature normalization strategy  
- `--groups` : number of feature groups for gated models  
- `--hidden` : hidden dimension size  
- `--lambda_sparse` : sparsity regularization weight  
- `--seed` : random seed for reproducibility  

---





## Notes

This repository is anonymized for double-blind review. All identifiers, paths, and metadata have been scrubbed to preserve reviewer anonymity.
