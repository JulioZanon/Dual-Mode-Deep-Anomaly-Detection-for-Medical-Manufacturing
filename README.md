# Dual-Mode-Deep-Anomaly-Detection-for-Medical-Manufacturing
### Structural Similarity and Feature Distance

**Authors:** Julio Zanon Diaz, George Siogkas, and Peter Corcoran, Fellow, IEEE  

---

## 1. Overview

This repository contains the experimental framework, datasets, and evaluation scripts used in the study:  
> **“Dual-Mode Deep Anomaly Detection for Medical Manufacturing: Structural Similarity and Feature Distance.”**

The work introduces two complementary *attention-guided autoencoder architectures* designed for automated inspection in safety-critical medical-device manufacturing:
- **4-MS-SSIM Mode:** Real-time, interpretable inline defect detection using a multi-scale structural similarity index.
- **RFD-MD Mode:** Lightweight post-deployment drift and lifecycle monitoring using Mahalanobis distance of randomly reduced latent features.

Together, these modes provide a **dual-purpose framework** supporting both inline anomaly detection and supervisory lifecycle monitoring, aligned with forthcoming **EU AI Act** and **U.S. FDA QSR** requirements.

---

## 2. Abstract

Automated visual inspection in medical-device manufacturing faces challenges including extremely low defect rates, limited annotated data, hardware restrictions, and the need for explainable AI.  
This study presents a dual-mode deep anomaly detection framework—one mode based on **structural similarity** (4-MS-SSIM) and the other on **feature-space distance** (RFD-MD)—that enables both inline inspection and post-deployment process monitoring.  
Evaluations on the **Surface Seal Image (SSI)** dataset demonstrate superior performance compared with MOCCA, CPCAE, and RAG-PaDiM baselines, while maintaining real-time operation on single-GPU hardware.  
The approach advances explainable and auditable AI architectures for safety-critical automation systems.

---

## 3. Repository Structure
George to complete
```
├── README.md
├── /src
│   ├── /models
│   │   ├── autoencoder_attention.py
│   │   ├── dual_mode_ssim.py
│   │   ├── dual_mode_mahalanobis.py
│   ├── /training
│   │   ├── train_ssim_mode.py
│   │   ├── train_rfd_md_mode.py
│   ├── /evaluation
│   │   ├── evaluate_ssim_thresholds.py
│   │   ├── evaluate_mahalanobis_monitoring.py
│   └── /utils
│       ├── dataset_loader.py
│       ├── metrics.py
│       ├── attention_mask.py
│       ├── visualization_tools.py
│
├── /experiments
│   ├── SSI_dataset/
│   │   ├── training_config.json
│   │   ├── results_ssim.csv
│   │   ├── results_rfd_md.csv
│   ├── MVTec-Zipper/
│       ├── training_config.json
│       ├── results_cross_domain.csv
│
├── /appendix_tables
│   ├── Table4_autoencoder_optimization_step1.csv
│   ├── Table5_autoencoder_optimization_step2.csv
│   ├── Table6_ssim_variant_comparison.csv
│   ├── Table7_feature_distance_methods.csv
│   ├── Table8_random_selection_ablation.csv
│   ├── Table9_inference_time_ablation.csv
│
├── /notebooks
│   ├── SSI_experiments.ipynb
│   ├── MVTEC_crossdomain.ipynb
│
└── /docs
    ├── figures/
    │   ├── Fig1_dual_mode_architecture.png
    │   ├── Fig2_SSI_dataset.png
    │   ├── Fig3_MVTec_Zipper.png
    │   ├── Fig4_deployment_framework.png
    └── references.bib
```

---

## 4. Ablation Studies

**Table 4.1 (6) — Attention AE Anomaly Detection with Supervised Threshold**  
*Results obtained from the SSI test subset.*

| Model | AUC | ACC | P | R | F1 |
|:------|:----:|:----:|:----:|:----:|:----:|
| MSE | 0.881 | 0.708 | 0.931 | 0.450 | 0.607 |
| SSIM | 0.930 | 0.825 | 0.838 | 0.806 | 0.822 |
| PSNR | 0.866 | 0.744 | 0.855 | 0.589 | 0.697 |
| MAE | 0.915 | 0.781 | 0.885 | 0.644 | 0.746 |
| MS-SSIM | 0.922 | 0.808 | 0.840 | 0.761 | 0.799 |
| 4-SSIM | 0.969 | 0.867 | 0.817 | 0.944 | 0.876 |
| **4-MS-SSIM** | **0.977** | **0.931** | **0.938** | **0.922** | **0.930** |
| 4-G-SSIM | 0.530 | 0.500 | 0.500 | 0.028 | 0.053 |
| 4-MS-G-SSIM | 0.843 | 0.700 | 0.853 | 0.483 | 0.617 |

**Table 4.2 (7) — Assessment of Feature-Distance-Based Methods**  
*All features in covariance matrix trained on and reported from SSI cross-validation.*

| Model | AUC | ACC | P | R | F1 |
|:------|:----:|:----:|:----:|:----:|:----:|
| PCA–K-Means | 0.663 | 0.617 | 0.566 | 0.994 | 0.722 |
| PCA–Mahalanobis | 0.543 | 0.500 | 0.500 | 1.000 | 0.667 |
| K-Means | 0.446 | 0.519 | 0.516 | 0.633 | 0.569 |
| ICA–Mahalanobis | 0.463 | 0.492 | 0.200 | 0.006 | 0.011 |
| Random Drop – K-Means | 0.589 | 0.589 | 0.554 | 0.906 | 0.688 |
| **Random Drop – Mahalanobis** | **0.909** | **0.836** | **0.789** | **0.917** | **0.848** |


**Table 4.3 (8) — Feature Random Selection Ablation Study (Accuracy)**  
*Trained on and reported from SSI cross-validation subset.*

| Layer | 100 | 200 | 500 | 600 | 1000 | 1500 |
|:------|:----:|:----:|:----:|:----:|:----:|:----:|
| Run 1–Conv_1 | 0.525 | 0.522 | 0.850 | 0.856 | 0.853 | 0.847 |
| Run 1–Conv_2 | 0.497 | 0.497 | 0.861 | 0.861 | 0.869 | 0.831 |
| Run 1–Conv_3 | 0.506 | 0.533 | 0.797 | 0.797 | 0.792 | 0.817 |
| Run 1–Bottleneck | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.494 |
| Run 2–Conv_1 | 0.503 | 0.775 | 0.842 | 0.864 | 0.867 | --- |
| Run 2–Conv_2 | 0.519 | 0.519 | 0.850 | 0.811 | 0.819 | --- |
| Run 2–Conv_3 | 0.511 | 0.517 | 0.811 | 0.797 | 0.797 | --- |
| Run 2–Bottleneck | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | --- |
| Run 3–Conv_1 | 0.514 | 0.511 | 0.861 | 0.875 | 0.844 | --- |
| Run 3–Conv_2 | 0.514 | 0.511 | 0.831 | 0.864 | 0.847 | --- |
| Run 3–Conv_3 | 0.514 | 0.519 | 0.803 | 0.789 | 0.803 | --- |
| Run 3–Bottleneck | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | --- |
| **Avg–Conv_1** | **0.514** | **0.603** | **0.851** | **0.865** | **0.855** | --- |
| **Avg–Conv_2** | **0.510** | **0.509** | **0.847** | **0.845** | **0.845** | --- |
| **Avg–Conv_3** | **0.510** | **0.523** | **0.804** | **0.794** | **0.797** | --- |
| **Avg–Bottleneck** | **0.500** | **0.500** | **0.500** | **0.500** | **0.500** | --- |


**Table 4.4 (9) — Random Selection Ablation Study (Time to Test)**  

| Layer | 100 | 200 | 500 | 600 | 1000 |
|:------|:----:|:----:|:----:|:----:|:----:|
| Conv_1 | 19.4 ms | 34.6 ms | 49.6 ms | 52.3 ms | 134.2 ms |
| Conv_2 | 19.1 ms | 30.6 ms |**48.2 ms** | 51.9 ms | 131.9 ms |
| Conv_3 | 17.6 ms | 30.2 ms | 44.8 ms | 50.1 ms | 131.2 ms |
| Bottleneck | 11.5 ms | 14.3 ms | 12.4 ms | 12.2 ms | 12.0 ms |


---

## 5. Environment, Dependencies and Optimisation Parameters

All experiments were executed under single-GPU industrial constraints:

| Component | Specification |
|------------|---------------|
| GPU | NVIDIA RTX 4080 (16 GB) |
| CPU | Intel Core i7 |
| RAM | 64 GB |
| OS | Ubuntu 22.04 LTS |
| Framework | TensorFlow 2.19 / Python 3.10 |


**Autoencoder Architecture Optimisation (Step 1)**  
*(Bold denotes best-performing parameters locked for next step)*

| Parameter Name | Fix parameter or Search Range |
|---|---|
| Encoder filters per layer and number of layers | [(32, 64, 96),<br>(32, 64, 128),<br>**(32, 64, 96, 128)**,<br>**(16, 32, 64, 128)**,<br>(16, 32, 96, 128, 160),<br>(16, 32, 96, 128, 160, 192),<br>(32, 64, 128, 256),<br>(32, 64, 128, 256, 512)] |
| Learning Rate | 0.00001 to 0.01 |
| Number of Epochs | 20 to **100** |
| Optimiser Function | [**Adam**, SGD, RMSProp] |
| Loss function | [mse, mae, ssim] |
| Dropout rate | (0.0, 0.5) (uniform) \**[0.11228]** |

**Autoencoder Architecture Optimisation (Step 2)**  
*(Bold denotes best-performing parameters selected for the final autoencoder)*

| Parameter Name | Fix parameter or Search Range |
|---|---|
| Encoder filters | [(16, 32, 64, 128),<br>(32, 64, 128, 256)] (categorical indices) |
| Batch size | [16, 32, 64, 128] (categorical) |
| Learning Rate | 0.001 to 0.01 (uniform) |
| Optimiser Function | Adam (fixed) |
| Optimiser Parameters | β₁: (0.90 to 0.99) (uniform)<br>β₂: (0.9990 to 0.9999) (uniform)<br>epsilon: (1e-8 to 1e-5) (uniform) **[1e-7]** |
| Loss function | [mse, mae, ssim] (categorical) |
| Dropout rate | **0.11228** (fixed) |


### Python Environment Example
```bash
conda create -n dualmode python=3.10
conda activate dualmode
pip install tensorflow==2.19 numpy==1.26 opencv-python matplotlib==3.8 scikit-learn==1.5 tqdm==4.66
```

---

## 6. Reproducibility and Validation

- All experiments repeated **3× with fixed random seeds**; results reported as *mean ± standard deviation*.
- Deterministic thresholding (`μ + 2σ`) for unsupervised mode ensures **traceability**.
- Hardware and configuration fixed per ISO 13485 validation principles.
- Datasets:  
  - **Surface Seal Image (SSI)** – sterile-barrier packaging inspection (public dataset [3]).  
  - **MVTec-Zipper** – benchmark dataset for cross-domain evaluation [4].

---

## 7. Citation

If you use this repository, please cite:

```bibtex
@article{ZanonDiaz2025DualMode,
  author = {Julio Zanon Diaz and George Siogkas and Peter Corcoran},
  title = {Dual-Mode Deep Anomaly Detection for Medical Manufacturing: Structural Similarity and Feature Distance},
  journal = {IEEE Transactions on Automation Science and Engineering},
  year = {2025}
}
```

---

## 8. Contact

For academic or industrial collaboration:

- **Julio Zanon Diaz** – Boston Scientific / University of Galway  
- **Dataset:** [Surface Seal Image (SSI) Dataset – ScienceDirect DOI: 10.1016/j.dib.2024.110996](https://doi.org/10.1016/j.dib.2024.110996)

