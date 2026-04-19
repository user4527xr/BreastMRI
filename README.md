# Vision-Language Pre-trained Model for Breast MRI Interpretation

Official code release for the paper:

**Vision-language Pre-trained Model for Breast MRI Interpretation: A Large-scale, Multicenter Model Development and Prospective Validation Study**

This repository contains two main components for breast MRI interpretation:

1. **Breast_diagnosis**: multimodal malignancy diagnosis based on multiparametric breast MRI.
2. **Breast_RG**: automated report generation from multiparametric breast MRI.

## Overview

Interpreting multiparametric breast MRI requires integrating information from multiple sequences, including **T2-weighted imaging (T2WI)**, **diffusion-weighted imaging (DWI)**, and **dynamic contrast-enhanced MRI (DCE-MRI)**. This repository provides code for two clinically relevant tasks:

- **Malignancy diagnosis**
- **Automated report generation**

The study was developed and validated on a large multicenter dataset and evaluated with **internal**, **prospective**, and **external** cohorts.

## Repository Structure

```text
.
├── Breast_diagnosis/
│   ├── configs/
│   ├── data/
│   ├── data_csv/
│   ├── module/
│   ├── scripts/
│   ├── requirements.txt
│   └── README.md
├── Breast_RG/
│   ├── configs/
│   ├── dataset/
│   ├── evalcap/
│   ├── datacsv/
│   ├── lightning_tools/
│   ├── models/
│   ├── scripts/
│   ├── train.py
│   └── save/
└── README.md
```

## Released Resources

### 1. Breast Diagnosis

- **Code directory**: `Breast_diagnosis/`
- **Task**: benign / malignant prediction from multiparametric breast MRI
- **Released resources**:
  - diagnosis model checkpoint
  - diagnosis `data_csv` examples
- **Baidu Netdisk**: [Baidu Netdisk Link](https://pan.baidu.com/s/1W9WaIdl4hdiOQIX8W75M7w?pwd=8nag 提取码: 8nag)


### 2. Breast Report Generation

- **Code directory**: `Breast_RG/`
- **Task**: automated breast MRI report generation
- **Released resources**:
  - report generation model checkpoint
  - report-generation `datacsv` examples
- **Baidu Netdisk**: [Baidu Netdisk Link](https://pan.baidu.com/s/1VivDEA8COctru5Z3YspzzA?pwd=skei 提取码: skei)


## Environment Setup

We recommend **Python 3.10** and **PyTorch with CUDA support**.

### Diagnosis environment

```bash
cd Breast_diagnosis
pip install -r requirements.txt
```

### Report-generation environment

The report-generation code depends on PyTorch Lightning, Transformers, NumPy, Pandas, Pillow, and related libraries. A typical setup is:

```bash
cd Breast_RG
pip install torch torchvision
pip install lightning transformers pandas numpy pillow scikit-learn tqdm pyyaml
```

## Data Preparation

### 1. Diagnosis (`Breast_diagnosis`)

The diagnosis code expects a CSV file containing three MRI inputs:

- `T2`: path to the T2 `.npy` volume
- `DWI`: path to the DWI `.npy` volume
- `SUB_concate`: path to the DCE subtraction `.npy` volume
- `Subject` (optional): case identifier
- `malignant` (optional for inference): ground-truth label

An example CSV file is expected at:

```text
Breast_diagnosis/data_csv/example_test.csv
```

Before running inference, please edit:

```text
Breast_diagnosis/configs/diagnosis_inference.yaml
```

and update the following fields:

- `data.csv_path`
- `data.data_root`
- `model.checkpoint_path`
- `runtime.device`

### 2. Report Generation (`Breast_RG`)

The report-generation code uses metadata files stored in the `datacsv` package provided through Baidu Netdisk. In the current release, the expected metadata fields include:

- `T2`: path to the T2 `.npy` volume
- `DWI`: path to the DWI `.npy` volume
- `SUB_concate`: path to the DCE subtraction `.npy` volume
- `report`: report text
- `Subject`: case identifier
- `malignant`: diagnosis label

The public examples are expected to include files such as:

```text
datacsv/example_test.csv
datacsv/example_data/
```

```

## Quick Start

### A. Malignancy Diagnosis Inference

```bash
cd Breast_diagnosis
python scripts/run_diagnosis_inference.py --config configs/diagnosis_inference.yaml
```

Outputs will be written to the folder specified by `runtime.output_dir`, typically including:

- `predictions.csv`
- `metrics.json` (if labels are available)

### B. Report Generation Testing

```bash
cd Breast_RG
bash scripts/3-2.deep_test_ds1_ours.sh
```

Before running the test script, please make sure you have updated:

- the checkpoint path (`delta_file`)
- the metadata / data root paths
- the save path if needed

