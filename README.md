# DUAI-Net

Official implementation of **DUAI-Net: Dynamic Uncertainty-Aware Interaction Network for Multimodal Emotion Recognition in Conversation**.

## I. Environment Configuration

### 1. Prerequisites

- Anaconda or Miniconda
- NVIDIA GPU with CUDA support
- DUAI-Net project code
- Pre-extracted multimodal features for:
  - IEMOCAP
  - MELD

Place the feature files in the `data` folder:

```text
data/
├── iemocap_multimodal_features.pkl
└── meld_multimodal_features.pkl
```

### 2. Operation Steps

```bash
# 1. Create and activate the conda environment
conda create -n DUAI-Net python=3.10 -y
conda activate DUAI-Net

# 2. Install dependencies
pip install -r requirements.txt

# 3. Navigate to the project directory
cd DUAI-Net

# 4. Grant execution permission to the scripts
chmod +x run_iemocap.sh
chmod +x run_meld.sh
```

## II. Run Experiments

```bash
# IEMOCAP
./run_iemocap.sh

# MELD
./run_meld.sh
```

