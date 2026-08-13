# DUAI-Net

**DUAI-Net: Dynamic Uncertainty-Aware Interaction Network for Multimodal Emotion Recognition in Conversation**

This repository provides the PyTorch implementation of **DUAI-Net** for Multimodal Emotion Recognition in Conversation (MERC).

## Architecture

DUAI-Net dynamically organizes contextual relations and multimodal evidence according to the current conversational state and prediction reliability. It consists of three key components:

- **Dynamic Composition Transformer (DCTransformer)** dynamically combines multi-head attention responses for audio and visual context encoding.
- **Dynamic Relational Propagation (DRP)** performs two-stage relational propagation and regulates source contributions according to utterance-level predictive confidence.
- **Dynamic Auxiliary Residual Fusion (DARF)** retrieves complementary audio-visual evidence using the textual relational representation and injects it through relevance- and uncertainty-aware residual fusion.

## Requirements

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Data Preparation

The code uses pre-extracted multimodal features for **IEMOCAP** and **MELD**.

By default, place the feature files at:

```text
./data/iemocap_multimodal_features.pkl
./data/meld_multimodal_features.pkl
```

You can also specify another feature path through `--data-path`.

## Training

For IEMOCAP:

```bash
bash run_iemocap.sh
```

For MELD:

```bash
bash run_meld.sh
```

## Results

| Dataset | Weighted Accuracy | Weighted F1 |
|---|---:|---:|
| IEMOCAP | 76.71 | 76.74 |
| MELD | 67.09 | 66.30 |

## Citation

Citation information will be updated after publication.
