import os

import matplotlib.pyplot as plt
import seaborn as sns


def confuPLT(confusion, dataset, save_dir="./fig"):
    """Plot and save the confusion matrix for IEMOCAP or MELD."""
    if dataset == "IEMOCAP":
        class_names = [
            "happy",
            "sadness",
            "neutral",
            "anger",
            "excited",
            "frustrated",
        ]
    elif dataset == "MELD":
        class_names = [
            "neutral",
            "surprise",
            "fear",
            "sadness",
            "joy",
            "disgust",
            "anger",
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(9, 7))
    sns.set(font_scale=1.2)
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{dataset} Confusion Matrix")
    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        f"{dataset.lower()}_confusion_matrix.png",
    )
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Confusion matrix saved to: {save_path}")
