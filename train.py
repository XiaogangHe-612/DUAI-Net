import argparse
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataloader import IEMOCAPDataset, MELDDataset
from model import DUAINet, MaskedNLLLoss


# =========================================================
# Reproducibility
# =========================================================
def seed_everything(seed):
    print(f"seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# =========================================================
# Data
# =========================================================
def build_full_train_sampler(dataset):
    """Use the complete training split, consistent with the experimental protocol."""
    return SubsetRandomSampler(list(range(len(dataset))))


def get_iemocap_loaders(data_path, batch_size=16, num_workers=0, pin_memory=True, windows=20):
    trainset = IEMOCAPDataset(data_path, windows=windows, train=True)
    testset = IEMOCAPDataset(data_path, windows=windows, train=False)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=build_full_train_sampler(trainset),
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=testset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def get_meld_loaders(data_path, batch_size=16, num_workers=0, pin_memory=True, windows=5):
    trainset = MELDDataset(data_path, windows=windows, train=True)
    testset = MELDDataset(data_path, windows=windows, train=False)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=build_full_train_sampler(trainset),
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=testset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


# =========================================================
# DUAI-Net controls and statistics
# =========================================================
def get_base_model(model):
    return model.module if hasattr(model, "module") else model


def set_drp_uncertainty_enabled(model, enabled):
    base_model = get_base_model(model)
    if hasattr(base_model, "set_drp_uncertainty"):
        base_model.set_drp_uncertainty(enabled)


def set_darf_enabled(model, enabled):
    base_model = get_base_model(model)
    if hasattr(base_model, "set_darf"):
        base_model.set_darf(enabled)


def collect_darf_stats(base_model, umask, stat_sums):
    """Collect mean DARF gates and fusion coefficients over valid utterances."""
    if not hasattr(base_model, "last_audio_gate"):
        return

    audio_gate = base_model.last_audio_gate
    visual_gate = base_model.last_visual_gate
    audio_coeff = base_model.last_audio_fusion_coeff
    visual_coeff = base_model.last_visual_fusion_coeff

    if audio_gate is None or visual_gate is None:
        return

    valid = umask.unsqueeze(-1).float()
    denom = valid.sum().item()
    if denom <= 0:
        return

    stat_sums["audio_gate_sum"] += (audio_gate * valid).sum().item()
    stat_sums["visual_gate_sum"] += (visual_gate * valid).sum().item()
    stat_sums["count"] += denom

    if audio_coeff is not None:
        stat_sums["audio_coeff_sum"] += float(audio_coeff.item() if torch.is_tensor(audio_coeff) else audio_coeff)
        stat_sums["audio_coeff_count"] += 1

    if visual_coeff is not None:
        stat_sums["visual_coeff_sum"] += float(
            visual_coeff.item() if torch.is_tensor(visual_coeff) else visual_coeff
        )
        stat_sums["visual_coeff_count"] += 1


def finalize_darf_stats(stat_sums):
    count = stat_sums["count"]

    return {
        "avg_audio_gate": stat_sums["audio_gate_sum"] / count if count > 0 else None,
        "avg_visual_gate": stat_sums["visual_gate_sum"] / count if count > 0 else None,
        "audio_fusion_coeff": (
            stat_sums["audio_coeff_sum"] / stat_sums["audio_coeff_count"]
            if stat_sums["audio_coeff_count"] > 0
            else None
        ),
        "visual_fusion_coeff": (
            stat_sums["visual_coeff_sum"] / stat_sums["visual_coeff_count"]
            if stat_sums["visual_coeff_count"] > 0
            else None
        ),
    }


# =========================================================
# Training / Evaluation
# =========================================================
def train_or_eval_model(
    model,
    loss_function,
    dataloader,
    device,
    optimizer=None,
    train=False,
    stage1_aux_weight=0.10,
):
    assert not train or optimizer is not None

    model.train() if train else model.eval()

    losses, preds, labels, masks = [], [], [], []
    darf_stat_sums = {
        "audio_gate_sum": 0.0,
        "visual_gate_sum": 0.0,
        "count": 0.0,
        "audio_coeff_sum": 0.0,
        "audio_coeff_count": 0,
        "visual_coeff_sum": 0.0,
        "visual_coeff_count": 0,
    }

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        (
            text_feat,
            visual_feat,
            audio_feat,
            qmask,
            umask,
            label,
            self_semantic_adj,
            cross_semantic_adj,
            semantic_adj,
        ) = [d.to(device) for d in data]

        qmask = qmask.permute(1, 0, 2)
        lengths = [
            int((umask[j] == 1).nonzero(as_tuple=False)[-1][0].item() + 1)
            for j in range(len(umask))
        ]

        with torch.set_grad_enabled(train):
            (
                modality_log_probs,
                fused_log_prob,
                fused_prob,
                _,
                stage1_aux_log_prob,
            ) = model(
                text_feat,
                visual_feat,
                audio_feat,
                umask,
                qmask,
                lengths,
                self_semantic_adj,
                cross_semantic_adj,
                semantic_adj,
            )

            labels_flat = label.view(-1)
            fused_loss = loss_function(
                fused_log_prob.view(-1, fused_log_prob.size(-1)),
                labels_flat,
                umask,
            )

            if len(modality_log_probs) == 3:
                text_loss = loss_function(
                    modality_log_probs[0].view(-1, modality_log_probs[0].size(-1)),
                    labels_flat,
                    umask,
                )
                audio_loss = loss_function(
                    modality_log_probs[1].view(-1, modality_log_probs[1].size(-1)),
                    labels_flat,
                    umask,
                )
                visual_loss = loss_function(
                    modality_log_probs[2].view(-1, modality_log_probs[2].size(-1)),
                    labels_flat,
                    umask,
                )
                loss = (
                    fused_loss
                    + text_loss * (text_loss / 10)
                    + audio_loss * (audio_loss / 10)
                    + visual_loss * (visual_loss / 10)
                )
            else:
                text_loss = loss_function(
                    modality_log_probs[0].view(-1, modality_log_probs[0].size(-1)),
                    labels_flat,
                    umask,
                )
                loss = fused_loss + text_loss * (text_loss / 10)

            stage1_aux_loss = loss_function(
                stage1_aux_log_prob.view(-1, stage1_aux_log_prob.size(-1)),
                labels_flat,
                umask,
            )
            loss = loss + stage1_aux_weight * stage1_aux_loss

            if train:
                loss.backward()
                optimizer.step()

        pred_flat = torch.argmax(fused_prob.view(-1, fused_prob.size(-1)), dim=1)

        preds.append(pred_flat.detach().cpu().numpy())
        labels.append(labels_flat.detach().cpu().numpy())
        masks.append(umask.view(-1).detach().cpu().numpy())
        losses.append(loss.item() * masks[-1].sum())

        collect_darf_stats(get_base_model(model), umask, darf_stat_sums)

    if not preds:
        return float("nan"), float("nan"), [], [], [], float("nan"), finalize_darf_stats(darf_stat_sums)

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_wf1 = round(f1_score(labels, preds, sample_weight=masks, average="weighted") * 100, 2)

    return (
        avg_loss,
        avg_accuracy,
        labels,
        preds,
        masks,
        avg_wf1,
        finalize_darf_stats(darf_stat_sums),
    )


# =========================================================
# Configuration
# =========================================================
def build_parser():
    parser = argparse.ArgumentParser(description="Train DUAI-Net for MERC.")

    parser.add_argument("--dataset", type=str, default="IEMOCAP", choices=["IEMOCAP", "MELD"])
    parser.add_argument("--data-path", type=str, default=None, help="Path to the pre-extracted feature file.")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory for the best checkpoint.")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA.")
    parser.add_argument("--seed", type=int, default=2094)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--l2", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--no-class-weight", action="store_true")

    # DRP
    parser.add_argument("--drp-warmup-epochs", type=int, default=0)
    parser.add_argument("--stage1-aux-weight", type=float, default=0.10)
    parser.add_argument("--edge-scale-gamma", type=float, default=0.45)
    parser.add_argument("--edge-scale-alpha", type=float, default=4.0)
    parser.add_argument("--edge-scale-beta", type=float, default=0.50)
    parser.add_argument("--edge-logit-scale", type=float, default=0.45)

    # DCTransformer
    parser.add_argument("--dc-rank", type=int, default=2)

    # DARF
    parser.add_argument("--darf-warmup-epochs", type=int, default=0)
    parser.add_argument("--fusion-rank", type=int, default=1)
    parser.add_argument("--audio-fusion-coeff", type=float, default=0.09)
    parser.add_argument("--visual-fusion-coeff", type=float, default=0.01)
    parser.add_argument("--darf-mode", type=str, default="a", choices=["a", "v", "av"])

    parser.add_argument(
        "--plot-confusion",
        action="store_true",
        help="Plot the confusion matrix after training (requires vision.py).",
    )
    return parser


def resolve_data_path(args):
    if args.data_path is not None:
        return args.data_path

    default_paths = {
        "IEMOCAP": "./data/iemocap_multimodal_features.pkl",
        "MELD": "./data/meld_multimodal_features.pkl",
    }
    return default_paths[args.dataset]


def build_loss_function(dataset, use_class_weight, device):
    if not use_class_weight:
        return MaskedNLLLoss()

    if dataset == "IEMOCAP":
        weights = torch.FloatTensor(
            [
                1 / 0.086747,
                1 / 0.144406,
                1 / 0.227883,
                1 / 0.160585,
                1 / 0.127711,
                1 / 0.252668,
            ]
        )
    else:
        weights = torch.FloatTensor(
            [
                1 / 0.481226,
                1 / 0.107663,
                1 / 0.191571,
                1 / 0.079669,
                1 / 0.154023,
                1 / 0.026054,
                1 / 0.132184,
            ]
        )

    return MaskedNLLLoss(weights.to(device))


# =========================================================
# Main
# =========================================================
def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(args)
    print(f"Running on {device}")

    data_path = resolve_data_path(args)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Feature file not found: {data_path}\n"
            "Use --data-path to specify the pre-extracted IEMOCAP or MELD feature file."
        )

    feature_dims = {
        "text": 1024,
        "visual": 342,
        "IEMOCAP_audio": 1582,
        "MELD_audio": 300,
    }
    d_text = feature_dims["text"]
    d_visual = feature_dims["visual"]
    d_audio = feature_dims[f"{args.dataset}_audio"]
    n_speakers = 2 if args.dataset == "IEMOCAP" else 9
    n_classes = 6 if args.dataset == "IEMOCAP" else 7

    model = DUAINet(
        dataset=args.dataset,
        D_text=d_text,
        D_visual=d_visual,
        D_audio=d_audio,
        n_head=args.n_head,
        n_classes=n_classes,
        hidden_dim=args.hidden_dim,
        n_speakers=n_speakers,
        dropout=args.dropout,
        use_drp_uncertainty=True,
        uncertainty_detach=True,
        edge_scale_gamma=args.edge_scale_gamma,
        edge_scale_alpha=args.edge_scale_alpha,
        edge_scale_beta=args.edge_scale_beta,
        edge_logit_scale=args.edge_logit_scale,
        use_darf=True,
        fusion_rank=args.fusion_rank,
        fusion_coeff=args.audio_fusion_coeff,
        visual_fusion_coeff=args.visual_fusion_coeff,
        fusion_mode=args.darf_mode,
        dc_rank=args.dc_rank,
        # Final DCTransformer configuration used by DUAI-Net.
        use_standard_mha=False,
        dc_use_qk_norm=True,
        dc_use_pre=False,
        dc_use_key_branch=False,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total parameters: {total_params}")
    print(f"training parameters: {trainable_params}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    loss_function = build_loss_function(
        args.dataset,
        use_class_weight=not args.no_class_weight,
        device=device,
    )

    pin_memory = device.type == "cuda"
    if args.dataset == "IEMOCAP":
        train_loader, test_loader = get_iemocap_loaders(
            data_path=data_path,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=pin_memory,
            windows=args.windows,
        )
    else:
        train_loader, test_loader = get_meld_loaders(
            data_path=data_path,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=pin_memory,
            windows=args.windows,
        )

    best_wf1 = None
    best_acc = None
    best_loss = None
    best_epoch = None
    best_labels = None
    best_preds = None
    best_mask = None
    best_state_dict = None

    for epoch in range(args.epochs):
        start_time = time.time()

        drp_uncertainty_on = epoch >= args.drp_warmup_epochs
        darf_on = epoch >= args.darf_warmup_epochs
        set_drp_uncertainty_enabled(model, drp_uncertainty_on)
        set_darf_enabled(model, darf_on)

        train_loss, train_acc, _, _, _, train_wf1, _ = train_or_eval_model(
            model=model,
            loss_function=loss_function,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            train=True,
            stage1_aux_weight=args.stage1_aux_weight,
        )

        test_loss, test_acc, test_labels, test_preds, test_mask, test_wf1, test_darf_stats = (
            train_or_eval_model(
                model=model,
                loss_function=loss_function,
                dataloader=test_loader,
                device=device,
                train=False,
                stage1_aux_weight=args.stage1_aux_weight,
            )
        )

        if best_wf1 is None or test_wf1 > best_wf1:
            best_wf1 = test_wf1
            best_acc = test_acc
            best_loss = test_loss
            best_epoch = epoch + 1
            best_labels = test_labels.copy()
            best_preds = test_preds.copy()
            best_mask = test_mask.copy()
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in get_base_model(model).state_dict().items()
            }

        def fmt_stat(value):
            return "None" if value is None else f"{value:.4f}"

        print(
            f"epoch: {epoch + 1}, "
            f"drp_uncertainty_on: {drp_uncertainty_on}, darf_on: {darf_on}, "
            f"train_loss: {train_loss}, train_acc: {train_acc}, train_wf1: {train_wf1}, "
            f"test_loss: {test_loss}, test_acc: {test_acc}, test_wf1: {test_wf1}, "
            f"audio_coeff: {fmt_stat(test_darf_stats['audio_fusion_coeff'])}, "
            f"visual_coeff: {fmt_stat(test_darf_stats['visual_fusion_coeff'])}, "
            f"audio_gate: {fmt_stat(test_darf_stats['avg_audio_gate'])}, "
            f"visual_gate: {fmt_stat(test_darf_stats['avg_visual_gate'])}, "
            f"time: {time.time() - start_time:.2f} sec"
        )

    save_dir = args.save_dir or f"./{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")
    torch.save(best_state_dict, save_path)

    print("\nModel Performance:")
    print(f"Best Test Epoch: {best_epoch}")
    print(f"Best Test W-F1: {best_wf1}")
    print(f"Best Test Accuracy: {best_acc}")
    print(f"Best Test Loss: {best_loss}")
    print(
        classification_report(
            best_labels,
            best_preds,
            sample_weight=best_mask,
            digits=4,
            zero_division=0,
        )
    )
    print(f"Best checkpoint saved to: {save_path}")

    if args.plot_confusion:
        from vision import confuPLT

        confuPLT(
            confusion_matrix(
                best_labels,
                best_preds,
                sample_weight=best_mask,
            ).astype(int),
            args.dataset,
        )


if __name__ == "__main__":
    main()
