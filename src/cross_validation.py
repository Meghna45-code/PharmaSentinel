import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

class CrossValidationEngine:
    def __init__(self, data, n_splits=5, seed=42):
        self.data = data
        self.n_splits = n_splits
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def run_5fold_cv(self, model_cls, trainer_cls, in_dim=1024, embedding_dim=256, epochs_per_fold=10, lr=0.008):
        num_nodes = self.data["num_nodes"]
        num_drugs = self.data["num_drugs"]
        num_relations = self.data["num_relations"]
        splits = self.data["splits"]

        # Combine all edges for 5-fold CV
        all_edge_index = torch.cat([
            splits["train_edge_index"],
            splits["val_edge_index"],
            splits["test_edge_index"]
        ], dim=1)

        all_edge_type = torch.cat([
            splits["train_edge_type"],
            splits["val_edge_type"],
            splits["test_edge_type"]
        ], dim=0)

        num_total_edges = all_edge_index.shape[1]
        perm = np.random.permutation(num_total_edges)

        fold_size = num_total_edges // self.n_splits
        fold_results = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n==========================================================================")
        print(f"   STARTING RIGOROUS {self.n_splits}-FOLD CROSS-VALIDATION (100% DATA COVERAGE)")
        print(f"==========================================================================\n")
        print(f"  Total Graph Edges evaluated across 5 folds: {num_total_edges:,}")
        print(f"  Edges per Fold (20% Test): {fold_size:,} | Training Edges (80%): {num_total_edges - fold_size:,}\n")

        oof_y_true = []
        oof_y_scores = []

        for fold in range(self.n_splits):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else num_total_edges
            
            val_indices = perm[val_start:val_end]
            train_indices = np.concatenate([perm[:val_start], perm[val_end:]])

            train_edge_index = all_edge_index[:, train_indices]
            train_edge_type = all_edge_type[train_indices]

            val_edge_index = all_edge_index[:, val_indices]
            val_edge_type = all_edge_type[val_indices]

            print(f"--- Running Fold {fold + 1} / {self.n_splits} ---")
            print(f"  Train Graph Edges: {train_edge_index.shape[1]:,} | Test Graph Edges: {val_edge_index.shape[1]:,}")

            # Instantiate fresh model for this fold
            model = model_cls(num_nodes=num_nodes, num_relations=num_relations, in_dim=in_dim, embedding_dim=embedding_dim)
            trainer = trainer_cls(model=model, num_nodes=num_nodes, num_drugs=num_drugs, lr=lr, device=device)

            for epoch in range(1, epochs_per_fold + 1):
                loss = trainer.train_epoch(train_edge_index, train_edge_type, batch_size=131072)

            # Strict Out-of-Fold Evaluation
            auroc, auprc, y_true, y_scores = trainer.evaluate(val_edge_index, val_edge_type, max_samples=150000)
            
            # Binary predictions at threshold 0.5
            y_pred = (y_scores >= 0.5).astype(int)
            f1 = f1_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)

            oof_y_true.extend(y_true)
            oof_y_scores.extend(y_scores)

            fold_res = {
                "fold": fold + 1,
                "auroc": float(auroc),
                "auprc": float(auprc),
                "f1_score": float(f1),
                "precision": float(prec),
                "recall": float(rec)
            }
            fold_results.append(fold_res)

            print(f"  [FOLD {fold + 1} RESULTS] AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}\n")

        # Global Out-of-Fold Summary
        oof_y_true = np.array(oof_y_true)
        oof_y_scores = np.array(oof_y_scores)
        oof_y_pred = (oof_y_scores >= 0.5).astype(int)

        global_auroc = float(roc_auc_score(oof_y_true, oof_y_scores))
        global_auprc = float(average_precision_score(oof_y_true, oof_y_scores))
        global_f1 = float(f1_score(oof_y_true, oof_y_pred))
        global_prec = float(precision_score(oof_y_true, oof_y_pred))
        global_rec = float(recall_score(oof_y_true, oof_y_pred))

        mean_auroc = float(np.mean([r["auroc"] for r in fold_results]))
        std_auroc = float(np.std([r["auroc"] for r in fold_results]))

        summary = {
            "fold_results": fold_results,
            "global_oof_metrics": {
                "auroc": global_auroc,
                "auprc": global_auprc,
                "f1_score": global_f1,
                "precision": global_prec,
                "recall": global_rec,
                "mean_auroc": mean_auroc,
                "std_auroc": std_auroc
            }
        }

        return summary
