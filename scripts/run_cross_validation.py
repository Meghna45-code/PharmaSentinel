import os
import sys
import json
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import GraphPreprocessor
from src.model import AdvancedPharmaSentinelModel
from src.train import AdvancedTrainer
from src.cross_validation import CrossValidationEngine

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def main():
    print("==========================================================================")
    print("     PharmaSentinel - 100% Honest 5-Fold Cross-Validation Benchmark       ")
    print("==========================================================================\n")

    graph_path = os.path.join(DATA_DIR, "decagon_graph_data.pt")

    if os.path.exists(graph_path):
        print(f"[LOADING] Loading graph dataset from: {graph_path}")
        data = torch.load(graph_path, weights_only=False)
    else:
        print("[PROCESSING] Preprocessing graph dataset...")
        pre = GraphPreprocessor(min_side_effect_count=500)
        data = pre.run_pipeline()
        torch.save(data, graph_path)

    cv_engine = CrossValidationEngine(data, n_splits=5, seed=42)
    summary = cv_engine.run_5fold_cv(
        model_cls=AdvancedPharmaSentinelModel,
        trainer_cls=AdvancedTrainer,
        in_dim=1024,
        embedding_dim=256,
        epochs_per_fold=8,
        lr=0.008
    )

    print("==========================================================================")
    print("            SUMMARY OF HONEST OUT-OF-FOLD (OOF) EVALUATION               ")
    print("==========================================================================")
    print(f"{'Fold':<8} | {'AUROC':<10} | {'AUPRC':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10}")
    print("--------------------------------------------------------------------------")
    for r in summary["fold_results"]:
        print(f"Fold {r['fold']:<3} | {r['auroc']:<10.4f} | {r['auprc']:<10.4f} | {r['f1_score']:<10.4f} | {r['precision']:<10.4f} | {r['recall']:<10.4f}")
    print("--------------------------------------------------------------------------")
    m = summary["global_oof_metrics"]
    print(f"GLOBAL OOF| {m['auroc']:<10.4f} | {m['auprc']:<10.4f} | {m['f1_score']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f}")
    print(f"\nMean AUROC across 5 folds: {m['mean_auroc']:.4f} +/- {m['std_auroc']:.4f}")
    print("==========================================================================\n")

    json_path = os.path.join(DATA_DIR, "honest_cross_validation_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SAVING] Saved honest cross-validation benchmark results to: {json_path}")
    print("[SUCCESS] 5-Fold Cross-Validation completed successfully!")

if __name__ == "__main__":
    main()
