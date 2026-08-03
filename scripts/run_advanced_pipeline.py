import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import GraphPreprocessor
from src.feature_extractor import extract_chemical_fingerprints, extract_protein_features
from src.model import AdvancedPharmaSentinelModel
from src.train import AdvancedTrainer, EnsembleClassifier

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def main():
    print("==========================================================================")
    print("   PharmaSentinel - Advanced R-GAT + RotatE + XGBoost Ensemble Pipeline  ")
    print("==========================================================================\n")

    os.makedirs(MODELS_DIR, exist_ok=True)
    graph_path = os.path.join(DATA_DIR, "decagon_graph_data.pt")

    if os.path.exists(graph_path):
        print(f"[LOADING] Loading preprocessed graph dataset from: {graph_path}")
        data = torch.load(graph_path, weights_only=False)
    else:
        print("[PROCESSING] Preprocessing graph data in memory...")
        preprocessor = GraphPreprocessor(min_side_effect_count=500)
        data = preprocessor.run_pipeline()
        torch.save(data, graph_path)

    num_nodes = data["num_nodes"]
    num_drugs = data["num_drugs"]
    num_proteins = data["num_proteins"]
    num_relations = data["num_relations"]
    splits = data["splits"]

    # 1. Feature Extraction (Biological Pre-training)
    print("\n-------------------------------------------------------------------------")
    print(" Improvement 1: Extracting Biological Features (1024-bit Morgan Fingerprints)")
    print("-------------------------------------------------------------------------")
    drug_feats = extract_chemical_fingerprints(data["drug2idx"], num_drugs, embedding_dim=1024)
    protein_feats = extract_protein_features(data["protein2idx"], num_proteins, embedding_dim=1024)
    initial_features = torch.cat([drug_feats, protein_feats], dim=0)

    train_edge_index = splits["train_edge_index"]
    train_edge_type = splits["train_edge_type"]
    val_edge_index = splits["val_edge_index"]
    val_edge_type = splits["val_edge_type"]
    test_edge_index = splits["test_edge_index"]
    test_edge_type = splits["test_edge_type"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Training Device: {device}")
    print(f"  Total Nodes: {num_nodes:,} | Relation Types: {num_relations:,}")
    print(f"  Training Edges: {train_edge_index.shape[1]:,} | Validation Edges: {val_edge_index.shape[1]:,}\n")

    # 2. Model Architecture Upgrade (R-GAT Encoder + RotatE Complex Decoder, D=256)
    print("-------------------------------------------------------------------------")
    print(" Improvements 2, 3 & 4: Training R-GAT + RotatE (D=256) with Hard Negative Sampling")
    print("-------------------------------------------------------------------------")
    model = AdvancedPharmaSentinelModel(num_nodes=num_nodes, num_relations=num_relations, in_dim=1024, embedding_dim=256)
    trainer = AdvancedTrainer(model=model, num_nodes=num_nodes, num_drugs=num_drugs, lr=0.008, device=device)

    # Cosine Annealing Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=20, eta_min=0.0005)

    num_epochs = 20
    best_val_auroc = 0.0

    print("Beginning Advanced Model Training Loop:")
    print("-------------------------------------------------------------------------")
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val AUROC':<12} | {'Val AUPRC':<12} | {'LR':<8}")
    print("-------------------------------------------------------------------------")

    for epoch in range(1, num_epochs + 1):
        loss = trainer.train_epoch(train_edge_index, train_edge_type, initial_features, batch_size=131072)
        val_auroc, val_auprc, _, _ = trainer.evaluate(val_edge_index, val_edge_type, initial_features)
        current_lr = trainer.optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f"{epoch:<8} | {loss:<12.5f} | {val_auroc:<12.4f} | {val_auprc:<12.4f} | {current_lr:<8.5f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            model_save_path = os.path.join(MODELS_DIR, "advanced_decagon_ensemble.pt")
            torch.save(model.state_dict(), model_save_path)

    print("-------------------------------------------------------------------------")

    # 3. Model Ensembling (GNN + Gradient Boosted Classifier)
    print("\n-------------------------------------------------------------------------")
    print(" Improvement 5: Multi-Model Ensembling (R-GAT + RotatE + XGBoost/GBDT)")
    print("-------------------------------------------------------------------------")
    _, _, y_train_true, y_train_scores = trainer.evaluate(train_edge_index, train_edge_type, initial_features, max_samples=50000)
    test_auroc_gnn, test_auprc_gnn, y_test_true, y_test_scores = trainer.evaluate(test_edge_index, test_edge_type, initial_features, max_samples=100000)

    ensemble = EnsembleClassifier(gnn_weight=0.75)
    ensemble_auroc, ensemble_auprc = ensemble.fit_and_predict(y_train_scores, y_train_true, y_test_scores, y_test_true)

    print("\n==========================================================================")
    print("                     FINAL PERFORMANCE COMPARISON                         ")
    print("==========================================================================")
    print(f"  Baseline Model (Step 3)             : AUROC = 0.9378 | AUPRC = 0.9216")
    print(f"  Advanced R-GAT + RotatE Model (D=256): AUROC = {test_auroc_gnn:.4f} | AUPRC = {test_auprc_gnn:.4f}")
    print(f"  FINAL ENSEMBLE MODEL (All 5 Imprs)  : AUROC = {ensemble_auroc:.4f} | AUPRC = {ensemble_auprc:.4f}")
    print("==========================================================================\n")

    # Export learned 256-dimensional spatial node embeddings
    embeddings_save_path = os.path.join(DATA_DIR, "advanced_node_embeddings.pt")
    embeddings = model.get_node_embeddings(initial_features.to(device)).detach().cpu()
    torch.save({
        "node_embeddings": embeddings,
        "drug2idx": data["drug2idx"],
        "protein2idx": data["protein2idx"],
        "se2idx": data["se2idx"]
    }, embeddings_save_path)

    print(f"[SAVING] Saved advanced model weights to: {os.path.join(MODELS_DIR, 'advanced_decagon_ensemble.pt')}")
    print(f"[SAVING] Saved learned 256D spatial node embeddings to: {embeddings_save_path}")
    print(f"\n[SUCCESS] Advanced pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
