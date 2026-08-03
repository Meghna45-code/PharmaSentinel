import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import GraphPreprocessor
from src.model import PharmaSentinelModel
from src.train import Trainer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def main():
    print("==========================================================")
    print("   PharmaSentinel - Graph Neural Network Training Engine   ")
    print("==========================================================\n")

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
    num_relations = data["num_relations"]
    splits = data["splits"]

    train_edge_index = splits["train_edge_index"]
    train_edge_type = splits["train_edge_type"]
    val_edge_index = splits["val_edge_index"]
    val_edge_type = splits["val_edge_type"]
    test_edge_index = splits["test_edge_index"]
    test_edge_type = splits["test_edge_type"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training Device: {device}")
    print(f"  Total Nodes: {num_nodes:,} | Relation Types: {num_relations:,}")
    print(f"  Training Edges: {train_edge_index.shape[1]:,} | Validation Edges: {val_edge_index.shape[1]:,}\n")

    model = PharmaSentinelModel(num_nodes=num_nodes, num_relations=num_relations, embedding_dim=128)
    trainer = Trainer(model=model, num_nodes=num_nodes, lr=0.01, device=device)

    num_epochs = 10
    best_val_auroc = 0.0

    print("Beginning Training Loop:")
    print("-------------------------------------------------------------------------")
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val AUROC':<12} | {'Val AUPRC':<12}")
    print("-------------------------------------------------------------------------")

    for epoch in range(1, num_epochs + 1):
        loss = trainer.train_epoch(train_edge_index, train_edge_type, batch_size=131072)
        val_auroc, val_auprc = trainer.evaluate(val_edge_index, val_edge_type)

        print(f"{epoch:<8} | {loss:<12.5f} | {val_auroc:<12.4f} | {val_auprc:<12.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            model_save_path = os.path.join(MODELS_DIR, "decagon_model.pt")
            torch.save(model.state_dict(), model_save_path)

    print("-------------------------------------------------------------------------")
    
    # Final Test Set Evaluation
    test_auroc, test_auprc = trainer.evaluate(test_edge_index, test_edge_type)
    print(f"\n[FINAL EVALUATION] Test AUROC: {test_auroc:.4f} | Test AUPRC: {test_auprc:.4f}")

    # Export learned 128-dimensional spatial node embeddings
    embeddings_save_path = os.path.join(DATA_DIR, "node_embeddings.pt")
    embeddings = model.get_node_embeddings().detach().cpu()
    torch.save({
        "node_embeddings": embeddings,
        "drug2idx": data["drug2idx"],
        "protein2idx": data["protein2idx"],
        "se2idx": data["se2idx"]
    }, embeddings_save_path)

    print(f"[SAVING] Saved trained model weights to: {os.path.join(MODELS_DIR, 'decagon_model.pt')}")
    print(f"[SAVING] Saved learned 128D spatial node embeddings to: {embeddings_save_path}")
    print(f"\n[SUCCESS] Model training and spatial embedding extraction completed successfully!")

if __name__ == "__main__":
    main()
