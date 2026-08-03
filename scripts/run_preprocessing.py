import os
import pickle
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import GraphPreprocessor, PROCESSED_DIR

def main():
    print("==========================================================")
    print("      PharmaSentinel - Graph Preprocessing Pipeline       ")
    print("==========================================================\n")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    preprocessor = GraphPreprocessor(min_side_effect_count=500)
    graph_data = preprocessor.run_pipeline()

    pt_output_path = os.path.join(PROCESSED_DIR, "decagon_graph_data.pt")
    pkl_output_path = os.path.join(PROCESSED_DIR, "decagon_graph_data.pkl")

    print(f"\n[SAVING] Saving preprocessed PyTorch graph data to: {pt_output_path}")
    torch.save(graph_data, pt_output_path)

    print(f"[SAVING] Saving python pickle graph data to: {pkl_output_path}")
    with open(pkl_output_path, "wb") as f:
        pickle.dump(graph_data, f)

    pt_size_mb = os.path.getsize(pt_output_path) / (1024 * 1024)
    print(f"\n[SUCCESS] Preprocessing completed successfully! Saved graph dataset artifact ({pt_size_mb:.2f} MB).")

if __name__ == "__main__":
    main()
