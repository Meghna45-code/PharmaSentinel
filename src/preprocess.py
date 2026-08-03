import os
import pickle
import numpy as np
import pandas as pd
import torch

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dpi-dataset")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")

class GraphPreprocessor:
    def __init__(self, data_dir=None, min_side_effect_count=500, val_ratio=0.10, test_ratio=0.10, seed=42):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.min_side_effect_count = min_side_effect_count
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def load_and_clean_raw(self):
        """Loads and cleans raw CSV files."""
        combo_path = os.path.join(self.data_dir, "bio-decagon-combo.csv")
        ppi_path = os.path.join(self.data_dir, "bio-decagon-ppi.csv")
        targets_path = os.path.join(self.data_dir, "bio-decagon-targets.csv")

        # Load CSVs with robust separator handling
        df_combo = pd.read_csv(combo_path).dropna()
        df_ppi = pd.read_csv(ppi_path).dropna()
        
        # Target CSV might be tab or comma separated
        try:
            df_targets = pd.read_csv(targets_path, sep='\t').dropna()
            if len(df_targets.columns) == 1 and ',' in df_targets.columns[0]:
                df_targets = pd.read_csv(targets_path, sep=',').dropna()
        except Exception:
            df_targets = pd.read_csv(targets_path).dropna()

        # Standardize target column names
        df_targets.columns = [c.replace('#', '').strip() for c in df_targets.columns]
        if "Drug" in df_targets.columns:
            df_targets = df_targets.rename(columns={"Drug": "STITCH"})
        if len(df_targets.columns) >= 2 and "STITCH" not in df_targets.columns:
            df_targets.columns = ["STITCH", "Gene"] + list(df_targets.columns[2:])

        # Remove duplicate rows
        df_combo = df_combo.drop_duplicates(subset=["STITCH 1", "STITCH 2", "Polypharmacy Side Effect"])
        df_ppi = df_ppi.drop_duplicates(subset=["Gene 1", "Gene 2"])
        df_targets = df_targets.drop_duplicates(subset=["STITCH", "Gene"])


        print(f"Loaded raw cleaned DataFrames:")
        print(f"  Combo interactions: {len(df_combo):,} rows")
        print(f"  PPI interactions: {len(df_ppi):,} rows")
        print(f"  Drug Target interactions: {len(df_targets):,} rows")

        return df_combo, df_ppi, df_targets

    def filter_rare_side_effects(self, df_combo):
        """Filters side effect categories with fewer than min_side_effect_count occurrences."""
        counts = df_combo["Polypharmacy Side Effect"].value_counts()
        frequent_se = counts[counts >= self.min_side_effect_count].index
        
        df_filtered = df_combo[df_combo["Polypharmacy Side Effect"].isin(frequent_se)].copy()
        print(f"\nSide Effect Filtering (Threshold >= {self.min_side_effect_count} occurrences):")
        print(f"  Original side effects: {counts.nunique():,} -> Retained side effects: {len(frequent_se):,}")
        print(f"  Retained drug-drug interaction pairs: {len(df_filtered):,} rows")

        return df_filtered

    def build_node_mappings(self, df_combo, df_ppi, df_targets):
        """Creates contiguous numerical index mappings for drugs and proteins."""
        unique_drugs = np.unique(np.concatenate([
            df_combo["STITCH 1"].unique(),
            df_combo["STITCH 2"].unique(),
            df_targets["STITCH"].unique()
        ]))

        unique_proteins = np.unique(np.concatenate([
            df_ppi["Gene 1"].unique(),
            df_ppi["Gene 2"].unique(),
            df_targets["Gene"].unique()
        ]))

        drug2idx = {drug: i for i, drug in enumerate(unique_drugs)}
        num_drugs = len(unique_drugs)

        protein2idx = {protein: i + num_drugs for i, protein in enumerate(unique_proteins)}
        num_proteins = len(unique_proteins)

        total_nodes = num_drugs + num_proteins

        print(f"\nGraph Node Indexing:")
        print(f"  Drug Nodes (0 to {num_drugs - 1}): {num_drugs:,}")
        print(f"  Protein Nodes ({num_drugs} to {total_nodes - 1}): {num_proteins:,}")
        print(f"  Total Nodes in Multimodal Graph: {total_nodes:,}")

        return drug2idx, protein2idx, num_drugs, num_proteins, total_nodes

    def build_graph_tensors(self, df_combo, df_ppi, df_targets, drug2idx, protein2idx):
        """Constructs edge_index and edge_type tensors for PyTorch Geometric / NetworkX."""
        
        # 1. Edge relation mapping
        unique_se = sorted(df_combo["Polypharmacy Side Effect"].unique())
        se2idx = {se: i for i, se in enumerate(unique_se)}
        num_se_types = len(unique_se)

        # Assign relation IDs
        # Side effect relations: 0 .. num_se_types-1
        # PPI relation: num_se_types
        # Target relation: num_se_types + 1
        ppi_rel_idx = num_se_types
        target_rel_idx = num_se_types + 1
        num_relations = num_se_types + 2

        src_list, dst_list, type_list = [], [], []

        # Drug-Drug Side Effect Edges (Bidirectional)
        for _, row in df_combo.iterrows():
            d1 = drug2idx[row["STITCH 1"]]
            d2 = drug2idx[row["STITCH 2"]]
            rel = se2idx[row["Polypharmacy Side Effect"]]
            
            src_list.extend([d1, d2])
            dst_list.extend([d2, d1])
            type_list.extend([rel, rel])

        num_ddi_edges = len(src_list)

        # Protein-Protein Interaction Edges (Bidirectional)
        for _, row in df_ppi.iterrows():
            p1 = protein2idx[row["Gene 1"]]
            p2 = protein2idx[row["Gene 2"]]
            
            src_list.extend([p1, p2])
            dst_list.extend([p2, p1])
            type_list.extend([ppi_rel_idx, ppi_rel_idx])

        num_ppi_edges = len(src_list) - num_ddi_edges

        # Drug-Target Edges (Bidirectional)
        for _, row in df_targets.iterrows():
            d = drug2idx[row["STITCH"]]
            p = protein2idx[row["Gene"]]
            
            src_list.extend([d, p])
            dst_list.extend([p, d])
            type_list.extend([target_rel_idx, target_rel_idx])

        num_target_edges = len(src_list) - num_ddi_edges - num_ppi_edges

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(type_list, dtype=torch.long)

        print(f"\nGraph Edges Construction:")
        print(f"  Side Effect Edge Types: {num_se_types:,}")
        print(f"  Drug-Drug Edges: {num_ddi_edges:,}")
        print(f"  Protein-Protein Edges: {num_ppi_edges:,}")
        print(f"  Drug-Target Edges: {num_target_edges:,}")
        print(f"  Total Graph Edges (Directed Tensors): {edge_index.shape[1]:,}")

        return edge_index, edge_type, se2idx, num_relations

    def create_train_val_test_splits(self, edge_index, edge_type):
        """Generates Train (80%), Val (10%), and Test (10%) splits."""
        num_edges = edge_index.shape[1]
        perm = np.random.permutation(num_edges)

        n_val = int(num_edges * self.val_ratio)
        n_test = int(num_edges * self.test_ratio)
        n_train = num_edges - n_val - n_test

        train_indices = perm[:n_train]
        val_indices = perm[n_train:n_train + n_val]
        test_indices = perm[n_train + n_val:]

        splits = {
            "train_edge_index": edge_index[:, train_indices],
            "train_edge_type": edge_type[train_indices],
            "val_edge_index": edge_index[:, val_indices],
            "val_edge_type": edge_type[val_indices],
            "test_edge_index": edge_index[:, test_indices],
            "test_edge_type": edge_type[test_indices],
        }

        print(f"\nTrain / Val / Test Edge Splits:")
        print(f"  Train Edges (80%): {splits['train_edge_index'].shape[1]:,}")
        print(f"  Validation Edges (10%): {splits['val_edge_index'].shape[1]:,}")
        print(f"  Test Edges (10%): {splits['test_edge_index'].shape[1]:,}")

        return splits

    def run_pipeline(self):
        """Executes full preprocessing and graph construction."""
        df_combo, df_ppi, df_targets = self.load_and_clean_raw()
        df_combo_filtered = self.filter_rare_side_effects(df_combo)
        
        drug2idx, protein2idx, num_drugs, num_proteins, total_nodes = self.build_node_mappings(
            df_combo_filtered, df_ppi, df_targets
        )

        edge_index, edge_type, se2idx, num_relations = self.build_graph_tensors(
            df_combo_filtered, df_ppi, df_targets, drug2idx, protein2idx
        )

        splits = self.create_train_val_test_splits(edge_index, edge_type)

        processed_data = {
            "num_nodes": total_nodes,
            "num_drugs": num_drugs,
            "num_proteins": num_proteins,
            "num_relations": num_relations,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "drug2idx": drug2idx,
            "protein2idx": protein2idx,
            "se2idx": se2idx,
            "splits": splits
        }

        return processed_data

if __name__ == "__main__":
    preprocessor = GraphPreprocessor()
    data = preprocessor.run_pipeline()
