import os
import pandas as pd
import numpy as np

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dpi-dataset")

class DataLoader:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.combo_df = None
        self.ppi_df = None
        self.targets_all_df = None
        self.targets_df = None

    def load_all(self):
        """Loads all raw datasets into pandas DataFrames."""
        print("Loading datasets from:", self.data_dir)
        combo_path = os.path.join(self.data_dir, "bio-decagon-combo.csv")
        ppi_path = os.path.join(self.data_dir, "bio-decagon-ppi.csv")
        targets_all_path = os.path.join(self.data_dir, "bio-decagon-targets-all.csv")
        targets_path = os.path.join(self.data_dir, "bio-decagon-targets.csv")

        self.combo_df = pd.read_csv(combo_path)
        self.ppi_df = pd.read_csv(ppi_path)
        self.targets_all_df = pd.read_csv(targets_all_path)
        self.targets_df = pd.read_csv(targets_path)

        return {
            "bio_decagon_combo": self.combo_df,
            "bio_decagon_ppi": self.ppi_df,
            "bio_decagon_targets_all": self.targets_all_df,
            "bio_decagon_targets": self.targets_df
        }

    def get_summary_stats(self):
        """Returns row counts and metadata for all datasets."""
        if self.combo_df is None:
            self.load_all()

        return {
            "bio_decagon_combo_rows": len(self.combo_df),
            "bio_decagon_ppi_rows": len(self.ppi_df),
            "bio_decagon_targets_all_rows": len(self.targets_all_df),
            "bio_decagon_targets_rows": len(self.targets_df),
            "unique_side_effects": self.combo_df["Polypharmacy Side Effect"].nunique(),
            "unique_stitch_drugs": len(np.unique(np.append(self.combo_df["STITCH 1"].values, self.combo_df["STITCH 2"].values))),
            "unique_ppi_proteins": len(np.unique(np.append(self.ppi_df["Gene 1"].values, self.ppi_df["Gene 2"].values)))
        }

    def build_node_and_edge_mappings(self):
        """Constructs index mappings for drugs, proteins, and side effect edge types."""
        if self.combo_df is None:
            self.load_all()

        # Side effect edge mapping
        side_effects = np.array(self.combo_df["Polypharmacy Side Effect"])
        groups = self.combo_df.groupby(["Polypharmacy Side Effect"]).ngroup()
        
        edge_type_mapping = np.stack((np.array(groups), side_effects), axis=1)
        edge_type_mapping = edge_type_mapping[edge_type_mapping[:, 0].argsort()]
        edge_type_dict = dict((value, key) for key, value in edge_type_mapping)

        max_idx = list(edge_type_dict.values())[-1]
        edge_type_dict["ppi"] = max_idx + 1
        edge_type_dict["target"] = max_idx + 2
        edge_type_dict["targeted_by"] = max_idx + 3

        # Drug nodes mapping
        unique_drugs = np.unique(np.append(self.combo_df["STITCH 1"].values, self.combo_df["STITCH 2"].values))
        drug_nodes_dict = {drug: i for i, drug in enumerate(unique_drugs)}

        # Protein nodes mapping
        unique_proteins = np.unique(np.append(self.ppi_df["Gene 1"].values, self.ppi_df["Gene 2"].values))
        last_drug_idx = drug_nodes_dict[unique_drugs[-1]]
        protein_nodes_dict = {uniprot: i + last_drug_idx + 1 for i, uniprot in enumerate(unique_proteins)}

        return {
            "edge_type_mapping": edge_type_dict,
            "drug_nodes_mapping": drug_nodes_dict,
            "protein_nodes_mapping": protein_nodes_dict
        }

if __name__ == "__main__":
    loader = DataLoader()
    datasets = loader.load_all()
    stats = loader.get_summary_stats()
    mappings = loader.build_node_and_edge_mappings()

    print("Data Ingestion & Validation Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")
    print(f"  Drug mapping count: {len(mappings['drug_nodes_mapping']):,}")
    print(f"  Protein mapping count: {len(mappings['protein_nodes_mapping']):,}")
