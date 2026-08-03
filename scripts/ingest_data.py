import os
import urllib.request
import gzip
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dpi-dataset")

OFFICIAL_DATASET_URLS = {
    "ChChSe-Decagon_polypharmacy.csv.gz": "https://snap.stanford.edu/biodata/datasets/10017/files/ChChSe-Decagon_polypharmacy.csv.gz",
    "PP-Decagon_ppi.csv.gz": "https://snap.stanford.edu/biodata/datasets/10008/files/PP-Decagon_ppi.csv.gz",
    "ChG-TargetDecagon_targets.csv.gz": "https://snap.stanford.edu/biodata/datasets/10015/files/ChG-TargetDecagon_targets.csv.gz",
    "ChSe-Decagon_monopharmacy.csv.gz": "https://snap.stanford.edu/biodata/datasets/10018/files/ChSe-Decagon_monopharmacy.csv.gz"
}

def download_and_extract_all():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Data Ingestion Directory: {DATA_DIR}\n")

    for filename, url in OFFICIAL_DATASET_URLS.items():
        gz_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(gz_path):
            print(f"[DOWNLOADING] {filename}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as resp, open(gz_path, 'wb') as out_file:
                while chunk := resp.read(1024 * 1024):
                    out_file.write(chunk)
            print(f"[DOWNLOADED] {filename} ({os.path.getsize(gz_path) / (1024*1024):.2f} MB)")
        else:
            print(f"[EXISTS] {filename} ({os.path.getsize(gz_path) / (1024*1024):.2f} MB)")

    print("\n[PROCESSING] Extracting and standardizing CSV files...")

    # 1. Polypharmacy Side Effects (bio-decagon-combo.csv)
    poly_csv_path = os.path.join(DATA_DIR, "bio-decagon-combo.csv")
    poly_gz = os.path.join(DATA_DIR, "ChChSe-Decagon_polypharmacy.csv.gz")
    if not os.path.exists(poly_csv_path) and os.path.exists(poly_gz):
        print("Processing polypharmacy side effects dataset...")
        df_poly = pd.read_csv(poly_gz, compression='gzip', sep='\t')
        df_poly.columns = ["STITCH 1", "STITCH 2", "Polypharmacy Side Effect", "Side Effect Name"]
        df_poly.to_csv(poly_csv_path, index=False)
        print(f"[PREPARED] bio-decagon-combo.csv ({len(df_poly):,} rows)")

    # 2. Protein-Protein Interactions (bio-decagon-ppi.csv)
    ppi_csv_path = os.path.join(DATA_DIR, "bio-decagon-ppi.csv")
    ppi_gz = os.path.join(DATA_DIR, "PP-Decagon_ppi.csv.gz")
    if not os.path.exists(ppi_csv_path) and os.path.exists(ppi_gz):
        print("Processing protein-protein interactions dataset...")
        df_ppi = pd.read_csv(ppi_gz, compression='gzip', sep='\t')
        df_ppi.columns = ["Gene 1", "Gene 2"]
        df_ppi.to_csv(ppi_csv_path, index=False)
        print(f"[PREPARED] bio-decagon-ppi.csv ({len(df_ppi):,} rows)")

    # 3. Drug Target Interactions (bio-decagon-targets.csv & bio-decagon-targets-all.csv)
    targets_csv_path = os.path.join(DATA_DIR, "bio-decagon-targets.csv")
    targets_all_csv_path = os.path.join(DATA_DIR, "bio-decagon-targets-all.csv")
    target_gz = os.path.join(DATA_DIR, "ChG-TargetDecagon_targets.csv.gz")
    if not os.path.exists(targets_csv_path) and os.path.exists(target_gz):
        print("Processing drug target interactions dataset...")
        df_target = pd.read_csv(target_gz, compression='gzip', sep='\t')
        df_target.columns = ["STITCH", "Gene"]
        df_target.to_csv(targets_csv_path, index=False)
        df_target.to_csv(targets_all_csv_path, index=False)
        print(f"[PREPARED] bio-decagon-targets.csv & bio-decagon-targets-all.csv ({len(df_target):,} rows)")

    print("\n[COMPLETED] All data ingestion and dataset preparation steps succeeded!")

if __name__ == "__main__":
    download_and_extract_all()
