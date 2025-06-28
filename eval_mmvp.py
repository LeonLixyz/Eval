import pandas as pd
from pathlib import Path
import sys

def paired_accuracy_from_hit(xlsx_path: str | Path,
                             hit_col: str = "hit") -> float:
    """
    Compute MMVP paired accuracy from the 0/1 `hit` column.
    Assumes each consecutive pair of rows belongs to the same image pair.
    """
    df = pd.read_excel(xlsx_path)                                # ← 1  :contentReference[oaicite:0]{index=0}

    # Ensure the hit column is integer 0/1
    df[hit_col] = df[hit_col].astype(int)

    # Build a pair id: 0,0,1,1,2,2,…
    df["pair_id"] = df.index // 2                               # ← 2  :contentReference[oaicite:1]{index=1}

    # For each pair, check if both hits are 1
    pair_correct = df.groupby("pair_id")[hit_col].sum() == 2    # ← 3  :contentReference[oaicite:2]{index=2}

    # Return percentage
    return pair_correct.mean() * 100


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python paired_acc.py <results.xlsx>")
    acc = paired_accuracy_from_hit(sys.argv[1])
    print(f"Paired accuracy: {acc:.2f}%")
