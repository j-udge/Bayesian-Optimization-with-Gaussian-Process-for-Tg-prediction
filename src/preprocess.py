"""
preprocess.py
=============
STEP 1 of the Active Learning Pipeline.

Run this ONCE on your original experimental dataset:
    python preprocess.py

This is your existing preprocessing script, lightly updated to:
  - Save a copy of the raw training data as 'original_train_data.csv'
    (used by retrain_loop.py to anchor future retraining runs)
  - Print clearer progress messages aligned with the pipeline stages

Output files:
  - processed_morgan_fp.parquet   → used by train_gp.py / retrain_loop.py
  - original_train_data.csv       → archive anchor for retrain_loop.py
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — Edit these to match your dataset
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FILE          = r"C:\Users\Lenovo\Downloads\experiment_polymer_data.xlsx"
SMILES_COLUMN_NAME  = "PSMILES"
TG_COLUMN_NAME      = "Tg_K"
OUTPUT_PARQUET      = "processed_morgan_fp.parquet"
MORGAN_RADIUS       = 2
MORGAN_NBITS        = 2048
# ─────────────────────────────────────────────────────────────────────────────


def process_smiles_to_morgan(
    input_file: str,
    smiles_col: str,
    tg_col: str,
    output_parquet: str,
) -> pd.DataFrame:

    print("=" * 60)
    print("  Polymer Active Learning — Preprocessing")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading dataset: {input_file}")
    df = pd.read_excel(input_file)
    print(f"  Raw rows: {len(df):,}")

    df_cleaned = df.dropna(subset=[tg_col])
    print(f"  After dropping NaN Tg: {len(df_cleaned):,} rows")

    # ── Fingerprint generation ─────────────────────────────────────────────────
    print(f"\n[2/4] Generating Morgan fingerprints "
          f"(radius={MORGAN_RADIUS}, nBits={MORGAN_NBITS})...")

    valid_smiles  = []
    valid_tgs     = []
    fingerprints  = []
    n_failed      = 0

    for idx, row in df_cleaned.iterrows():
        smiles = str(row[smiles_col])
        tg     = row[tg_col]

        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            fp       = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS)
            fp_array = np.zeros((MORGAN_NBITS,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, fp_array)

            fingerprints.append(fp_array)
            valid_smiles.append(smiles)
            valid_tgs.append(tg)
        else:
            n_failed += 1
            print(f"  ⚠️  Could not parse SMILES at index {idx}: {smiles[:60]}")

    print(f"  ✅ Valid: {len(valid_smiles):,}  |  Failed: {n_failed}")

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    print(f"\n[3/4] Assembling fingerprint DataFrame...")

    fp_matrix  = np.array(fingerprints, dtype=np.int8)
    bit_cols   = [f"bit_{i}" for i in range(fp_matrix.shape[1])]
    fp_df      = pd.DataFrame(fp_matrix, columns=bit_cols)

    base_df    = pd.DataFrame({"SMILES": valid_smiles, "Tg": valid_tgs})
    final_df   = pd.concat([base_df, fp_df], axis=1)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving outputs...")

    final_df.to_parquet(output_parquet, index=False)
    print(f"  ✅ Fingerprint parquet → {output_parquet}  ({len(final_df):,} records)")

    # Save a clean CSV anchor for retrain_loop.py (SMILES + Tg only, no fingerprints)
    anchor_csv = "original_train_data.csv"
    base_df.to_csv(anchor_csv, index=False)
    print(f"  ✅ Original training anchor → {anchor_csv}")

    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Valid molecules : {len(final_df):,}")
    print(f"  Tg range        : {min(valid_tgs):.1f} – {max(valid_tgs):.1f} K")
    print()
    print("  ✅ NEXT STEP:")
    print("     python train_gp.py       (train your GP model)")
    print("     python batch_inference.py (score your virtual library)")
    print("     streamlit run app.py     (launch the dashboard)")
    print("=" * 60)

    return final_df


if __name__ == "__main__":
    process_smiles_to_morgan(
        input_file     = INPUT_FILE,
        smiles_col     = SMILES_COLUMN_NAME,
        tg_col         = TG_COLUMN_NAME,
        output_parquet = OUTPUT_PARQUET,
    )
