"""
batch_inference.py
==================
STEP 2 of the Active Learning Pipeline.

Run this ONCE from your terminal after training your GP model:
    python batch_inference.py

This script:
  - Loads your 1-million molecule virtual library
  - Processes fingerprints in memory-safe batches
  - Runs GP inference (mean + uncertainty) on every molecule
  - Saves results to virtual_library_predictions.parquet

The resulting parquet is what the Streamlit dashboard loads instantly.
"""

import torch
import gpytorch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — Edit these paths to match your environment
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH        = r"C:\Tg_Bayesian_Optimization\src\gpytorch_tg_model.pth"
VIRTUAL_LIB_CSV   = r"C:\Tg_Bayesian_Optimization\src\one_million_virtual_polymers.csv"
SMILES_COLUMN     = "SMILES"          # Column name in your virtual library CSV
OUTPUT_PARQUET    = "virtual_library_predictions.parquet"
BATCH_SIZE        = 5_000              # Reduce to 2000 if you hit RAM issues
# ─────────────────────────────────────────────────────────────────────────────


# ── 1. Define model architecture (must match training script exactly) ─────────
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ── 2. Load trained model ─────────────────────────────────────────────────────
print("=" * 60)
print("  Polymer Active Learning — Batch Inference Engine")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[1/4] Device: {device}")

print(f"[2/4] Loading GP model from: {MODEL_PATH}")
payload    = torch.load(MODEL_PATH, map_location=device)
train_x    = payload["train_data"]["x"].to(device)
train_y    = payload["train_data"]["y"].to(device)

likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model      = ExactGPModel(train_x, train_y, likelihood).to(device)
model.load_state_dict(payload["model_state"])
likelihood.load_state_dict(payload["likelihood_state"])
model.eval()
likelihood.eval()
print("    Model loaded and set to eval mode.")


# ── 3. Load virtual library ───────────────────────────────────────────────────
print(f"\n[3/4] Loading virtual library from: {VIRTUAL_LIB_CSV}")
df_library = pd.read_csv(VIRTUAL_LIB_CSV)
print(f"    Found {len(df_library):,} molecules.")

smiles_list = df_library[SMILES_COLUMN].tolist()
n_molecules = len(smiles_list)


# ── 4. Batch inference ────────────────────────────────────────────────────────
print(f"\n[4/4] Running GP inference in batches of {BATCH_SIZE:,}...")
print(f"    This may take several minutes for {n_molecules:,} molecules.\n")

all_results   = []
n_failed      = 0
start_time    = time.time()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for batch_start in tqdm(range(0, n_molecules, BATCH_SIZE), desc="Batches", unit="batch"):
        batch_smiles = smiles_list[batch_start : batch_start + BATCH_SIZE]

        # ── Fingerprint generation ────────────────────────────────────────────
        fps          = []
        valid_smiles = []

        for smi in batch_smiles:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fps.append(np.array(fp, dtype=np.float32))
                valid_smiles.append(smi)
            else:
                n_failed += 1

        if not fps:
            continue

        # ── GP prediction ─────────────────────────────────────────────────────
        fp_tensor   = torch.tensor(np.array(fps), dtype=torch.float32).to(device)
        predictions = likelihood(model(fp_tensor))
        means       = predictions.mean.cpu().numpy()
        sigmas      = predictions.stddev.cpu().numpy()

        # ── Acquisition scores ────────────────────────────────────────────────
        # Pre-compute UCB components; the dashboard will apply target penalty on-the-fly
        batch_df = pd.DataFrame({
            "SMILES":        valid_smiles,
            "Predicted_Tg":  means.astype(np.float32),
            "Uncertainty":   sigmas.astype(np.float32),
        })
        all_results.append(batch_df)

        # Free GPU memory after each batch
        del fp_tensor, predictions, means, sigmas
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ── 5. Save results ───────────────────────────────────────────────────────────
elapsed = time.time() - start_time
final_df = pd.concat(all_results, ignore_index=True)

# Round floats to save ~40% parquet file size
final_df["Predicted_Tg"] = final_df["Predicted_Tg"].round(4)
final_df["Uncertainty"]  = final_df["Uncertainty"].round(4)

final_df.to_parquet(OUTPUT_PARQUET, index=False, compression="snappy")

print("\n" + "=" * 60)
print("  BATCH INFERENCE COMPLETE")
print("=" * 60)
print(f"  Processed : {len(final_df):,} molecules")
print(f"  Skipped   : {n_failed:,} invalid SMILES")
print(f"  Time      : {elapsed/60:.1f} minutes")
print(f"  Output    : {OUTPUT_PARQUET}  ({os.path.getsize(OUTPUT_PARQUET)/1e6:.1f} MB)")
print("\n  ✅ Ready for Streamlit dashboard (app.py)")
print("=" * 60)
