"""
retrain_loop.py
===============
STEP 4 of the Active Learning Pipeline.

Run this AFTER logging at least one lab result in the dashboard:
    python retrain_loop.py

This script:
  1. Loads the original training data (parquet from preprocessing)
  2. Loads new lab results logged via the Streamlit dashboard
  3. Combines them and retrains the GP model from scratch
  4. Saves the updated model as gpytorch_tg_model.pth
  5. Archives the consumed lab results so they aren't double-counted
  6. Prompts you to re-run batch_inference.py to update all predictions

After this completes, run batch_inference.py again to close the loop.
"""

import torch
import gpytorch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os
import shutil
import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — Edit these paths to match your environment
# ─────────────────────────────────────────────────────────────────────────────
ORIGINAL_PARQUET  = r"C:\Tg_Bayesian_Optimization\src\processed_morgan_fp.parquet"
LAB_RESULTS_CSV   = "new_lab_results.csv"
MODEL_OUTPUT_PATH = r"C:\Tg_Bayesian_Optimization\src\gpytorch_tg_model.pth"
ARCHIVE_DIR       = "lab_results_archive"
TRAINING_ITERS    = 100
LEARNING_RATE     = 0.1
# ─────────────────────────────────────────────────────────────────────────────


print("=" * 60)
print("  Polymer Active Learning — Retrain Loop")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")


# ── 1. Load lab results ───────────────────────────────────────────────────────
print(f"\n[1/6] Checking for new lab results in '{LAB_RESULTS_CSV}'...")

if not os.path.exists(LAB_RESULTS_CSV):
    print("  ❌ No lab results file found. Log results in the Streamlit dashboard first.")
    exit(1)

lab_df = pd.read_csv(LAB_RESULTS_CSV)

if lab_df.empty:
    print("  ❌ Lab results file is empty. Nothing to retrain on.")
    exit(1)

print(f"  ✅ Found {len(lab_df)} new lab results.")
print(lab_df[["SMILES", "Actual_Tg"]].to_string(index=False))


# ── 2. Convert lab SMILES → fingerprints ─────────────────────────────────────
print(f"\n[2/6] Converting new lab SMILES to Morgan fingerprints...")

new_fps     = []
new_tgs     = []
new_smiles_valid = []

for _, row in lab_df.iterrows():
    mol = Chem.MolFromSmiles(str(row["SMILES"]))
    if mol is not None:
        fp       = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_array = np.zeros((2048,), dtype=np.float32)
        from rdkit.Chem import DataStructs
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        new_fps.append(fp_array)
        new_tgs.append(float(row["Actual_Tg"]))
        new_smiles_valid.append(row["SMILES"])
    else:
        print(f"  ⚠️ Could not parse SMILES: {row['SMILES']} — skipping.")

if not new_fps:
    print("  ❌ No valid SMILES in lab results. Exiting.")
    exit(1)

print(f"  ✅ {len(new_fps)} valid lab fingerprints generated.")

# Build a compatible dataframe
bit_cols      = [f"bit_{i}" for i in range(2048)]
new_lab_fp_df = pd.DataFrame(np.array(new_fps), columns=bit_cols)
new_lab_fp_df.insert(0, "SMILES", new_smiles_valid)
new_lab_fp_df.insert(1, "Tg",     new_tgs)


# ── 3. Load & merge with original training data ───────────────────────────────
print(f"\n[3/6] Loading original training data from '{ORIGINAL_PARQUET}'...")

original_df = pd.read_parquet(ORIGINAL_PARQUET)
print(f"  Original dataset: {len(original_df):,} samples")

# Stack new lab data on top of original — same column structure
combined_df = pd.concat([original_df, new_lab_fp_df], ignore_index=True)
print(f"  Combined dataset: {len(combined_df):,} samples (+{len(new_lab_fp_df)} new)")


# ── 4. Prepare tensors ────────────────────────────────────────────────────────
print(f"\n[4/6] Preparing training tensors...")

X = combined_df[[c for c in combined_df.columns if c.startswith("bit_")]].to_numpy(dtype=np.float32)
y = combined_df["Tg"].to_numpy(dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_x = torch.tensor(X_train).to(device)
train_y = torch.tensor(y_train).to(device)
test_x  = torch.tensor(X_test).to(device)

print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")


# ── 5. Define & retrain GP ────────────────────────────────────────────────────
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x)
        )


print(f"\n[5/6] Retraining GP model for {TRAINING_ITERS} iterations...")

likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model      = ExactGPModel(train_x, train_y, likelihood).to(device)
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
mll       = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

start_time = time.time()

for i in range(TRAINING_ITERS):
    optimizer.zero_grad()
    loss = -mll(model(train_x), train_y)
    loss.backward()
    optimizer.step()

    if (i + 1) % 20 == 0:
        print(f"  Iter {i+1:>3}/{TRAINING_ITERS}  —  Loss: {loss.item():.4f}")

elapsed = time.time() - start_time
print(f"  ✅ Training complete in {elapsed:.1f}s")

# ── Evaluate ──────────────────────────────────────────────────────────────────
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds    = likelihood(model(test_x))
    y_pred   = preds.mean.cpu().numpy()

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n  R²  = {r2:.4f}")
print(f"  RMSE = {rmse:.2f} K")


# ── 6. Save updated model ─────────────────────────────────────────────────────
print(f"\n[6/6] Saving updated model to '{MODEL_OUTPUT_PATH}'...")

# Archive old model first
old_model_dir = os.path.dirname(MODEL_OUTPUT_PATH)
archive_name  = os.path.join(
    old_model_dir,
    f"gpytorch_tg_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
)
if os.path.exists(MODEL_OUTPUT_PATH):
    shutil.copy2(MODEL_OUTPUT_PATH, archive_name)
    print(f"  📦 Old model archived as: {os.path.basename(archive_name)}")

payload = {
    "model_state":      model.state_dict(),
    "likelihood_state": likelihood.state_dict(),
    "metrics": {
        "r2":         float(r2),
        "rmse":       float(rmse),
        "train_size": len(X_train),
        "test_size":  len(X_test),
    },
    "train_data": {
        "x": train_x.cpu(),
        "y": train_y.cpu(),
    },
}
torch.save(payload, MODEL_OUTPUT_PATH)
print(f"  ✅ Model saved.")

# Archive consumed lab results (so we don't double-count next iteration)
os.makedirs(ARCHIVE_DIR, exist_ok=True)
archive_csv = os.path.join(
    ARCHIVE_DIR,
    f"lab_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)
shutil.move(LAB_RESULTS_CSV, archive_csv)
print(f"  📦 Lab results archived to: {archive_csv}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RETRAINING COMPLETE")
print("=" * 60)
print(f"  New training set : {len(X_train):,} samples")
print(f"  R²               : {r2:.4f}")
print(f"  RMSE             : {rmse:.2f} K")
print()
print("  ✅ NEXT STEP:")
print("     python batch_inference.py")
print("     (Re-score all 1M molecules with the updated model)")
print("=" * 60)
