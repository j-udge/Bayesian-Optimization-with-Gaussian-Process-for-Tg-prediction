import pandas as pd
import numpy as np
import torch
import gpytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

# Hardcoded to use CUDA
device = torch.device("cuda")
print(f"0. Using device: {device}")

# 1. Load Data
print("1. Loading processed fingerprints...")
df = pd.read_parquet(r"C:\Tg_Bayesian_Optimization\src\processed_morgan_fp.parquet")

X = df[[col for col in df.columns if col.startswith('bit_')]].to_numpy(dtype=np.float32)
y = df['Tg'].to_numpy(dtype=np.float32)

if 'SMILES' in df.columns:
    smiles = df['SMILES'].to_numpy(dtype=str)
else:
    smiles = np.arange(len(y))

# 2. Split Data
print("2. Splitting data (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
    X, y, smiles, test_size=0.2, random_state=42
)

# Convert NumPy arrays to PyTorch Tensors and push them to the GPU
train_x = torch.tensor(X_train).contiguous().to(device)
train_y = torch.tensor(y_train).contiguous().to(device)
test_x = torch.tensor(X_test).contiguous().to(device)

# 3. Define the GPyTorch Model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

print("3. Initializing GPyTorch Model...")
# Initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = ExactGPModel(train_x, train_y, likelihood).to(device)

# 4. Train Model
print(f"4. Training model on {len(X_train)} polymers on {device}...")
model.train()
likelihood.train()

# Use the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 100 # 50-100 is usually plenty for Exact GPs
start_time = time.time()

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    
    if (i+1) % 20 == 0:
        print(f"Iter {i+1}/{training_iterations} - Loss: {loss.item():.3f}")
        
    optimizer.step()

print(f"Training completed in {time.time() - start_time:.2f} seconds!")

# 5. Evaluate on Test Set
print("5. Evaluating on test set...")
model.eval()
likelihood.eval()

# Make predictions without tracking gradients to save memory
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))
    
    # Move predictions back to CPU for sklearn metrics
    y_pred_test = observed_pred.mean.cpu().numpy()
    sigma_test = observed_pred.stddev.cpu().numpy()

r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

# --- TERMINAL REPORT ---
print("\n" + "="*30)
print("     MODEL EVALUATION REPORT")
print("="*30)
print(f"R^2 Score: {r2:.3f}")
print(f"RMSE:      {rmse:.2f} K")
print("="*30 + "\n")

# 6. Package and Freeze
print("6. Packaging deployment payload...")

# For PyTorch, we save the state_dict instead of the whole object
deployment_payload = {
    'model_state': model.state_dict(),
    'likelihood_state': likelihood.state_dict(),
    'metrics': {
        'r2': float(r2),
        'rmse': float(rmse),
        'train_size': len(X_train),
        'test_size': len(X_test)
    },
    'train_data': { # We need to save the training data because ExactGPs need it for inference
        'x': train_x.cpu(),
        'y': train_y.cpu()
    }
}

# 7. Save to Disk
print("7. Saving model to disk...")
# We use torch.save instead of joblib for PyTorch tensors/models
torch.save(deployment_payload, 'gpytorch_tg_model.pth')
print("Done! Model saved as 'gpytorch_tg_model.pth'")
