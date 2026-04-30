# PolyBayes: Uncertainty-Aware Glass Transition (Tg) Prediction for Polymers

A full **Active Learning pipeline** and interactive web dashboard for predicting the glass transition temperature (Tg) of polymers. This project leverages Morgan Fingerprints and GPU-accelerated Gaussian Process Regression to not only predict thermal properties but also quantify model uncertainty — enabling closed-loop Bayesian optimization across virtual libraries of up to 1 million polymers.

---

## Features

- **Data Preprocessing Pipeline** — Robust automated conversion of polymer SMILES strings into 2048-bit Morgan Fingerprints using RDKit, with NaN removal, SMILES validation, and high-performance `.parquet` output.
- **GPU-Accelerated Training** — Utilizes PyTorch and GPyTorch for exact Gaussian Process inference on CUDA, dramatically reducing training time compared to CPU-bound alternatives.
- **1-Million Molecule Batch Inference** — A dedicated offline engine scores entire virtual libraries in memory-safe batches (configurable chunk size), saving results to a compressed Snappy parquet for instant dashboard loading.
- **Bayesian Uncertainty Quantification** — Outputs both a predicted Tg and a standard deviation (σ), identifying sparse regions of chemical space and guiding the most information-rich experiments.
- **Interactive Streamlit Dashboard** — A scientist-facing web interface for real-time candidate screening with UCB acquisition scoring, target Tg search, and Bayesian predictive distribution visualization.
- **Closed-Loop Active Learning** — Lab results logged in the dashboard feed directly into a retraining script that augments the GP model, triggering a new inference pass so every synthesis improves the next recommendation.

---

## Active Learning Architecture

The pipeline is divided into four isolated stages, keeping heavy compute separate from the interactive UI:

```
preprocess.py         →   batch_inference.py   →   app.py (Streamlit)
(SMILES → fingerprints)   (1M GP predictions)       (screen + log results)
                                ↑                            |
                        batch_inference.py          retrain_loop.py
                        (re-score library)    ←   (incorporate lab data)
```

Each lab result logged in the dashboard closes the loop: the model grows smarter with every synthesis cycle.

---

## Technology Stack

| Category | Libraries |
|---|---|
| Machine Learning | PyTorch, GPyTorch, Scikit-Learn |
| Cheminformatics | RDKit |
| Web Dashboard | Streamlit |
| Data Handling | Pandas, NumPy, PyArrow (Parquet/Snappy) |
| GPU Acceleration | CUDA via PyTorch |

---

## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/j-udge/PolyBayes.git
cd PolyBayes
```

**2. Create a virtual environment:**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

**3. Install dependencies:**

> **Note:** Install the correct PyTorch version for your CUDA version first. Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the right command. Then install the rest:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the four scripts in order. Steps 1–3 are run once; Steps 3–4 repeat each active learning cycle.

### Step 1 — Preprocess Experimental Data (`preprocess.py`)

Converts your raw polymer Excel/CSV dataset into Morgan Fingerprints. Handles NaN removal, SMILES validation, and outputs a high-performance `.parquet` file and a training anchor CSV used by the retrain loop.

```bash
python src/preprocess.py
```

Edit the `CONFIG` block at the top of the script to point to your dataset file and column names before running.

**Outputs:**
- `processed_morgan_fp.parquet` — fingerprint matrix used for GP training
- `original_train_data.csv` — anchor file used by `retrain_loop.py`

---

### Step 2 — Train the GP Model (`model_form.py`)

Loads the fingerprint parquet, pushes tensors to GPU (CUDA), and trains a GPyTorch ExactGP model with a Matérn 1.5 kernel. Saves model weights, training data, and evaluation metrics (R², RMSE) to a deployment-ready `.pth` file.

```bash
python src/model_form.py
```

**Output:** `gpytorch_tg_model.pth`

---

### Step 3 — Batch Inference on Virtual Library (`batch_inference.py`)

Scores your entire virtual library (up to 1 million molecules) in memory-safe batches. Each molecule receives a predicted Tg and uncertainty (σ). Results are saved to a compressed parquet that the dashboard loads instantly.

```bash
python src/batch_inference.py
```

Edit `VIRTUAL_LIB_CSV` and `BATCH_SIZE` in the `CONFIG` block to match your setup. Reduce `BATCH_SIZE` to 2,000 if you run into RAM issues.

**Output:** `virtual_library_predictions.parquet`

---

### Step 4 — Launch the Dashboard (`app.py`)

Launches the interactive Streamlit interface. The scientist sets a target Tg and exploration factor (β), and the UCB acquisition function ranks all candidates instantly. The top recommendation is shown with its predicted Tg, uncertainty, and distance to target.

```bash
streamlit run src/app.py
```

**UCB Acquisition Score:**
```
Score = -(|Predicted_Tg - Target_Tg|) + β × σ
```
- **β = 0** → pure exploitation (trust the model's best guess)
- **β = 5** → high exploration (prefer uncertain, under-explored molecules)

Use a higher β early in your campaign when the model needs information; lower it as confidence grows.

---

### Step 5 — Log Lab Results & Close the Loop

After synthesizing the top candidate and measuring its Tg by DSC, enter the result in the **"Log Lab Result"** form at the bottom of the dashboard. This saves it to `new_lab_results.csv`.

Then run the retrain script:

```bash
python src/retrain_loop.py
```

This merges the new data point with your original training set, retrains the GP from scratch, archives the consumed lab results (so they are never double-counted), and backs up the previous model with a timestamp. Then re-run Step 3 to re-score the virtual library with the improved model.

**Full active learning cycle:**
```
Step 3 (batch_inference) → Step 4 (dashboard) → log result → Step 5 (retrain) → repeat Step 3
```

---

## Bayesian Optimization & Active Learning

When the scientist queries the dashboard, the UCB acquisition function ranks all candidates by balancing two competing objectives:

- **Exploitation** — molecules close to the target Tg based on the model's predictions
- **Exploration** — molecules in sparse, under-explored regions of chemical space (high σ)

High-uncertainty candidates provide the maximum information gain when synthesized, improving model accuracy in those regions for future iterations. Over successive cycles the model's uncertainty narrows, and its recommendations become increasingly reliable.

---

## Project Structure

```
PolyBayes/
├── src/
│   ├── preprocess.py          # SMILES → Morgan fingerprint parquet
│   ├── model_form.py          # GP model training (GPU)
│   ├── batch_inference.py     # 1M molecule offline scoring engine
│   ├── app.py                 # Streamlit active learning dashboard
│   └── retrain_loop.py        # Closed-loop model retraining
├── requirements.txt
├── LICENSE
└── README.md
```

---

## License

[MIT License](https://github.com/j-udge/PolyBayes/blob/main/LICENSE)
