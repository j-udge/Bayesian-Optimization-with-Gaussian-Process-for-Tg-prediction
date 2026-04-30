# PolyBayes: Uncertainty-Aware Polymer Tg Prediction

> ⚠️ **Work in Progress** — this project is actively being developed. Expect rough edges.

A Bayesian active learning pipeline for predicting the glass transition temperature (Tg) of polymers. It combines Morgan Fingerprints with GPU-accelerated Gaussian Process Regression to predict thermal properties *and* quantify how confident the model actually is — which is the whole point for guiding lab synthesis decisions.

---

## What's in this repo

- `src/preprocess.py` — converts raw polymer SMILES → 2048-bit Morgan Fingerprints, outputs a `.parquet` file
- `src/model_form.py` — trains a GPyTorch ExactGP model on GPU, saves weights + metrics to `.pth`
- `src/batch_inference.py` — scores up to 1M molecules in memory-safe batches, saves predictions to parquet
- `src/app.py` — Streamlit dashboard for screening candidates, visualizing uncertainty, and logging lab results
- `src/retrain_loop.py` — merges new lab data with original training set and retrains the model
- `experiment_polymer_data.xlsx` — the experimental dataset used for training

---

## Dataset

The 1-million molecule virtual library comes from [PI1M](https://github.com/RUIMINMA1996/PI1M), a dataset of computationally generated polyimide structures.

**Note:** The pre-computed parquet file (`virtual_library_predictions.parquet`) is not included in this repo — it's several GB after GP inference. You'll need to run `batch_inference.py` yourself to generate it, or reach out and I can share it separately.

---

## How it works

```
preprocess.py → model_form.py → batch_inference.py → app.py (dashboard)
                                        ↑                    |
                                batch_inference.py    retrain_loop.py
                                  (re-score)       ← (new lab data)
```

Every time a scientist synthesizes a candidate and logs the real DSC result, `retrain_loop.py` folds it back into the training set. The model gets smarter with each iteration.

The dashboard ranks candidates using UCB acquisition:
```
Score = -(|Predicted_Tg - Target_Tg|) + β × σ
```
High β = explore unknown regions. Low β = exploit what the model already knows.

---

## Installation

```bash
git clone https://github.com/j-udge/PolyBayes.git
cd PolyBayes
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

> Install PyTorch for your CUDA version first: [pytorch.org](https://pytorch.org/get-started/locally/)

---

## Usage

```bash
# 1. Preprocess your dataset
python src/preprocess.py

# 2. Train the GP model
python src/model_form.py

# 3. Score the virtual library (slow, run once)
python src/batch_inference.py

# 4. Launch the dashboard
streamlit run src/app.py

# 5. After logging lab results in the dashboard, retrain:
python src/retrain_loop.py
# then re-run batch_inference.py to update predictions
```

Edit the `CONFIG` block at the top of each script to set your file paths before running.

---

## Stack

PyTorch · GPyTorch · RDKit · Streamlit · Pandas · PyArrow

---

## License

[MIT](https://github.com/j-udge/PolyBayes/blob/main/LICENSE)
