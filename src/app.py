"""
app.py
======
STEP 3 of the Active Learning Pipeline — The Streamlit Dashboard.

Run with:
    streamlit run app.py

Workflow:
  1. Loads pre-computed virtual_library_predictions.parquet instantly
  2. Scientist sets a target Tg and exploration factor
  3. UCB acquisition ranks all 1M candidates in milliseconds
  4. Top candidates shown with predicted Tg + uncertainty
  5. Scientist logs actual DSC lab result → saved to new_lab_results.csv
  6. Run retrain_loop.py + batch_inference.py to close the loop
"""

import streamlit as st
import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PREDICTIONS_FILE  = "virtual_library_predictions.parquet"
LAB_RESULTS_FILE  = "new_lab_results.csv"
TOP_N_CANDIDATES  = 20     # How many candidates to show in the table
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Polymer Active Learning Engine",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    code, .stCode              { font-family: 'IBM Plex Mono', monospace !important; }

    .metric-card {
        background: #0f1117;
        border: 1px solid #2a2d3e;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .metric-card .label { color: #7f8ea3; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .metric-card .value { color: #e8f4f8; font-size: 1.6rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
    .metric-card .delta { color: #4fc3f7; font-size: 0.8rem; }

    .winner-card {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 100%);
        border: 1px solid #4fc3f7;
        border-radius: 10px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
    }
    .winner-smiles {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.95rem;
        color: #80deea;
        word-break: break-all;
        background: #070d16;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        margin-top: 0.5rem;
    }
    .section-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #7f8ea3;
        margin-bottom: 0.2rem;
    }
    .iteration-badge {
        background: #4fc3f7;
        color: #000;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    div[data-testid="stMetric"] { background: #0f1117; border: 1px solid #2a2d3e; border-radius: 8px; padding: 0.75rem 1rem; }
    div[data-testid="stMetric"] label { color: #7f8ea3 !important; font-size: 0.75rem !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e8f4f8 !important; font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached so it only happens once) ─────────────────────────────
@st.cache_data(show_spinner="Loading virtual library predictions...")
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def count_lab_results() -> int:
    if not os.path.exists(LAB_RESULTS_FILE):
        return 0
    return len(pd.read_csv(LAB_RESULTS_FILE))


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🧪 Polymer Active Learning Engine")
st.markdown("**Bayesian Optimization · Gaussian Process · UCB Acquisition**")

# Active learning iteration tracker
n_lab = count_lab_results()
iteration = n_lab + 1
col_iter, col_loop = st.columns([2, 8])
col_iter.markdown(f'<span class="iteration-badge">AL Iteration {iteration}</span>', unsafe_allow_html=True)
if n_lab > 0:
    col_loop.caption(f"🔁 {n_lab} lab result(s) logged — run `retrain_loop.py` then `batch_inference.py` to update predictions.")

st.divider()

# ── Check parquet exists ──────────────────────────────────────────────────────
if not os.path.exists(PREDICTIONS_FILE):
    st.error(
        f"**`{PREDICTIONS_FILE}` not found.**\n\n"
        "You need to run the offline batch inference first:\n"
        "```\npython batch_inference.py\n```\n"
        "This is a one-time step that pre-computes GP predictions for your entire virtual library."
    )
    st.stop()

library_df = load_predictions(PREDICTIONS_FILE)

# ── Library stats in sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Library Stats")
    st.metric("Total Molecules", f"{len(library_df):,}")
    st.metric("Tg Range",
              f"{library_df['Predicted_Tg'].min():.0f} – {library_df['Predicted_Tg'].max():.0f} K")
    st.metric("Mean Uncertainty", f"± {library_df['Uncertainty'].mean():.2f} K")
    st.metric("High-Uncertainty Molecules",
              f"{(library_df['Uncertainty'] > library_df['Uncertainty'].quantile(0.9)).sum():,}")

    st.divider()
    st.markdown("### 🔁 Active Learning Loop")
    st.markdown("""
**How the cycle works:**
1. Dashboard ranks 1M molecules → you pick the top candidate
2. Scientist synthesizes it in the lab (DSC measurement)
3. Log the real Tg below (Step 2 on main panel)
4. Run `retrain_loop.py` to update the GP model
5. Run `batch_inference.py` to re-score all 1M molecules
6. Repeat → model gets smarter with every iteration
""")

    st.divider()
    st.caption("Files in use:")
    st.caption(f"📂 `{PREDICTIONS_FILE}`")
    st.caption(f"📂 `{LAB_RESULTS_FILE}`")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — SCREEN THE VIRTUAL LIBRARY
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### Step 1 — Screen Virtual Library")
st.markdown("Set your synthesis goal. The UCB acquisition function balances predicted performance vs. model uncertainty.")

ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])

with ctrl1:
    target_tg = st.number_input(
        "🎯 Target $T_g$ (K)",
        min_value=float(library_df["Predicted_Tg"].min()),
        max_value=float(library_df["Predicted_Tg"].max()),
        value=350.0,
        step=5.0,
        help="The glass transition temperature you want to achieve."
    )

with ctrl2:
    beta = st.slider(
        "🔭 Exploration Factor (β)",
        min_value=0.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help=(
            "β = 0 → pure exploitation (trust the model's best prediction)\n"
            "β = 5 → high exploration (prefer uncertain, unexplored molecules)\n\n"
            "Early in your campaign: use higher β. Later: lower β."
        )
    )

with ctrl3:
    st.markdown("<br>", unsafe_allow_html=True)
    run_screen = st.button("▶ Run Screen", type="primary", use_container_width=True)


if run_screen:
    with st.spinner(f"Scoring {len(library_df):,} molecules..."):
        # ── UCB with target penalty ───────────────────────────────────────────
        # Score = -(|predicted - target|) + β * σ
        # Maximising this balances closeness to target with exploration bonus
        target_penalty = -np.abs(library_df["Predicted_Tg"].values - target_tg)
        acq_scores     = target_penalty + beta * library_df["Uncertainty"].values

        results_df = library_df.copy()
        results_df["Acquisition_Score"] = acq_scores
        results_df["Target_Distance_K"] = np.abs(results_df["Predicted_Tg"] - target_tg).round(2)

        top_candidates = (
            results_df
            .sort_values("Acquisition_Score", ascending=False)
            .head(TOP_N_CANDIDATES)
            .reset_index(drop=True)
        )

    # ── #1 Recommended candidate ──────────────────────────────────────────────
    winner = top_candidates.iloc[0]

    st.markdown("#### 🏆 Top Recommended Candidate")
    st.markdown(f"""
<div class="winner-card">
    <div class="section-label">SMILES Structure</div>
    <div class="winner-smiles">{winner['SMILES']}</div>
    <br/>
</div>
""", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted $T_g$",      f"{winner['Predicted_Tg']:.2f} K")
    m2.metric("Uncertainty (σ)",      f"± {winner['Uncertainty']:.2f} K")
    m3.metric("Distance to Target",   f"{winner['Target_Distance_K']:.2f} K")
    m4.metric("Acquisition Score",    f"{winner['Acquisition_Score']:.3f}")

    conf_pct = max(0, 100 - (winner['Uncertainty'] / winner['Predicted_Tg'] * 100))
    if winner['Uncertainty'] > library_df['Uncertainty'].quantile(0.8):
        st.info(
            "ℹ️ **High uncertainty** — This molecule lies in an under-explored region. "
            "Synthesizing it will give your model the most information gain (exploration)."
        )
    else:
        st.success(
            "✅ **High confidence prediction** — The model has seen similar structures. "
            "This candidate is a reliable bet (exploitation)."
        )

    # ── Full top-N table ──────────────────────────────────────────────────────
    st.markdown(f"#### Top {TOP_N_CANDIDATES} Candidates")
    display_df = top_candidates[
        ["SMILES", "Predicted_Tg", "Uncertainty", "Target_Distance_K", "Acquisition_Score"]
    ].copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df.columns = [
        "SMILES", "Predicted Tg (K)", "Uncertainty ± (K)",
        "Distance to Target (K)", "Acquisition Score"
    ]
    st.dataframe(display_df, use_container_width=True)

    # ── Distribution chart ────────────────────────────────────────────────────
    st.markdown("#### Predicted $T_g$ Distribution — Full Library")
    hist_data = library_df["Predicted_Tg"].values
    bins      = np.linspace(hist_data.min(), hist_data.max(), 80)
    counts, edges = np.histogram(hist_data, bins=bins)
    chart_df = pd.DataFrame({
        "Tg (K)": edges[:-1],
        "Count":  counts,
    }).set_index("Tg (K)")
    st.bar_chart(chart_df, height=220)

    # Store winner in session state for the "Close the Loop" section
    st.session_state["last_winner_smiles"] = winner["SMILES"]
    st.session_state["last_winner_tg"]     = round(float(winner["Predicted_Tg"]), 2)


st.divider()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — CLOSE THE LOOP
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### Step 2 — Log Lab Result (Close the Loop)")
st.markdown(
    "Once you've synthesized the top candidate and measured its $T_g$ by DSC, "
    "enter the result here. This queues it for model retraining."
)

# Pre-fill with the last winner if available
default_smiles = st.session_state.get("last_winner_smiles", "")
default_tg     = st.session_state.get("last_winner_tg", 300.0)

with st.form("lab_result_form", clear_on_submit=True):
    form_col1, form_col2 = st.columns([3, 1])

    tested_smiles = form_col1.text_input(
        "SMILES of synthesized polymer",
        value=default_smiles,
        placeholder="Paste the SMILES you actually made in the lab"
    )
    actual_tg = form_col2.number_input(
        "Measured $T_g$ (K, DSC)",
        min_value=0.0,
        max_value=2000.0,
        value=float(default_tg),
        step=1.0
    )

    submitted = st.form_submit_button("💾 Log Lab Result", type="primary")

    if submitted:
        if not tested_smiles.strip():
            st.error("Please enter a SMILES string.")
        else:
            new_row = pd.DataFrame({
                "SMILES":    [tested_smiles.strip()],
                "Actual_Tg": [actual_tg],
                "Iteration": [iteration],
            })

            if os.path.exists(LAB_RESULTS_FILE):
                existing = pd.read_csv(LAB_RESULTS_FILE)
                updated  = pd.concat([existing, new_row], ignore_index=True)
            else:
                updated = new_row

            updated.to_csv(LAB_RESULTS_FILE, index=False)

            st.success(
                f"✅ **Lab result saved!** Iteration {iteration} complete.\n\n"
                f"**Next steps:**\n"
                f"1. Run `python retrain_loop.py` to incorporate this new data point\n"
                f"2. Run `python batch_inference.py` to re-score all {len(library_df):,} molecules\n"
                f"3. Return here — the model will now recommend a smarter candidate"
            )
            st.balloons()


# ── Previous lab results log ──────────────────────────────────────────────────
if os.path.exists(LAB_RESULTS_FILE):
    lab_df = pd.read_csv(LAB_RESULTS_FILE)
    if not lab_df.empty:
        st.divider()
        st.markdown(f"#### 📋 Lab Results Log ({len(lab_df)} entries)")
        st.dataframe(lab_df, use_container_width=True)

        # Show error between predicted and actual if we can cross-reference
        if "SMILES" in library_df.columns and "Actual_Tg" in lab_df.columns:
            merged = lab_df.merge(
                library_df[["SMILES", "Predicted_Tg"]],
                on="SMILES", how="left"
            )
            if "Predicted_Tg" in merged.columns:
                merged["Error_K"]    = (merged["Actual_Tg"] - merged["Predicted_Tg"]).round(2)
                merged["Error_%"]    = ((merged["Error_K"] / merged["Actual_Tg"]) * 100).round(1)
                st.markdown("**Prediction accuracy on synthesized molecules:**")
                st.dataframe(
                    merged[["SMILES", "Predicted_Tg", "Actual_Tg", "Error_K", "Error_%"]],
                    use_container_width=True
                )
