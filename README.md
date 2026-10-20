# Grid Stability Predictor

A Streamlit app that predicts whether a 4-generator electrical grid is
**stable** or **unstable** from 12 electrical parameters, using an
XGBoost classifier.

## Dataset

[Electrical Grid Stability Simulated Data](https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data)
(UCI Machine Learning Repository). 10,000 simulated observations of a
4-node star grid (1 producer, 3 consumers) with a differential-equation
model of grid stability.

The 12 input features:

| Feature | Meaning | Range in this dataset |
|---|---|---|
| `tau1`–`tau4` | Reaction time of each node (seconds) | 0.5 – 10.0 |
| `p1` | Power produced by the generator node. Always positive. | ~1.6 – 5.9 |
| `p2`–`p4` | Power consumed by each consumer node. Always negative. | ~-2.0 – -0.5 |
| `g1`–`g4` | Price elasticity / coupling gamma for each node | 0.05 – 1.0 |

The dataset also includes `stab` (a continuous stability margin) and
`stabf` (its sign, as `stable`/`unstable`). `train.py` predicts `stabf`
and drops `stab`, since `stabf` is directly derived from it and would
leak the answer.

`data/sg_raw_data.csv` (10k rows, vendored in this repo) is the source
used for the shipped model. `data/sg_aug_data.csv` (60k rows, the
standard generator-symmetry augmentation) is used on the
`experiment/augmented-60k` branch, which trains a second model at
~98% accuracy for comparison — not merged into `main` by default.

## Model

- XGBoost classifier, scikit-learn `StandardScaler` + `LabelEncoder`
- Trained on an 80/20 stratified split, fixed random seed (42)
- **Test accuracy: 94.65%** on the 10k dataset with XGBoost 3.2 (verified by re-running
  `train.py`; see `experiment/augmented-60k` for the 60k comparison)

## Running locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Retraining

```bash
python train.py  # trains on data/sg_raw_data.csv (10k)
```

The `experiment/augmented-60k` branch adds a `--augmented` flag to
train on the 60k dataset instead; not present on `main`.

Regenerates `grid_stability_model.pkl`, `scaler.pkl`,
`label_encoder.pkl`, and `feature_columns.pkl` in place.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Codespaces

Opening this repo in a GitHub Codespace runs the app automatically via
`.devcontainer/devcontainer.json` (`streamlit run app.py` on attach,
port 8501 forwarded). CORS/XSRF protection is disabled in that config
for Codespaces convenience only — do not reuse those flags if deploying
this app anywhere else.
