import pandas as pd

from model import load_artifacts, predict

STABLE_EXAMPLE = {
    "tau1": 2.0, "tau2": 2.0, "tau3": 2.0, "tau4": 2.0,
    "p1": 3.0, "p2": -1.0, "p3": -1.0, "p4": -1.0,
    "g1": 0.1, "g2": 0.1, "g3": 0.1, "g4": 0.1,
}

UNSTABLE_EXAMPLE = {
    "tau1": 5.0, "tau2": 5.0, "tau3": 5.0, "tau4": 5.0,
    "p1": 4.5, "p2": -1.5, "p3": -1.5, "p4": -1.5,
    "g1": 0.8, "g2": 0.8, "g3": 0.8, "g4": 0.8,
}


def test_predicts_stable_example():
    model, scaler, le, feature_cols = load_artifacts()
    row = pd.DataFrame([STABLE_EXAMPLE])[feature_cols]
    label, probs = predict(row, model, scaler, le)
    assert label == "stable"
    assert set(probs) == {"stable", "unstable"}


def test_predicts_unstable_example():
    model, scaler, le, feature_cols = load_artifacts()
    row = pd.DataFrame([UNSTABLE_EXAMPLE])[feature_cols]
    label, probs = predict(row, model, scaler, le)
    assert label == "unstable"
