"""Generate synthetic claims with planted flaws and known correct verdicts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "cases"
K = 4  # number of feature columns

GOOD_NOTES = "Stratified 5-fold cross-validation; hyperparameters tuned on a separate validation set."


def feats(rng, n):
    return pd.DataFrame(rng.normal(size=(n, K)).round(4), columns=[f"f{i}" for i in range(K)])


def preds_df(rng, n, acc=None, pos_rate=0.5, all_zero=False):
    X = feats(rng, n)
    y = (rng.random(n) < pos_rate).astype(int)
    if all_zero:
        yp = np.zeros(n, dtype=int)
    else:
        yp = np.where(rng.random(n) < acc, y, 1 - y)
    X["y_true"], X["y_pred"] = y, yp
    return X


def train_df(rng, n=20000, pos_rate=0.5, extra=None):
    X = feats(rng, n)
    X["y_true"] = (rng.random(n) < pos_rate).astype(int)
    if extra is not None:
        X = pd.concat([X, extra[[f"f{i}" for i in range(K)] + ["y_true"]]], ignore_index=True)
    return X


def paper(value, baseline=True):
    s = f"Our model reaches an accuracy of {value * 100:.1f}% on the held-out test set."
    if baseline:
        s += " A majority-class baseline scores far lower and logistic regression is compared with it."
    return s


def save(name, test, train, claimed, notes, paper_text, expected, flaw):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "expected": expected, "flaw": flaw,
            "claim": {"value": claimed, "text": f"{claimed:.1%} accuracy",
                      "eval_notes": notes, "paper_text": paper_text}}
    if test is not None:
        test.to_csv(d / "preds.csv", index=False)
        meta["claim"]["predictions_path"] = f"data/cases/{name}/preds.csv"
    if train is not None:
        train.to_csv(d / "train.csv", index=False)
        meta["claim"]["train_path"] = f"data/cases/{name}/train.csv"
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def acc_of(df):
    return round(float((df.y_true == df.y_pred).mean()), 3)


def main():
    rng = np.random.default_rng(7)

    t = preds_df(rng, 5000, acc=0.90)
    c = acc_of(t)
    save("clean", t, train_df(rng), c, GOOD_NOTES, paper(c), "supported", None)

    t = preds_df(rng, 5000, pos_rate=0.02, all_zero=True)
    c = acc_of(t)
    save("imbalance", t, train_df(rng, pos_rate=0.02), c, GOOD_NOTES,
         paper(c, baseline=False), "contradicted", "class_balance")

    t = preds_df(rng, 5000, acc=0.85)
    save("inflated", t, train_df(rng), 0.95, GOOD_NOTES, "", "contradicted", "recompute_metric")

    t = preds_df(rng, 40, acc=0.95)
    c = acc_of(t)
    save("tiny_test", t, train_df(rng), c, GOOD_NOTES, paper(c), "insufficient_evidence", "sample_size")

    t = preds_df(rng, 5000, acc=0.97)
    c = acc_of(t)
    leak = t.sample(frac=0.4, random_state=1)
    save("leakage", t, train_df(rng, extra=leak), c, GOOD_NOTES, paper(c), "contradicted", "leakage")

    t = preds_df(rng, 5000, acc=0.92)
    c = acc_of(t)
    save("tuned_on_test", t, train_df(rng), c,
         "Hyperparameters were tuned on the test set, then the best configuration was reported.",
         paper(c), "contradicted", "eval_method")

    save("paper_mismatch", None, None, 0.98, "",
         "Our model reaches an accuracy of 91.2% on the held-out test set.",
         "contradicted", "paper_check")
    print(f"Wrote 7 cases to {OUT}")


if __name__ == "__main__":
    main()
