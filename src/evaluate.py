"""Offline evaluation + charts for the Open-Source Library Recommender (MP-C02).

Protocol
--------
* For every developer, ~20% of their ratings (at least 2) are hidden as a test set;
  the rest is the training set. All recommenders are fitted on the training set only.
* Ranking task  : each method recommends the Top-5 items the user has NOT rated in
  training. A hidden item counts as relevant when its rating >= 4.
      Precision@5 = hits / 5,  Recall@5 = hits / #relevant,  HitRate@5 = 1[hits > 0]
  (users with no relevant hidden item are skipped).
* Rating task   : RMSE / MAE of predicted vs. true rating on all hidden ratings.
* Because the data set is tiny, the whole protocol is repeated with N_REPEATS random
  splits and the mean (and std) over splits is reported.

Run:  python src/evaluate.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recommender import (LIKE_THRESHOLD, ContentRecommender, HybridRecommender, ItemCFRecommender,
                         PopularityRecommender, load_data)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
K = 5
N_REPEATS = 30
TEST_FRAC = 0.2
METHODS = ["Random", "Popularity", "Content-Based (NLP)", "Item-CF", "Hybrid (NLP+CF+Pop)"]


# --------------------------------------------------------------------------- #
def split_ratings(ratings: pd.DataFrame, seed: int, test_frac: float = TEST_FRAC):
    """Per-user hold-out split (>= 2 hidden ratings per user)."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, grp in ratings.groupby("user_id"):
        n_test = max(2, int(round(len(grp) * test_frac)))
        test_idx += list(rng.choice(grp.index.to_numpy(), size=n_test, replace=False))
    test = ratings.loc[test_idx]
    return ratings.drop(index=test_idx), test


def precision_recall_hit(recs: list[str], relevant: set[str], k: int = K):
    hits = len(set(recs[:k]) & relevant)
    return hits / k, hits / len(relevant), float(hits > 0)


def build_models(items: pd.DataFrame, train: pd.DataFrame, seed: int):
    hybrid = HybridRecommender(items).fit(train)
    return {
        "Popularity": hybrid.pop,
        "Content-Based (NLP)": hybrid.content,
        "Item-CF": hybrid.cf,
        "Hybrid (NLP+CF+Pop)": hybrid,
    }


def recommend_ids(name: str, model, user_id: str, items: pd.DataFrame, train_seen: set, rng) -> list[str]:
    if name == "Random":
        pool = [i for i in items["item_id"] if i not in train_seen]
        return list(rng.choice(pool, size=K, replace=False))
    if name == "Popularity":
        recs = model.recommend(user_id, K)
    else:
        recs = model.recommend_user(user_id, K) if name != "Item-CF" else model.recommend(user_id, K)
    return [r["item_id"] for r in recs]


def evaluate_split(items: pd.DataFrame, ratings: pd.DataFrame, seed: int) -> pd.DataFrame:
    train, test = split_ratings(ratings, seed)
    models = build_models(items, train, seed)
    rng = np.random.default_rng(seed + 1000)
    rows = []
    for name in METHODS:
        P, R, H, errs = [], [], [], []
        for user_id, grp in test.groupby("user_id"):
            relevant = set(grp.loc[grp.rating >= LIKE_THRESHOLD, "item_id"])
            if relevant:
                seen = set(train.loc[train.user_id == user_id, "item_id"])
                recs = recommend_ids(name, models.get(name), user_id, items, seen, rng)
                p, r, h = precision_recall_hit(recs, relevant)
                P.append(p); R.append(r); H.append(h)
            if name != "Random":
                for item_id, true_r in zip(grp.item_id, grp.rating):
                    pred = models[name].predict(user_id, item_id)
                    if np.isnan(pred):                       # item-CF with no neighbours
                        pred = models["Popularity"].predict(user_id, item_id)
                    errs.append(float(np.clip(pred, 1, 5)) - true_r)
        errs = np.array(errs)
        rows.append({"seed": seed, "method": name,
                     "precision@5": np.mean(P), "recall@5": np.mean(R), "hitrate@5": np.mean(H),
                     "rmse": np.sqrt(np.mean(errs ** 2)) if len(errs) else np.nan,
                     "mae": np.mean(np.abs(errs)) if len(errs) else np.nan})
    return pd.DataFrame(rows)


def run_evaluation(items: pd.DataFrame, ratings: pd.DataFrame, n_repeats: int = N_REPEATS) -> pd.DataFrame:
    return pd.concat([evaluate_split(items, ratings, s) for s in range(n_repeats)], ignore_index=True)


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def plot_data_overview(items: pd.DataFrame, ratings: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    counts = ratings["rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    ax[0].bar(counts.index, counts.values, color="#4C72B0")
    ax[0].set(title="Rating distribution", xlabel="rating", ylabel="# ratings")
    cat = items["category"].value_counts()
    ax[1].barh(cat.index[::-1], cat.values[::-1], color="#55A868")
    ax[1].set(title="Libraries per category", xlabel="# libraries")
    merged = ratings.merge(items[["item_id", "category"]], on="item_id")
    avg = merged.groupby("category")["rating"].mean().sort_values()
    ax[2].barh(avg.index, avg.values, color="#C44E52")
    ax[2].set(title="Average rating per category", xlabel="mean rating", xlim=(1, 5))
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_matrix(items: pd.DataFrame, ratings: pd.DataFrame, path: Path) -> None:
    mat = ratings.pivot(index="user_id", columns="item_id", values="rating").reindex(columns=items["item_id"])
    fig, ax = plt.subplots(figsize=(13, 4.8))
    im = ax.imshow(np.ma.masked_invalid(mat.values), cmap="YlGnBu", vmin=1, vmax=5, aspect="auto")
    ax.set_xticks(range(len(items))); ax.set_xticklabels(items["package_name"], rotation=90, fontsize=7)
    ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.index, fontsize=8)
    miss = mat.isna().to_numpy().mean()
    ax.set_title(f"User-item rating matrix (white = missing, {miss:.0%} of entries missing)")
    fig.colorbar(im, ax=ax, label="rating"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_metrics(summary: pd.DataFrame, path: Path) -> None:
    colors = ["#BBBBBB", "#4C72B0", "#55A868", "#DD8452", "#C44E52"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, (col, title) in zip(ax, [("precision@5", "Precision@5 (higher is better)"),
                                    ("hitrate@5", "Hit-rate@5 (higher is better)"),
                                    ("rmse", "RMSE (lower is better)")]):
        sub = summary.loc[summary[f"{col}_mean"].notna()]
        bars = a.bar(range(len(sub)), sub[f"{col}_mean"], yerr=sub[f"{col}_std"], capsize=3,
                     color=[colors[METHODS.index(m)] for m in sub.index])
        a.set_xticks(range(len(sub))); a.set_xticklabels([m.replace(" (", "\n(") for m in sub.index], fontsize=8)
        a.set_title(title)
        for b, v in zip(bars, sub[f"{col}_mean"]):
            a.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Offline comparison - mean +/- std over {N_REPEATS} random 80/20 splits per user", y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    g = results.groupby("method")[["precision@5", "recall@5", "hitrate@5", "rmse", "mae"]]
    summary = pd.concat({"mean": g.mean(), "std": g.std()}, axis=1)
    summary.columns = [f"{m}_{s}" for s, m in summary.columns]
    return summary.reindex(METHODS)


def main() -> None:
    items, ratings = load_data(ROOT / "data")
    RESULTS.mkdir(exist_ok=True)
    plot_data_overview(items, ratings, RESULTS / "data_overview.png")
    plot_matrix(items, ratings, RESULTS / "rating_matrix.png")

    results = run_evaluation(items, ratings)
    summary = summarise(results)
    results.to_csv(RESULTS / "evaluation_per_split.csv", index=False)
    summary.round(4).to_csv(RESULTS / "evaluation_summary.csv")
    plot_metrics(summary, RESULTS / "metrics_comparison.png")

    pd.set_option("display.width", 200)
    print(f"Evaluation over {N_REPEATS} random splits (test = ~{TEST_FRAC:.0%} of each user's ratings, relevant = rating >= {LIKE_THRESHOLD})\n")
    table = pd.DataFrame({
        "Precision@5": summary.apply(lambda r: f"{r['precision@5_mean']:.3f} +/- {r['precision@5_std']:.3f}", axis=1),
        "Recall@5": summary["recall@5_mean"].round(3),
        "HitRate@5": summary["hitrate@5_mean"].round(3),
        "RMSE": summary["rmse_mean"].round(3),
        "MAE": summary["mae_mean"].round(3)})
    print(table.to_string())
    (RESULTS / "evaluation_table.txt").write_text(table.to_string() + "\n")
    print("\nSaved: results/evaluation_summary.csv, results/data_overview.png, "
          "results/rating_matrix.png, results/metrics_comparison.png")


if __name__ == "__main__":
    main()
