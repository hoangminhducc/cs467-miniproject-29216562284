"""Recommendation engines for the Open-Source Library Recommender (MP-C02).

1. PopularityRecommender    - weighted (Bayesian) average rating with a minimum
                              rating-count threshold.
2. ContentRecommender       - cosine similarity over NLP TF-IDF vectors:
                              (a) free-text query, (b) item-to-item,
                              (c) user profile built from the user's liked items.
3. ItemCFRecommender        - item-based collaborative filtering (adjusted cosine).
4. HybridRecommender        - project extension: weighted blend of
                              NLP content + item-CF + popularity. Components that
                              are unavailable for an item (e.g. a cold-start item
                              with zero ratings) are skipped and the remaining
                              weights are re-normalised for that item.

All recommenders return a list of dicts:
    {item_id, package_name, language, category, score, keywords, ...}
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from nlp_pipeline import NLPPipeline, build_item_doc, build_item_docs

LIKE_THRESHOLD = 4  # rating >= 4 counts as "liked" (relevant)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalise ignoring NaN (NaN stays NaN)."""
    out = np.full_like(x, np.nan, dtype=float)
    ok = ~np.isnan(x)
    if ok.sum() == 0:
        return out
    lo, hi = x[ok].min(), x[ok].max()
    out[ok] = 0.5 if hi - lo < 1e-12 else (x[ok] - lo) / (hi - lo)
    return out


class _RatingsMixin:
    def _index_ratings(self, ratings: pd.DataFrame) -> None:
        self.ratings = ratings.copy()
        self.user_ratings: dict[str, dict[str, float]] = {}
        for u, i, r in ratings[["user_id", "item_id", "rating"]].itertuples(index=False):
            self.user_ratings.setdefault(u, {})[i] = float(r)
        self.global_mean = float(ratings["rating"].mean()) if len(ratings) else 3.0

    def rated_by(self, user_id: str) -> dict[str, float]:
        return self.user_ratings.get(user_id, {})

    def user_mean(self, user_id: str) -> float:
        r = self.rated_by(user_id)
        return float(np.mean(list(r.values()))) if r else self.global_mean


def _result(items: pd.DataFrame, idx: int, score: float, keywords=None, **extra) -> dict:
    row = items.iloc[idx]
    out = {"item_id": row["item_id"], "package_name": row["package_name"],
           "language": row["language"], "category": row["category"],
           "score": float(score), "keywords": keywords or []}
    out.update(extra)
    return out


# --------------------------------------------------------------------------- #
# 1. Popularity baseline
# --------------------------------------------------------------------------- #
class PopularityRecommender(_RatingsMixin):
    """Weighted rating  WR = v/(v+m) * R + m/(v+m) * C
    (v = #ratings of the item, R = item mean, C = global mean, m = min_count).
    Items with fewer than `min_count` ratings are not recommended."""

    def __init__(self, items: pd.DataFrame, min_count: int = 2):
        self.items = items.reset_index(drop=True)
        self.min_count = min_count

    def fit(self, ratings: pd.DataFrame) -> "PopularityRecommender":
        self._index_ratings(ratings)
        g = ratings.groupby("item_id")["rating"].agg(["mean", "count"])
        m, C = self.min_count, self.global_mean
        g["wr"] = g["count"] / (g["count"] + m) * g["mean"] + m / (g["count"] + m) * C
        self.stats = g
        return self

    def item_score(self, item_id: str) -> float:
        """Weighted rating, or NaN when the item has fewer than min_count ratings."""
        if item_id not in self.stats.index or self.stats.loc[item_id, "count"] < self.min_count:
            return float("nan")
        return float(self.stats.loc[item_id, "wr"])

    def predict(self, user_id: str, item_id: str) -> float:
        if item_id in self.stats.index:
            return float(self.stats.loc[item_id, "wr"])
        return self.global_mean

    def recommend(self, user_id: str | None = None, k: int = 5, exclude: Iterable[str] = ()) -> list[dict]:
        seen = set(exclude) | (set(self.rated_by(user_id)) if user_id else set())
        scored = []
        for idx, iid in enumerate(self.items["item_id"]):
            s = self.item_score(iid)
            if iid not in seen and not np.isnan(s):
                scored.append((s, idx))
        scored.sort(reverse=True)
        return [_result(self.items, idx, s,
                        n_ratings=int(self.stats.loc[self.items.item_id[idx], "count"]))
                for s, idx in scored[:k]]


# --------------------------------------------------------------------------- #
# 2. NLP content-based recommender
# --------------------------------------------------------------------------- #
class ContentRecommender(_RatingsMixin):
    def __init__(self, items: pd.DataFrame, nlp: NLPPipeline | None = None):
        self.items = items.reset_index(drop=True).copy()
        self.nlp = nlp or NLPPipeline()
        if not self.nlp.feature_names.size:
            self.nlp.fit(build_item_docs(self.items))
        self.X = self.nlp.transform(build_item_docs(self.items)).tocsr()
        self.id_to_idx = {iid: i for i, iid in enumerate(self.items["item_id"])}
        self._index_ratings(pd.DataFrame(columns=["user_id", "item_id", "rating"]))

    def fit(self, ratings: pd.DataFrame) -> "ContentRecommender":
        self._index_ratings(ratings)
        return self

    # -- catalogue management (cold start) -------------------------------- #
    def add_item(self, item: dict) -> int:
        """Add a brand-new library (no ratings). Its description is projected into the
        SAME TF-IDF vocabulary, so it is immediately matchable."""
        vec = self.nlp.transform([build_item_doc(item)])
        self.X = sparse.vstack([self.X, vec]).tocsr()
        self.items = pd.concat([self.items, pd.DataFrame([item])], ignore_index=True)
        self.id_to_idx[item["item_id"]] = len(self.items) - 1
        return len(self.items) - 1

    # -- similarity primitives ------------------------------------------- #
    def sims_to_vector(self, vec) -> np.ndarray:
        """Cosine similarity of a (L2-normalised) TF-IDF vector to every item."""
        v = vec.toarray().ravel() if sparse.issparse(vec) else np.asarray(vec).ravel()
        n = np.linalg.norm(v)
        if n == 0:
            return np.zeros(self.X.shape[0])
        return np.asarray(self.X @ (v / n)).ravel()

    def sims_to_item(self, item_id: str) -> np.ndarray:
        return self.sims_to_vector(self.X[self.id_to_idx[item_id]])

    # -- (a) free-text query --------------------------------------------- #
    def recommend_query(self, query: str, k: int = 5, exclude: Iterable[str] = ()) -> list[dict]:
        qv = self.nlp.vectorize_query(query)
        sims = self.sims_to_vector(qv)
        return self._top_k(sims, k, set(exclude), lambda i: self.nlp.shared_keywords(qv, self.X[i]))

    # -- (b) item-to-item -------------------------------------------------- #
    def similar_items(self, item_id: str, k: int = 5) -> list[dict]:
        src = self.X[self.id_to_idx[item_id]]
        sims = self.sims_to_vector(src)
        return self._top_k(sims, k, {item_id}, lambda i: self.nlp.shared_keywords(src, self.X[i]))

    # -- (c) user profile ---------------------------------------------------- #
    def user_profile(self, user_id: str) -> np.ndarray | None:
        """Weighted sum of the TF-IDF vectors of items the user liked
        (weight = rating - 3, so a 5 counts twice a 4). Falls back to all rated
        items weighted by rating when the user liked nothing."""
        rated = {i: r for i, r in self.rated_by(user_id).items() if i in self.id_to_idx}
        if not rated:
            return None
        liked = {i: r - 3 for i, r in rated.items() if r >= LIKE_THRESHOLD}
        weights = liked or rated
        prof = np.zeros(self.X.shape[1])
        for iid, w in weights.items():
            prof += w * self.X[self.id_to_idx[iid]].toarray().ravel()
        return prof

    def recommend_user(self, user_id: str, k: int = 5, exclude: Iterable[str] = ()) -> list[dict]:
        prof = self.user_profile(user_id)
        if prof is None:
            return []
        sims = self.sims_to_vector(prof)
        seen = set(exclude) | set(self.rated_by(user_id))
        pv = prof / (np.linalg.norm(prof) or 1.0)
        return self._top_k(sims, k, seen, lambda i: self.nlp.shared_keywords(pv, self.X[i]))

    def user_scores(self, user_id: str) -> np.ndarray | None:
        prof = self.user_profile(user_id)
        return None if prof is None else self.sims_to_vector(prof)

    # -- rating prediction (for RMSE/MAE) ----------------------------------- #
    def predict(self, user_id: str, item_id: str, k_nn: int = 5) -> float:
        """Similarity-weighted mean of the user's ratings of the k most similar items."""
        rated = {i: r for i, r in self.rated_by(user_id).items()
                 if i in self.id_to_idx and i != item_id}
        if not rated:
            return self.global_mean
        sims = self.sims_to_item(item_id)
        cand = sorted(((sims[self.id_to_idx[i]], r) for i, r in rated.items()), reverse=True)[:k_nn]
        num = sum(s * r for s, r in cand)
        den = sum(s for s, _ in cand)
        return num / den if den > 1e-9 else self.user_mean(user_id)

    # -- internals ----------------------------------------------------------- #
    def _top_k(self, sims: np.ndarray, k: int, exclude: set, kw_fn) -> list[dict]:
        order = np.argsort(-sims)
        out = []
        for idx in order:
            if self.items["item_id"][idx] in exclude or sims[idx] <= 0:
                continue
            out.append(_result(self.items, idx, sims[idx], kw_fn(idx)))
            if len(out) == k:
                break
        return out


# --------------------------------------------------------------------------- #
# 3. Item-based collaborative filtering
# --------------------------------------------------------------------------- #
class ItemCFRecommender(_RatingsMixin):
    """Item-item adjusted-cosine similarity on user-mean-centred ratings, shrunk by
    the number of co-raters (sim * n/(n+shrink)) because the matrix is small."""

    def __init__(self, items: pd.DataFrame, k_neighbors: int = 10, shrink: float = 3.0):
        self.items = items.reset_index(drop=True)
        self.k_neighbors, self.shrink = k_neighbors, shrink

    def fit(self, ratings: pd.DataFrame) -> "ItemCFRecommender":
        self._index_ratings(ratings)
        self.item_ids = list(self.items["item_id"])
        self.col = {iid: j for j, iid in enumerate(self.item_ids)}
        users = sorted(ratings["user_id"].unique())
        self.row = {u: i for i, u in enumerate(users)}
        R = np.zeros((len(users), len(self.item_ids)))
        M = np.zeros_like(R)
        for u, i, r in ratings[["user_id", "item_id", "rating"]].itertuples(index=False):
            if i in self.col:
                R[self.row[u], self.col[i]] = r
                M[self.row[u], self.col[i]] = 1
        self.R, self.M = R, M
        means = np.array([R[i][M[i] > 0].mean() if M[i].sum() else 0 for i in range(len(users))])
        Rc = (R - means[:, None]) * M
        num = Rc.T @ Rc
        norm = np.sqrt((Rc ** 2).sum(axis=0))
        den = np.outer(norm, norm)
        sim = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
        co = M.T @ M
        sim *= co / (co + self.shrink)
        np.fill_diagonal(sim, 0.0)
        self.sim = sim
        self.item_counts = M.sum(axis=0)
        return self

    def has_ratings(self, item_id: str) -> bool:
        return item_id in self.col and self.item_counts[self.col[item_id]] > 0

    def _neighbors(self, user_id: str, item_id: str):
        """(similarity, rating) of the user's rated items most similar to item_id (sim > 0)."""
        j = self.col[item_id]
        pairs = [(self.sim[j, self.col[i]], r) for i, r in self.rated_by(user_id).items()
                 if i in self.col and i != item_id and self.sim[j, self.col[i]] > 0]
        return sorted(pairs, reverse=True)[: self.k_neighbors]

    def score(self, user_id: str, item_id: str) -> float:
        """Ranking score: sum(sim * rating/5) over positive neighbours; NaN if none."""
        if not self.has_ratings(item_id) or user_id not in self.user_ratings:
            return float("nan")
        nb = self._neighbors(user_id, item_id)
        return float(sum(s * r / 5.0 for s, r in nb)) if nb else float("nan")

    def predict(self, user_id: str, item_id: str) -> float:
        """Rating prediction: user mean + weighted deviation of neighbours."""
        if not self.has_ratings(item_id) or user_id not in self.user_ratings:
            return float("nan")
        nb = self._neighbors(user_id, item_id)
        if not nb:
            return float("nan")
        mu = self.user_mean(user_id)
        den = sum(s for s, _ in nb)
        return mu + sum(s * (r - mu) for s, r in nb) / den

    def item_sims(self, item_id: str) -> np.ndarray:
        """Co-rating similarity of item_id to every catalogue item (NaN when unavailable)."""
        out = np.full(len(self.item_ids), np.nan)
        if self.has_ratings(item_id):
            out[:] = self.sim[self.col[item_id]]
        return out

    def recommend(self, user_id: str, k: int = 5, exclude: Iterable[str] = ()) -> list[dict]:
        seen = set(exclude) | set(self.rated_by(user_id))
        scored = [(self.score(user_id, iid), idx) for idx, iid in enumerate(self.item_ids)
                  if iid not in seen]
        scored = sorted([(s, i) for s, i in scored if not np.isnan(s)], reverse=True)
        return [_result(self.items, idx, s) for s, idx in scored[:k]]


# --------------------------------------------------------------------------- #
# 4. Hybrid recommender (project extension)
# --------------------------------------------------------------------------- #
class HybridRecommender:
    """score(u,i) = w_c * content(u,i) + w_cf * itemCF(u,i) + w_p * popularity(i)
    Each component is min-max normalised over the candidate items; components that are
    NaN for an item (cold start / too few ratings) are dropped and the remaining
    weights re-normalised for that item."""

    def __init__(self, items: pd.DataFrame, weights=(0.4, 0.4, 0.2), nlp: NLPPipeline | None = None,
                 min_count: int = 2, query_pop_weight: float = 0.1):
        self.w_content, self.w_cf, self.w_pop = weights
        self.query_pop_weight = query_pop_weight
        self.content = ContentRecommender(items, nlp)
        self.cf = ItemCFRecommender(items)
        self.pop = PopularityRecommender(items, min_count)
        self.items = self.content.items

    def fit(self, ratings: pd.DataFrame) -> "HybridRecommender":
        self.content.fit(ratings)
        self.cf.fit(ratings)
        self.pop.fit(ratings)
        return self

    @property
    def nlp(self) -> NLPPipeline:
        return self.content.nlp

    def add_item(self, item: dict) -> int:
        """Cold start: only the content component can score the new item."""
        idx = self.content.add_item(item)
        self.items = self.content.items
        self.pop.items = self.items
        return idx

    def _combine(self, comps: list[tuple[float, np.ndarray]]) -> np.ndarray:
        num = np.zeros(len(self.items))
        den = np.zeros(len(self.items))
        for w, arr in comps:
            ok = ~np.isnan(arr)
            num[ok] += w * arr[ok]
            den[ok] += w
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    def _pop_vec(self) -> np.ndarray:
        return np.array([self.pop.item_score(i) for i in self.items["item_id"]])

    def _cf_vec(self, user_id: str) -> np.ndarray:
        return np.array([self.cf.score(user_id, i) if i in self.cf.col else np.nan
                         for i in self.items["item_id"]])

    def recommend_user(self, user_id: str, k: int = 5, exclude: Iterable[str] = ()) -> list[dict]:
        seen = set(exclude) | set(self.content.rated_by(user_id))
        cand = np.array([i not in seen for i in self.items["item_id"]])
        csc = self.content.user_scores(user_id)
        if csc is None:                                   # cold-start USER -> popularity
            return self.pop.recommend(user_id, k, exclude)
        comps = []
        for w, arr in ((self.w_content, csc), (self.w_cf, self._cf_vec(user_id)), (self.w_pop, self._pop_vec())):
            arr = np.where(cand, arr, np.nan)
            comps.append((w, _minmax(arr)))
        total = np.where(cand, self._combine(comps), -1.0)
        prof = self.content.user_profile(user_id)
        pv = prof / (np.linalg.norm(prof) or 1.0)
        out = []
        for idx in np.argsort(-total)[:k]:
            parts = {n: (None if np.isnan(a[idx]) else float(a[idx]))
                     for n, (_, a) in zip(("content", "cf", "popularity"), comps)}
            out.append(_result(self.items, idx, total[idx],
                               self.nlp.shared_keywords(pv, self.content.X[idx]), components=parts))
        return out

    def recommend_query(self, query: str, k: int = 5, exclude: Iterable[str] = ()) -> list[dict]:
        """Text query: content similarity blended with a small popularity prior
        (query_pop_weight) that mostly acts as a tie-breaker between relevant items."""
        qv = self.nlp.vectorize_query(query)
        sims = self.content.sims_to_vector(qv)
        ex = set(exclude)
        cand = np.array([i not in ex for i in self.items["item_id"]])
        comps = [(1.0 - self.query_pop_weight, _minmax(np.where(cand, sims, np.nan))),
                 (self.query_pop_weight, _minmax(np.where(cand, self._pop_vec(), np.nan)))]
        total = np.where(cand & (sims > 0), self._combine(comps), -1.0)
        out = []
        for idx in np.argsort(-total)[:k]:
            if total[idx] < 0:
                break
            out.append(_result(self.items, idx, total[idx],
                               self.nlp.shared_keywords(qv, self.content.X[idx]), content_sim=float(sims[idx])))
        return out

    def similar_items(self, item_id: str, k: int = 5) -> list[dict]:
        """Item-to-item: content similarity blended with co-rating similarity."""
        csim = self.content.sims_to_item(item_id)
        cf = self.cf.item_sims(item_id)
        cf_full = np.full(len(self.items), np.nan)
        cf_full[:len(cf)] = cf
        cand = np.array([i != item_id for i in self.items["item_id"]])
        comps = [(self.w_content + self.w_pop, _minmax(np.where(cand, csim, np.nan))),
                 (self.w_cf, _minmax(np.where(cand, np.clip(cf_full, 0, None), np.nan)))]
        total = np.where(cand, self._combine(comps), -1.0)
        src = self.content.X[self.content.id_to_idx[item_id]]
        return [_result(self.items, idx, total[idx],
                        self.nlp.shared_keywords(src, self.content.X[idx]))
                for idx in np.argsort(-total)[:k]]

    def predict(self, user_id: str, item_id: str) -> float:
        preds = [(self.w_content, self.content.predict(user_id, item_id)),
                 (self.w_cf, self.cf.predict(user_id, item_id)),
                 (self.w_pop, self.pop.predict(user_id, item_id))]
        preds = [(w, p) for w, p in preds if not np.isnan(p)]
        return sum(w * p for w, p in preds) / sum(w for w, _ in preds)


def load_data(data_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    from pathlib import Path
    d = Path(data_dir)
    return pd.read_csv(d / "items.csv"), pd.read_csv(d / "ratings.csv")
