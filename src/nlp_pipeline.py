"""NLP pipeline for the Open-Source Library Recommender (MP-C02).

Steps
-----
1. Text cleaning  : lowercase, remove punctuation / extra spaces.
2. Tokenization   : whitespace tokenization of the cleaned text.
3. Stop-words     : generic English stop-words + a few domain fillers
                    ("library", "package", ...).
4. Light stemming : tiny rule-based stemmer so "scraping", "scraper" and
                    "scrape" match each other (the same code is applied to
                    documents and to queries).
5. TF-IDF         : unigrams + bigrams (bigrams never cross a segment
                    boundary such as a tag separator or a full stop).
6. Key phrases    : top TF-IDF terms of an item, and the terms that
                    contribute most to the similarity of two vectors, mapped
                    back to readable words (used to explain a match).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

STOP_WORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how i
if in into is it its itself just me more most my no nor not now of off on once only or other our out over own same
she should so some such than that the their them then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your yours
""".split())

# words that appear in almost every README and carry no discriminative signal
DOMAIN_STOP_WORDS = frozenset("""
library libraries package packages provides provide offers offer allows allow use used using build builds built
make makes thousands tools tool simple yet widely many single
""".split())

ALL_STOP_WORDS = STOP_WORDS | DOMAIN_STOP_WORDS

SEGMENT_SPLIT = re.compile(r"[.;|,:()!?]")


# --------------------------------------------------------------------------- #
# Basic text processing
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Lowercase, turn separators into spaces, strip punctuation, squeeze spaces."""
    text = str(text).lower()
    text = re.sub(r"[-_/\\]", " ", text)          # web-scraping -> web scraping
    text = re.sub(r"[^a-z0-9\s]", " ", text)      # drop remaining punctuation
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Whitespace tokenization of already-cleaned text."""
    return text.split()


def remove_stopwords(tokens: Iterable[str]) -> list[str]:
    return [t for t in tokens if t not in ALL_STOP_WORDS and len(t) > 1]


def light_stem(token: str) -> str:
    """Very small suffix stripper (consistent for documents and queries)."""
    t = token
    if len(t) <= 3 or t.isdigit():
        return t
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("sses"):
        t = t[:-2]
    elif t.endswith("ing") and len(t) > 5:
        t = t[:-3]
    elif t.endswith("ed") and len(t) > 4:
        t = t[:-2]
    elif t.endswith("er") and len(t) > 4:
        t = t[:-2]
    elif t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        t = t[:-1]
    if t.endswith("e") and len(t) > 3:
        t = t[:-1]
    return t


def preprocess(text: str, stem: bool = False) -> list[str]:
    """clean -> tokenize -> stop-word removal (-> optional stemming)."""
    tokens = remove_stopwords(tokenize(clean_text(text)))
    return [light_stem(t) for t in tokens] if stem else tokens


def segments(text: str) -> list[list[str]]:
    """Split a raw document into segments (sentences / tags) and preprocess each."""
    out = []
    for seg in SEGMENT_SPLIT.split(str(text)):
        toks = preprocess(seg)
        if toks:
            out.append(toks)
    return out


# --------------------------------------------------------------------------- #
# Item documents
# --------------------------------------------------------------------------- #
def build_item_doc(item) -> str:
    """Text used to represent a library: README summary + category + language +
    topic tags (tags are repeated so that curated tags weigh more than prose)."""
    tags = str(item["topic_tags"])
    return (f"{item['readme_summary']} | {item['category']} | {item['language']} | "
            f"{tags} | {tags}")


def build_item_docs(items: pd.DataFrame) -> list[str]:
    return [build_item_doc(row) for _, row in items.iterrows()]


# --------------------------------------------------------------------------- #
# TF-IDF pipeline
# --------------------------------------------------------------------------- #
class NLPPipeline:
    """Fits TF-IDF (unigrams + bigrams) on the item corpus and re-uses the same
    vocabulary for new queries / new items (cold start)."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 2)):
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(analyzer=self._analyze, sublinear_tf=True, norm="l2")
        self._surface: dict[str, Counter] = defaultdict(Counter)
        self.feature_names: np.ndarray = np.array([])

    # -- analyzer ---------------------------------------------------------- #
    def _analyze(self, doc: str) -> list[str]:
        feats: list[str] = []
        lo, hi = self.ngram_range
        for seg in segments(doc):
            stems = [light_stem(t) for t in seg]
            for n in range(lo, hi + 1):
                feats += [" ".join(stems[i:i + n]) for i in range(len(stems) - n + 1)]
        return feats

    # -- fitting / transforming ------------------------------------------- #
    def fit(self, docs: Sequence[str]) -> "NLPPipeline":
        for doc in docs:                           # remember readable surface form of each stem
            for seg in segments(doc):
                for tok in seg:
                    self._surface[light_stem(tok)][tok] += 1
        self.vectorizer.fit(docs)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        return self

    def transform(self, docs: Sequence[str]) -> sparse.csr_matrix:
        return self.vectorizer.transform(docs)

    def fit_transform(self, docs: Sequence[str]) -> sparse.csr_matrix:
        self.fit(docs)
        return self.transform(docs)

    def vectorize_query(self, query: str) -> sparse.csr_matrix:
        """Free-text query (or a new item description) -> TF-IDF vector in the
        SAME vocabulary that was fitted on the dataset."""
        return self.vectorizer.transform([query])

    # -- explanations ------------------------------------------------------ #
    def readable(self, feature: str) -> str:
        """'web scrap' -> 'web scraping' (most frequent surface form of each stem)."""
        words = []
        for stem in feature.split(" "):
            counts = self._surface.get(stem)
            words.append(counts.most_common(1)[0][0] if counts else stem)
        return " ".join(words)

    def _top_features(self, weights: np.ndarray, n: int) -> list[tuple[str, float]]:
        order = np.argsort(-weights)
        chosen: list[tuple[str, float]] = []
        for idx in order:
            if weights[idx] <= 0 or len(chosen) >= n:
                break
            feat = self.feature_names[idx]
            # skip a unigram already covered by a chosen bigram (and vice versa)
            parts = set(feat.split(" "))
            redundant = any(parts <= set(c.split(" ")) or set(c.split(" ")) <= parts
                            for c, _ in chosen)
            if redundant:
                continue
            chosen.append((feat, float(weights[idx])))
        return [(self.readable(f), w) for f, w in chosen]

    def shared_keywords(self, vec_a, vec_b, n: int = 5) -> list[str]:
        """Terms that contribute most to cos(vec_a, vec_b) -> "why it matches"."""
        a = vec_a.toarray().ravel() if sparse.issparse(vec_a) else np.asarray(vec_a).ravel()
        b = vec_b.toarray().ravel() if sparse.issparse(vec_b) else np.asarray(vec_b).ravel()
        return [k for k, _ in self._top_features(a * b, n)]

    def key_phrases(self, vec, n: int = 5) -> list[str]:
        """Top TF-IDF capability phrases of one item (bigrams are preferred)."""
        w = vec.toarray().ravel() if sparse.issparse(vec) else np.asarray(vec).ravel()
        boost = np.array([1.25 if " " in f else 1.0 for f in self.feature_names])
        return [k for k, _ in self._top_features(w * boost, n)]


if __name__ == "__main__":  # tiny smoke test
    demo = "A lightweight python library for web scraping and html parsing!"
    print("clean   :", clean_text(demo))
    print("tokens  :", preprocess(demo))
    print("stemmed :", preprocess(demo, stem=True))
