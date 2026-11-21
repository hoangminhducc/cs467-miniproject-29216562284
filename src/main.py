"""Runnable demo / CLI for the Open-Source Library Recommender (MP-C02).

Examples
--------
python src/main.py                              # run the 3 required test cases
python src/main.py --save                       # ... and write results/sample_outputs.txt
python src/main.py --user U03                   # Top-5 for a developer (all 3 methods)
python src/main.py --query "async http client for python"
python src/main.py --similar seaborn            # 5 alternatives to a library (name or id)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from recommender import HybridRecommender, load_data

ROOT = Path(__file__).resolve().parent.parent
TEST_QUERY = "lightweight python library for web scraping and html parsing"

NEW_ITEM = {  # brand-new library with ZERO developer ratings (cold-start test)
    "item_id": "L41", "package_name": "pagepick", "language": "Python", "category": "Web Scraping",
    "readme_summary": ("A tiny zero-dependency python library for quickly scraping web pages. It parses html, "
                       "supports css selectors and extracts text, links and tables from markup."),
    "topic_tags": "html-parsing;web-scraping;css-selectors;lightweight",
}


def fmt(recs: list[dict], show_components: bool = False) -> str:
    lines = []
    for n, r in enumerate(recs, 1):
        kw = ", ".join(r["keywords"][:4]) if r["keywords"] else "-"
        line = f"  {n}. {r['package_name']:<15} [{r['language']} | {r['category']}]  score={r['score']:.3f}"
        if r["keywords"]:
            line += f"\n       matched on: {kw}"
        if show_components and r.get("components"):
            comp = ", ".join(f"{k}={'n/a' if v is None else f'{v:.2f}'}" for k, v in r["components"].items())
            line += f"\n       hybrid components: {comp}"
        lines.append(line)
    return "\n".join(lines) if lines else "  (no recommendation)"


def resolve_item(items, ref: str) -> str:
    m = items[(items.item_id.str.lower() == ref.lower()) | (items.package_name.str.lower() == ref.lower())]
    if m.empty:
        raise SystemExit(f"Unknown library '{ref}'. Examples: {', '.join(items.package_name.head(8))}")
    return m.iloc[0]["item_id"]


def name_of(items, item_id: str) -> str:
    return items.loc[items.item_id == item_id, "package_name"].iloc[0]


# --------------------------------------------------------------------------- #
def test_case_1(model) -> str:
    out = [f'TEST CASE 1 - Need query: "{TEST_QUERY}"', "",
           "NLP content-based (TF-IDF cosine):", fmt(model.content.recommend_query(TEST_QUERY)), "",
           "Hybrid (content + small popularity prior):", fmt(model.recommend_query(TEST_QUERY))]
    return "\n".join(out)


def test_case_2(model, ref: str = "matplotlib") -> str:
    iid = resolve_item(model.items, ref)
    out = [f"TEST CASE 2 - Similar-package test: 5 alternatives to '{name_of(model.items, iid)}'", "",
           "NLP content-based (item-to-item cosine):", fmt(model.content.similar_items(iid)), "",
           "Hybrid (content similarity + co-rating similarity):", fmt(model.similar_items(iid))]
    return "\n".join(out)


def test_case_3(items, ratings) -> str:
    model = HybridRecommender(items).fit(ratings)      # fresh model -> catalogue is not polluted
    n_before = len(model.items)
    model.add_item(dict(NEW_ITEM))
    iid = NEW_ITEM["item_id"]
    out = [f"TEST CASE 3 - Cold start: new library '{NEW_ITEM['package_name']}' ({iid}) added with 0 ratings", "",
           f"  catalogue size: {n_before} -> {len(model.items)}",
           f"  ratings of {iid} in the rating matrix: {int((ratings.item_id == iid).sum())}",
           f"  key phrases (TF-IDF): {', '.join(model.nlp.key_phrases(model.content.X[model.content.id_to_idx[iid]], 5))}", ""]

    sim = model.content.similar_items(iid)
    out += ["(a) Most similar existing libraries (same vocabulary / vectorizer):", fmt(sim), ""]

    q = model.content.recommend_query(TEST_QUERY, k=10)
    rank = next((n for n, r in enumerate(q, 1) if r["item_id"] == iid), None)
    out += [f'(b) Need query "{TEST_QUERY}" -> Top-5 (content-based):',
            fmt(q[:5]), f"    rank of the new library in the content-based list: {rank}", ""]

    hq = model.recommend_query(TEST_QUERY, k=10)
    hrank = next((n for n, r in enumerate(hq, 1) if r["item_id"] == iid), None)
    out += [f"    rank of the new library in the hybrid list: {hrank}", ""]

    pop = model.pop.recommend(None, k=len(model.items))
    out += [f"(c) Popularity baseline can recommend it? {'yes' if any(r['item_id'] == iid for r in pop) else 'NO - needs >= 2 ratings'}", ""]

    user = "U03"  # developer who rated many scraping libraries
    ur = model.recommend_user(user, k=len(model.items))
    urank = next((n for n, r in enumerate(ur, 1) if r["item_id"] == iid), None)
    out += [f"(d) Hybrid Top-5 for developer {user} (scraping fan); '{NEW_ITEM['package_name']}' ranked #{urank} of {len(ur)} unseen libraries:",
            fmt(ur[:5], show_components=True)]

    verdict = (sim and sim[0]["category"] == NEW_ITEM["category"]) and rank is not None and rank <= 5
    out += ["", f"VERIFICATION: nearest neighbours are scraping/parsing libraries and the new item is in the "
                f"Top-5 of the scraping query -> {'PASS' if verdict else 'FAIL'}"]
    return "\n".join(out)


def user_report(model, items, user: str) -> str:
    if user not in set(model.content.ratings["user_id"]):
        raise SystemExit(f"Unknown user '{user}'. Valid ids: {', '.join(sorted(set(model.content.ratings.user_id)))}")
    rated = model.content.rated_by(user)
    liked = [name_of(items, i) for i, r in sorted(rated.items(), key=lambda x: -x[1]) if r >= 4][:6]
    return "\n".join([f"Developer {user} - rated {len(rated)} libraries; top liked: {', '.join(liked)}", "",
                      "Popularity baseline:", fmt(model.pop.recommend(user)), "",
                      "NLP content-based (user profile):", fmt(model.content.recommend_user(user)), "",
                      "Hybrid (NLP + item-CF + popularity):", fmt(model.recommend_user(user), True)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Open-source library recommender demo")
    ap.add_argument("--user", help="developer id, e.g. U03")
    ap.add_argument("--query", help="free-text need, e.g. 'async http client'")
    ap.add_argument("--similar", help="library name or id to find alternatives for")
    ap.add_argument("--save", action="store_true", help="write results/sample_outputs.txt")
    args = ap.parse_args()

    items, ratings = load_data(ROOT / "data")
    model = HybridRecommender(items).fit(ratings)
    custom = args.user or args.query or args.similar
    blocks = []
    if args.user:
        blocks.append(user_report(model, items, args.user))
    if args.query:
        blocks.append(f'Query: "{args.query}"\n\nNLP content-based:\n{fmt(model.content.recommend_query(args.query))}'
                      f'\n\nHybrid:\n{fmt(model.recommend_query(args.query))}')
    if args.similar:
        blocks.append(test_case_2(model, args.similar))
    if not custom:
        blocks = [test_case_1(model), test_case_2(model), test_case_3(items, ratings)]
    text = ("\n\n" + "=" * 78 + "\n\n").join(blocks)
    print(text)
    if args.save:
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / "sample_outputs.txt").write_text(text + "\n", encoding="utf-8")
        print("\nSaved to results/sample_outputs.txt")


if __name__ == "__main__":
    main()
