"""Generate data/items.csv and data/ratings.csv (deterministic, seed = 467).

Items: 40 well-known open-source libraries. The README summaries are short,
original paraphrases of each project's public description (see README.md for
sources). Ratings: 14 synthetic developer personas, each with a taste profile
(preferred categories / languages), so that the interaction matrix is
realistic: sparse (about 50% missing), popularity-biased and noisy.

Usage:  python scripts/make_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# (package_name, language, category, readme_summary, topic_tags)
ITEMS = [
    ("beautifulsoup4", "Python", "Web Scraping",
     "A lightweight python library for pulling data out of HTML and XML files. It parses messy markup and offers simple methods for navigating, searching and modifying the parse tree.",
     "html-parsing;web-scraping;xml;parser"),
    ("scrapy", "Python", "Web Scraping",
     "A fast high-level web crawling and scraping framework used to crawl websites and extract structured data. Supports spiders, item pipelines, middleware and asynchronous request scheduling.",
     "web-crawling;web-scraping;spiders;framework;async"),
    ("lxml", "Python", "Web Scraping",
     "Fast and feature-rich python binding for the libxml2 and libxslt libraries. Provides XPath, XSLT, quick html parsing and xml processing for large documents.",
     "xml;html-parsing;xpath;parser"),
    ("selectolax", "Python", "Web Scraping",
     "A lightweight and very fast html5 parser with css selectors built on the Modest and Lexbor engines. A speedy alternative for scraping pages and extracting text from markup.",
     "html-parsing;css-selectors;fast;web-scraping"),
    ("playwright", "Python", "Browser Automation",
     "Python library to automate Chromium, Firefox and WebKit browsers with a single API. Handles dynamic javascript pages, screenshots, headless runs and end-to-end browser testing.",
     "browser-automation;headless;web-scraping;testing"),
    ("requests", "Python", "HTTP Client",
     "A simple and elegant HTTP library for python for sending requests to web services. Handles sessions, cookies, headers, JSON bodies and file uploads with a human friendly API.",
     "http;api-client;requests;web"),
    ("httpx", "Python", "HTTP Client",
     "A next generation HTTP client for python with async support, HTTP/2 and a requests-compatible API. Useful for calling REST services and downloading content concurrently.",
     "http;async;http2;api-client"),
    ("aiohttp", "Python", "HTTP Client",
     "Asynchronous HTTP client and server framework built on asyncio. Supports websockets, streaming responses and many concurrent requests in a single event loop.",
     "async;asyncio;http;websockets"),
    ("cheerio", "JavaScript", "Web Scraping",
     "Fast, flexible and lean implementation of core jQuery designed for the server. Parse markup and traverse the DOM with familiar selectors for web scraping in node.",
     "html-parsing;web-scraping;dom;jquery;nodejs"),
    ("colly", "Go", "Web Scraping",
     "Lightning fast and elegant scraping framework for Go. Offers a clean interface to write crawlers, spiders and scrapers with parallel requests, caching and rate limiting.",
     "web-scraping;crawler;golang;fast"),
    ("matplotlib", "Python", "Data Visualization",
     "Comprehensive library for creating static, animated and interactive plots in python. Produces publication quality line charts, histograms, scatter plots and bar charts.",
     "plotting;charts;visualization;figures"),
    ("seaborn", "Python", "Data Visualization",
     "Statistical data visualization library built on top of matplotlib with a high-level interface for drawing attractive heatmaps, distribution plots and regression charts from pandas dataframes.",
     "statistical-plots;visualization;heatmap;pandas"),
    ("plotly", "Python", "Data Visualization",
     "Interactive browser based graphing library offering over forty chart types including 3D plots, maps and dashboards. Charts can be embedded in web apps and notebooks.",
     "interactive-charts;visualization;dashboard;plotting"),
    ("bokeh", "Python", "Data Visualization",
     "Interactive visualization library that targets modern web browsers for presentation. Build streaming dashboards and plots with python and serve them to large datasets.",
     "interactive-charts;visualization;dashboard;streaming"),
    ("altair", "Python", "Data Visualization",
     "Declarative statistical visualization library based on Vega-Lite. Describe charts with a concise grammar of graphics instead of low-level plotting commands.",
     "declarative;visualization;vega-lite;charts"),
    ("d3", "JavaScript", "Data Visualization",
     "Low-level JavaScript library for bespoke data visualization using SVG, canvas and HTML. Bind data to the DOM and build custom interactive charts.",
     "visualization;svg;charts;dom;interactive-charts"),
    ("chart.js", "JavaScript", "Data Visualization",
     "Simple yet flexible JavaScript charting library for the HTML5 canvas element. Draws responsive bar, line and pie charts with smooth animations.",
     "charts;canvas;visualization;responsive"),
    ("numpy", "Python", "Data Science",
     "Fundamental package for scientific computing with python, providing fast n-dimensional arrays, linear algebra, random numbers and broadcasting.",
     "arrays;linear-algebra;numerical-computing;scientific"),
    ("pandas", "Python", "Data Science",
     "Fast and flexible data analysis and manipulation library offering dataframes for cleaning, merging, grouping and reshaping tabular data from csv files and sql.",
     "dataframe;data-analysis;tabular-data;csv"),
    ("polars", "Rust", "Data Science",
     "Blazingly fast dataframe library written in Rust with lazy query optimisation and multi-threaded execution. A fast alternative to pandas for large tabular data.",
     "dataframe;fast;rust;tabular-data;data-analysis"),
    ("scipy", "Python", "Data Science",
     "Open-source library for scientific computing with algorithms for optimization, integration, interpolation, signal processing and statistics built on numpy arrays.",
     "scientific;optimization;statistics;numerical-computing"),
    ("scikit-learn", "Python", "Machine Learning",
     "Simple and efficient tools for predictive data analysis including classification, regression, clustering and model selection, built on numpy and scipy.",
     "machine-learning;classification;clustering;regression"),
    ("pytorch", "Python", "Machine Learning",
     "Deep learning framework with dynamic computation graphs, GPU acceleration and automatic differentiation for building and training neural networks.",
     "deep-learning;neural-networks;gpu;tensors"),
    ("tensorflow", "Python", "Machine Learning",
     "End-to-end open source platform for machine learning with tools for training neural networks and deploying models to production, mobile devices and browsers.",
     "deep-learning;neural-networks;deployment;gpu"),
    ("xgboost", "Python", "Machine Learning",
     "Optimized distributed gradient boosting library for fast and accurate tree based models on tabular data, widely used in competitions.",
     "gradient-boosting;machine-learning;tabular-data;classification"),
    ("nltk", "Python", "NLP",
     "Leading platform for building python programs that work with human language data. Provides tokenizers, stemmers, corpora and text classification tools for teaching and research.",
     "nlp;tokenization;text-processing;corpora"),
    ("spacy", "Python", "NLP",
     "Industrial strength natural language processing library with fast tokenization, named entity recognition, part of speech tagging and dependency parsing pipelines.",
     "nlp;named-entity-recognition;tokenization;pipeline"),
    ("transformers", "Python", "NLP",
     "Thousands of pretrained transformer models for text, vision and audio tasks. Fine tune BERT or GPT style models for text classification, translation and summarization.",
     "nlp;transformers;pretrained-models;text-classification"),
    ("gensim", "Python", "NLP",
     "Library for topic modelling and document similarity using word2vec, LDA and TF-IDF that streams large text corpora efficiently.",
     "topic-modeling;nlp;word-embeddings;text-similarity"),
    ("flask", "Python", "Web Framework",
     "Lightweight WSGI micro web framework to build web applications and REST APIs quickly with routing, templates and extensions.",
     "web-framework;microframework;rest-api;lightweight"),
    ("django", "Python", "Web Framework",
     "High level batteries included web framework that encourages rapid development with an ORM, admin interface, authentication and templating.",
     "web-framework;orm;admin;full-stack"),
    ("fastapi", "Python", "Web Framework",
     "Modern high performance web framework for building APIs with python type hints, automatic OpenAPI documentation, async support and data validation.",
     "web-framework;rest-api;async;openapi;validation"),
    ("express", "JavaScript", "Web Framework",
     "Fast unopinionated minimalist web framework for node providing routing, middleware and HTTP helpers to build APIs and web apps.",
     "web-framework;nodejs;middleware;rest-api"),
    ("gin", "Go", "Web Framework",
     "High performance HTTP web framework for Go with a martini-like API, fast routing, middleware and JSON validation.",
     "web-framework;golang;fast;rest-api"),
    ("pytest", "Python", "Testing",
     "Mature testing framework that makes it easy to write small readable tests and scales to complex functional testing with fixtures and plugins.",
     "testing;fixtures;unit-tests;plugins"),
    ("jest", "JavaScript", "Testing",
     "Delightful JavaScript testing framework with a focus on simplicity, offering zero config setup, mocking, snapshot tests and code coverage.",
     "testing;mocking;snapshot;unit-tests"),
    ("click", "Python", "CLI Tools",
     "Composable library for creating beautiful command line interfaces with decorators, argument parsing, prompts and automatic help pages.",
     "cli;argument-parsing;command-line"),
    ("rich", "Python", "CLI Tools",
     "Library for rich text and beautiful formatting in the terminal with colors, tables, progress bars, syntax highlighting and markdown rendering.",
     "terminal;formatting;cli;progress-bar"),
    ("sqlalchemy", "Python", "Database",
     "Python SQL toolkit and object relational mapper giving application developers the full power of SQL with a flexible ORM and connection pooling.",
     "orm;sql;database;connection-pooling"),
    ("pydantic", "Python", "Data Validation",
     "Data validation and settings management using python type annotations, parsing input into typed models and producing clear error messages.",
     "validation;type-hints;data-parsing;settings"),
]

# Developer personas: category weights, language weights, extra favourite packages.
USERS = {
    "U01": dict(cats={"Data Science": 1, "Machine Learning": .9, "Data Visualization": .8, "NLP": .3},
                lang={"Python": 1.0}, fav=["pandas", "scikit-learn", "matplotlib"]),
    "U02": dict(cats={"Web Framework": 1, "Database": .9, "Data Validation": .8, "HTTP Client": .6, "Testing": .6},
                lang={"Python": 1.0}, fav=["django", "sqlalchemy", "pytest"]),
    "U03": dict(cats={"Web Scraping": 1, "Browser Automation": .9, "HTTP Client": .7},
                lang={"Python": 1.0, "JavaScript": .5, "Go": .4}, fav=["beautifulsoup4", "scrapy", "requests"]),
    "U04": dict(cats={"NLP": 1, "Machine Learning": .7, "Data Science": .5},
                lang={"Python": 1.0}, fav=["spacy", "nltk", "transformers"]),
    "U05": dict(cats={"Data Visualization": 1, "Web Framework": .6, "Testing": .5},
                lang={"JavaScript": 1.0, "Python": .3}, fav=["d3", "chart.js", "jest"]),
    "U06": dict(cats={"Machine Learning": 1, "Data Science": .6, "NLP": .5},
                lang={"Python": 1.0}, fav=["pytorch", "tensorflow", "transformers"]),
    "U07": dict(cats={"Web Framework": .8, "Web Scraping": .6, "Data Science": .6, "HTTP Client": .4},
                lang={"Go": 1.0, "Rust": 1.0, "Python": .25}, fav=["gin", "polars", "colly"]),
    "U08": dict(cats={"Data Visualization": 1, "Data Science": .8},
                lang={"Python": 1.0, "JavaScript": .5}, fav=["seaborn", "plotly", "altair", "pandas"]),
    "U09": dict(cats={"Web Framework": 1, "Testing": .7, "Data Visualization": .5, "Web Scraping": .4},
                lang={"JavaScript": 1.0, "Python": .3}, fav=["express", "jest", "cheerio"]),
    "U10": dict(cats={"CLI Tools": 1, "Testing": .6, "HTTP Client": .5, "Data Validation": .5},
                lang={"Python": 1.0}, fav=["click", "rich", "pytest"]),
    "U11": dict(cats={}, lang={}, fav=["requests", "numpy", "matplotlib", "flask"]),  # generalist student
    "U12": dict(cats={"Data Science": 1, "Database": .9, "Data Validation": .5, "Data Visualization": .4},
                lang={"Python": 1.0, "Rust": .7}, fav=["pandas", "polars", "sqlalchemy", "scipy"]),
    "U13": dict(cats={"Web Framework": .9, "HTTP Client": 1, "Data Validation": .9, "Testing": .5},
                lang={"Python": 1.0}, fav=["fastapi", "httpx", "pydantic", "aiohttp"]),
    "U14": dict(cats={"Testing": 1, "Browser Automation": .9, "HTTP Client": .6, "CLI Tools": .4},
                lang={"Python": 1.0, "JavaScript": .6}, fav=["pytest", "playwright", "jest", "requests"]),
}

POPULAR = {"requests", "numpy", "pandas", "matplotlib", "pytest", "flask", "scikit-learn"}

POS = ["Solid choice for {t1}; the docs are great.",
       "Use it daily for {t1} and {t2}. Highly recommend.",
       "{pkg} just works. Clean API and good {t1} support.",
       "Fast to set up and reliable for {t1}.",
       "Love the community and the {t2} features.",
       "Became a core part of my stack for {t1}.",
       "Great {t1} tooling, saved me a lot of time on {t2}."]
MID = ["Does the job for {t1} but the learning curve is real.",
       "Okay for {t1}, though I prefer other tools for {t2}.",
       "Decent, a bit heavy for small scripts.",
       "Works fine; documentation could be clearer."]
LOW = ["Too complex for what I need; {t1} feels clunky.",
       "Hard to configure and the docs confused me.",
       "Not a good fit for {t1}; found simpler alternatives.",
       "Slow in my tests and the API changes too often."]


def main() -> None:
    rng = np.random.default_rng(467)
    items = pd.DataFrame(ITEMS, columns=["package_name", "language", "category",
                                         "readme_summary", "topic_tags"])
    items.insert(0, "item_id", [f"L{i + 1:02d}" for i in range(len(items))])
    items.to_csv(DATA / "items.csv", index=False)

    rows = []
    for uid, prof in USERS.items():
        aff = np.zeros(len(items))
        for i, it in items.iterrows():
            cat_w = prof["cats"].get(it.category, 0.15) if prof["cats"] else 0.5
            lang_w = prof["lang"].get(it.language, 0.3) if prof["lang"] else 0.7
            aff[i] = cat_w * lang_w
            if it.package_name in prof["fav"]:
                aff[i] = max(aff[i], 0.9)
        pop = np.array([p in POPULAR for p in items.package_name], dtype=float)
        p = 0.08 + aff ** 1.2 + 0.25 * pop
        p = p / p.sum()
        n_rated = int(rng.integers(16, 25))  # 16-24 of 40 => 40%-60% missing
        chosen = rng.choice(len(items), size=n_rated, replace=False, p=p)
        for i in sorted(chosen):
            mean = 1.9 + 3.0 * aff[i] + 0.35 * pop[i]
            r = int(np.clip(round(rng.normal(mean, 0.55)), 1, 5))
            tags = [t.replace("-", " ") for t in items.topic_tags[i].split(";")]
            pool = POS if r >= 4 else MID if r == 3 else LOW
            comment = pool[int(rng.integers(len(pool)))].format(
                pkg=items.package_name[i], t1=tags[0], t2=tags[min(1, len(tags) - 1)])
            rows.append((uid, items.item_id[i], r, comment))
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "dev_comment"])
    ratings.to_csv(DATA / "ratings.csv", index=False)

    miss = 1 - len(ratings) / (ratings.user_id.nunique() * len(items))
    print(f"items={len(items)} users={ratings.user_id.nunique()} ratings={len(ratings)} missing={miss:.1%}")
    per_user = ratings.groupby("user_id").size()
    print(f"ratings per user: min={per_user.min()} max={per_user.max()}")


if __name__ == "__main__":
    main()
