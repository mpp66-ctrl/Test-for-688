"""
Cross-Category Headline Analysis
===================================
Every other script in this repo hits /v2/everything with the same
"job market hiring employment" query. This script uses a different
NewsAPI endpoint entirely -- /v2/top-headlines -- to pull current top
headlines across six distinct categories (business, technology, health,
science, entertainment, sports) and compares tone, headline length, and
vocabulary across them. No script here has looked outside the
job-market query before.

Requires NEWS_API_KEY to be set as an environment variable.
"""
import os
import re
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from collections import Counter

API_KEY = os.environ.get("NEWS_API_KEY")
if not API_KEY:
    raise SystemExit("Set the NEWS_API_KEY environment variable before running this script.")

CATEGORIES = ["business", "technology", "health", "science", "entertainment", "sports"]

# ============================================================
# FETCH TOP HEADLINES PER CATEGORY
# ============================================================
url = "https://newsapi.org/v2/top-headlines"
rows = []
for cat in CATEGORIES:
    params = {"category": cat, "language": "en", "pageSize": 20, "apiKey": API_KEY}
    resp = requests.get(url, params=params)
    payload = resp.json()
    if payload.get("status") != "ok":
        print(f"Skipping {cat}: {payload}")
        continue
    for a in payload.get("articles", []):
        rows.append({
            "category": cat,
            "title": a.get("title") or "",
            "description": a.get("description") or "",
            "source": (a.get("source") or {}).get("name", "Unknown"),
        })
    print(f"{cat}: {len(payload.get('articles', []))} headlines")
    time.sleep(0.3)  # be polite to the API

df = pd.DataFrame(rows)
df = df[df["title"].str.len() > 0].reset_index(drop=True)
print(f"\nTotal headlines collected: {len(df)}")

# ============================================================
# SENTIMENT + LENGTH FEATURES
# ============================================================
df["sentiment"] = df["title"].apply(lambda t: TextBlob(str(t)).sentiment.polarity)
df["subjectivity"] = df["title"].apply(lambda t: TextBlob(str(t)).sentiment.subjectivity)
df["headline_length"] = df["title"].str.len()
df["word_count"] = df["title"].str.split().str.len()

def clean_text(text):
    text = str(text).lower()
    return re.sub(r"[^a-z\s]", "", text)

stopwords = {"the", "a", "an", "in", "of", "to", "and", "for", "on", "is", "are",
             "with", "at", "by", "from", "as", "its", "that", "this", "it", "was",
             "be", "have", "has", "will", "not", "but", "or", "about", "more", "new"}

def top_words(series, n=5):
    words = " ".join(series.apply(clean_text)).split()
    filtered = [w for w in words if w not in stopwords and len(w) > 3]
    return [w for w, _ in Counter(filtered).most_common(n)]

summary = df.groupby("category").agg(
    articles=("title", "count"),
    avg_sentiment=("sentiment", "mean"),
    avg_subjectivity=("subjectivity", "mean"),
    avg_headline_length=("headline_length", "mean"),
    avg_word_count=("word_count", "mean"),
).round(3)
summary["top_words"] = df.groupby("category")["title"].apply(lambda s: ", ".join(top_words(s)))
summary = summary.sort_values("avg_sentiment", ascending=False)
summary.to_csv("category_comparison_summary.csv")

print("\n=== Category Comparison Summary ===")
print(summary.to_string())

# ============================================================
# CHARTS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Cross-Category Headline Analysis (NewsAPI top-headlines)", fontsize=16, fontweight="bold")

colors = plt.cm.tab10(np.linspace(0, 1, len(summary)))

# Chart 1: Avg sentiment by category
bars = axes[0, 0].bar(summary.index, summary["avg_sentiment"], color=colors, edgecolor="black")
axes[0, 0].axhline(y=0, color="black", linewidth=1)
axes[0, 0].set_title("Avg Headline Sentiment by Category", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("Avg Sentiment (TextBlob polarity)")
axes[0, 0].tick_params(axis="x", rotation=30)
for bar, val in zip(bars, summary["avg_sentiment"]):
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (0.005 if val >= 0 else -0.015),
                     f"{val:.3f}", ha="center", fontsize=9)

# Chart 2: Avg subjectivity by category
bars2 = axes[0, 1].bar(summary.index, summary["avg_subjectivity"], color=colors, edgecolor="black")
axes[0, 1].set_title("Avg Headline Subjectivity by Category", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("Avg Subjectivity (0=factual, 1=opinion)")
axes[0, 1].tick_params(axis="x", rotation=30)
axes[0, 1].set_ylim(0, 1)
for bar, val in zip(bars2, summary["avg_subjectivity"]):
    axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", fontsize=9)

# Chart 3: Avg headline word count by category
bars3 = axes[1, 0].bar(summary.index, summary["avg_word_count"], color=colors, edgecolor="black")
axes[1, 0].set_title("Avg Headline Word Count by Category", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("Avg Words per Headline")
axes[1, 0].tick_params(axis="x", rotation=30)
for bar, val in zip(bars3, summary["avg_word_count"]):
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.1f}", ha="center", fontsize=9)

# Chart 4: Sentiment vs subjectivity scatter, colored by category
for i, cat in enumerate(summary.index):
    cat_df = df[df["category"] == cat]
    axes[1, 1].scatter(cat_df["sentiment"], cat_df["subjectivity"],
                        label=cat, color=colors[i], alpha=0.6, s=40)
axes[1, 1].axvline(x=0, color="gray", linestyle="--", linewidth=1)
axes[1, 1].set_title("Sentiment vs Subjectivity (per headline)", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("Sentiment")
axes[1, 1].set_ylabel("Subjectivity")
axes[1, 1].legend(fontsize=8, loc="upper right")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("category_comparison_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved category_comparison_analysis.png")
print("Saved category_comparison_summary.csv")
