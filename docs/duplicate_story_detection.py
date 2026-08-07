"""
Near-Duplicate / Syndicated Story Detection
=============================================
Pulls a fresh sample from NewsAPI and uses TF-IDF + cosine similarity to
find articles that are really the same wire story republished by multiple
outlets. Different technique from every other analysis in this repo
(correlation stats, KMeans clustering, Naive Bayes sentiment, LDA topics) —
none of them check whether "distinct articles" are actually duplicates.

Requires NEWS_API_KEY to be set as an environment variable.
"""
import os
import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = os.environ.get("NEWS_API_KEY")
if not API_KEY:
    raise SystemExit("Set the NEWS_API_KEY environment variable before running this script.")

DUPLICATE_THRESHOLD = 0.55

# ============================================================
# FETCH FRESH DATA
# ============================================================
url = "https://newsapi.org/v2/everything"
params = {
    "q": "job market hiring employment",
    "language": "en",
    "sortBy": "publishedAt",
    "pageSize": 100,
    "apiKey": API_KEY,
}
resp = requests.get(url, params=params)
payload = resp.json()
if payload.get("status") != "ok":
    raise RuntimeError(f"NewsAPI error: {payload}")

df = pd.DataFrame(payload["articles"])
df["source_name"] = df["source"].apply(lambda x: x.get("name", "Unknown") if isinstance(x, dict) else "Unknown")
df["title"] = df["title"].fillna("")
df["description"] = df["description"].fillna("")
df["text"] = (df["title"] + ". " + df["description"]).str.strip()
print(f"Fetched {len(df)} fresh articles")

def clean_text(text):
    text = str(text).lower()
    return re.sub(r"[^a-z\s]", "", text)

df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].str.len() > 5].reset_index(drop=True)
print(f"Comparing {len(df)} articles for near-duplicates")

# ============================================================
# TF-IDF COSINE SIMILARITY
# ============================================================
vectorizer = TfidfVectorizer(max_features=300, stop_words="english")
X = vectorizer.fit_transform(df["clean_text"])
sim_matrix = cosine_similarity(X)

n = len(df)
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        sim = sim_matrix[i, j]
        if sim > DUPLICATE_THRESHOLD:
            pairs.append({
                "title_a": df.loc[i, "title"],
                "source_a": df.loc[i, "source_name"],
                "title_b": df.loc[j, "title"],
                "source_b": df.loc[j, "source_name"],
                "similarity": round(float(sim), 3),
            })

dup_df = pd.DataFrame(pairs).sort_values("similarity", ascending=False)
dup_df.to_csv("duplicate_story_summary.csv", index=False)

dup_article_count = len(set(
    [p["title_a"] for p in pairs] + [p["title_b"] for p in pairs]
))
dup_rate = dup_article_count / n * 100

print(f"\nFound {len(dup_df)} near-duplicate pairs above similarity {DUPLICATE_THRESHOLD}")
print(f"{dup_article_count} of {n} articles ({dup_rate:.1f}%) are involved in at least one near-duplicate pair")
if len(dup_df):
    print("\nTop duplicate pairs:")
    print(dup_df.head(8).to_string(index=False))

# ============================================================
# CHART: Similarity distribution + top duplicate pairs
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Near-Duplicate / Syndicated Story Detection (fresh NewsAPI pull)", fontsize=16, fontweight="bold")

# Left: similarity distribution
upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
axes[0].hist(upper_tri, bins=30, color="mediumpurple", edgecolor="black", alpha=0.8)
axes[0].axvline(x=DUPLICATE_THRESHOLD, color="red", linestyle="--", linewidth=2,
                label=f"Duplicate threshold ({DUPLICATE_THRESHOLD})")
axes[0].set_title(f"Article-Pair Similarity Distribution\n({len(dup_df)} likely duplicates found)",
                   fontsize=12, fontweight="bold")
axes[0].set_xlabel("Cosine Similarity")
axes[0].set_ylabel("Pair Count")
axes[0].legend(fontsize=9)

# Right: top duplicate pairs by similarity
top_pairs = dup_df.head(8).iloc[::-1]
pair_labels = [f"{a[:30]}...\nvs {b[:30]}..." if len(a) > 30 or len(b) > 30 else f"{a}\nvs {b}"
               for a, b in zip(top_pairs["source_a"], top_pairs["source_b"])]
bars = axes[1].barh(range(len(top_pairs)), top_pairs["similarity"], color="coral", edgecolor="black")
axes[1].set_yticks(range(len(top_pairs)))
axes[1].set_yticklabels(pair_labels, fontsize=8)
axes[1].set_xlim(0, 1)
axes[1].set_title("Top Duplicate Pairs (by source outlets)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Cosine Similarity")
for bar, val in zip(bars, top_pairs["similarity"]):
    axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}",
                 ha="left", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("duplicate_story_detection.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved duplicate_story_detection.png")
print("Saved duplicate_story_summary.csv")
