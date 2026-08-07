"""
LDA Topic Modeling of Job Market News
======================================
Pulls a fresh sample from NewsAPI and applies Latent Dirichlet Allocation
(probabilistic topic modeling) — a technique not used elsewhere in this repo,
which so far only has hard KMeans clustering (kmeans_clustering.py).

Requires NEWS_API_KEY to be set as an environment variable.
"""
import os
import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

API_KEY = os.environ.get("NEWS_API_KEY")
if not API_KEY:
    raise SystemExit("Set the NEWS_API_KEY environment variable before running this script.")

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
df["title"] = df["title"].fillna("")
df["description"] = df["description"].fillna("")
df["text"] = (df["title"] + ". " + df["description"]).str.strip()
print(f"Fetched {len(df)} fresh articles")

def clean_text(text):
    text = str(text).lower()
    return re.sub(r"[^a-z\s]", "", text)

df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].str.len() > 5].reset_index(drop=True)
print(f"Modeling topics on {len(df)} articles")

# ============================================================
# LDA TOPIC MODELING
# ============================================================
vectorizer = CountVectorizer(max_features=150, stop_words="english", ngram_range=(1, 2))
doc_term = vectorizer.fit_transform(df["clean_text"])

n_topics = 6
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=25)
doc_topic = lda.fit_transform(doc_term)
df["topic"] = doc_topic.argmax(axis=1)
df["topic_confidence"] = doc_topic.max(axis=1)

feature_names = vectorizer.get_feature_names_out()
topic_top_words = {}
for t in range(n_topics):
    top_idx = lda.components_[t].argsort()[-8:][::-1]
    topic_top_words[t] = [feature_names[i] for i in top_idx]

topic_labels = {t: " / ".join(words[:3]) for t, words in topic_top_words.items()}
df["topic_label"] = df["topic"].map(topic_labels)

summary = pd.DataFrame([
    {
        "topic": t,
        "label": topic_labels[t],
        "top_words": ", ".join(words),
        "article_count": int((df["topic"] == t).sum()),
        "avg_confidence": round(float(df.loc[df["topic"] == t, "topic_confidence"].mean()), 3),
    }
    for t, words in topic_top_words.items()
]).sort_values("article_count", ascending=False)

summary.to_csv("lda_topic_summary.csv", index=False)
print("\nTopic summary:")
print(summary.to_string(index=False))

# ============================================================
# CHART: Topic distribution + top words per topic
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("LDA Topic Modeling — Job Market News (fresh NewsAPI pull)", fontsize=16, fontweight="bold")

colors = plt.cm.tab10(np.linspace(0, 1, n_topics))

# Left: article count per topic
order = summary["topic"].tolist()
counts = summary["article_count"].tolist()
labels = summary["label"].tolist()
bars = axes[0].barh(labels, counts, color=[colors[t] for t in order], edgecolor="black")
axes[0].set_title("Articles per Topic", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Number of Articles")
axes[0].invert_yaxis()
for bar, val in zip(bars, counts):
    axes[0].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, str(val),
                 ha="left", va="center", fontsize=9)

# Right: top word weights for the single largest topic
top_topic = order[0]
top_topic_words = topic_top_words[top_topic]
weights = lda.components_[top_topic]
word_idx = [np.where(feature_names == w)[0][0] for w in top_topic_words]
word_weights = [weights[i] for i in word_idx]
axes[1].barh(top_topic_words, word_weights, color=colors[top_topic], edgecolor="black")
axes[1].invert_yaxis()
axes[1].set_title(f"Top Words in Largest Topic ({topic_labels[top_topic]})", fontsize=13, fontweight="bold")
axes[1].set_xlabel("LDA Term Weight")

plt.tight_layout()
plt.savefig("lda_topic_modeling.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved lda_topic_modeling.png")
print("Saved lda_topic_summary.csv")
