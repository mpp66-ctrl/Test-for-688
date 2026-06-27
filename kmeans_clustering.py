import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from textblob import TextBlob
import re
from collections import Counter

# Load data
df = pd.read_csv("job_market_news.csv")
df['source_name'] = df['source'].apply(lambda x: eval(x)['name'] if isinstance(x, str) else 'Unknown')
df['publishedAt'] = pd.to_datetime(df['publishedAt'])

# ============================================================
# TEXT PREPROCESSING
# ============================================================
def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def get_sentiment(text):
    if pd.isna(text):
        return 0
    return TextBlob(str(text)).sentiment.polarity

df['clean_title'] = df['title'].apply(clean_text)
df['sentiment'] = df['title'].apply(get_sentiment)
df = df[df['clean_title'].str.len() > 5].copy()

print(f"Clustering {len(df)} articles...")

# ============================================================
# TFIDF + KMEANS CLUSTERING
# ============================================================
vectorizer = TfidfVectorizer(
    max_features=200,
    stop_words='english',
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(df['clean_title'])

# KMeans with 5 clusters
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

# Get top words per cluster
feature_names = vectorizer.get_feature_names_out()
cluster_labels = {}
cluster_top_words = {}

for i in range(k):
    center = kmeans.cluster_centers_[i]
    top_indices = center.argsort()[-8:][::-1]
    top_words = [feature_names[idx] for idx in top_indices]
    cluster_top_words[i] = top_words

# Manually label clusters based on top words
print("\nCluster Top Words:")
for i, words in cluster_top_words.items():
    print(f"Cluster {i}: {', '.join(words)}")

cluster_names = {
    0: "Hiring & Employment",
    1: "Market & Economy",
    2: "AI & Technology",
    3: "Global & Policy",
    4: "Salary & Growth"
}

df['cluster_name'] = df['cluster'].map(cluster_names)

# ============================================================
# SAVE TO CSV
# ============================================================
output_cols = ['title', 'source_name', 'publishedAt', 'cluster', 'cluster_name', 'sentiment']
df[output_cols].to_csv("kmeans_clusters.csv", index=False)

cluster_summary = df.groupby('cluster_name').agg(
    article_count=('title', 'count'),
    avg_sentiment=('sentiment', 'mean')
).round(3).reset_index()
cluster_summary.to_csv("cluster_summary.csv", index=False)

print("\nCSV files saved!")
print(df['cluster_name'].value_counts())

# ============================================================
# PCA for 2D visualization
# ============================================================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X.toarray())
df['pca_x'] = X_pca[:, 0]
df['pca_y'] = X_pca[:, 1]

# ============================================================
# FIGURE 1: Main Clustering Dashboard
# ============================================================
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Job Market News - KMeans Topic Clustering Analysis', fontsize=18, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Chart 1: PCA scatter plot of clusters
ax1 = fig.add_subplot(gs[0, 0])
for i, name in cluster_names.items():
    mask = df['cluster'] == i
    ax1.scatter(df[mask]['pca_x'], df[mask]['pca_y'],
                c=colors[i], label=name, alpha=0.6, s=40)
ax1.set_title('Article Clusters (PCA 2D View)', fontsize=12, fontweight='bold')
ax1.set_xlabel('PCA Component 1')
ax1.set_ylabel('PCA Component 2')
ax1.legend(fontsize=7, loc='upper right')
ax1.grid(True, alpha=0.3)

# Chart 2: Article count per cluster
ax2 = fig.add_subplot(gs[0, 1])
cluster_counts = df['cluster_name'].value_counts()
bars = ax2.bar(range(len(cluster_counts)), cluster_counts.values,
               color=colors, edgecolor='black')
ax2.set_xticks(range(len(cluster_counts)))
ax2.set_xticklabels(cluster_counts.index, rotation=30, ha='right', fontsize=9)
ax2.set_title('Article Count per Topic Cluster', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Articles')
for bar, val in zip(bars, cluster_counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             str(val), ha='center', va='bottom', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Chart 3: Average sentiment per cluster
ax3 = fig.add_subplot(gs[1, 0])
cluster_sent = df.groupby('cluster_name')['sentiment'].mean().reindex(cluster_counts.index)
bar_colors = ['#4CAF50' if v > 0 else '#F44336' for v in cluster_sent.values]
bars3 = ax3.bar(range(len(cluster_sent)), cluster_sent.values,
                color=bar_colors, edgecolor='black')
ax3.set_xticks(range(len(cluster_sent)))
ax3.set_xticklabels(cluster_sent.index, rotation=30, ha='right', fontsize=9)
ax3.axhline(y=0, color='black', linewidth=1)
ax3.set_title('Average Sentiment per Topic Cluster', fontsize=12, fontweight='bold')
ax3.set_ylabel('Average Sentiment Score')
for bar, val in zip(bars3, cluster_sent.values):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.008,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# Chart 4: Top words per cluster (horizontal bars)
ax4 = fig.add_subplot(gs[1, 1])
all_words = []
all_colors = []
all_labels = []
for i in range(k):
    words = cluster_top_words[i][:3]
    for w in words:
        all_words.append(w)
        all_colors.append(colors[i])
        all_labels.append(f"C{i}: {w}")

ax4.barh(range(len(all_labels)), [1]*len(all_labels), color=all_colors, edgecolor='white')
ax4.set_yticks(range(len(all_labels)))
ax4.set_yticklabels(all_labels, fontsize=9)
ax4.set_title('Top Keywords per Cluster', fontsize=12, fontweight='bold')
ax4.set_xlabel('Keyword Relevance')
ax4.grid(True, alpha=0.3, axis='x')

plt.savefig('kmeans_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Dashboard saved!")

# ============================================================
# FIGURE 2: Cluster Deep Dive
# ============================================================
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('KMeans Clustering - Deep Dive Analysis', fontsize=16, fontweight='bold')

# Chart 5: Cluster size pie chart
cluster_sizes = df['cluster_name'].value_counts()
axes[0].pie(cluster_sizes.values, labels=cluster_sizes.index,
            autopct='%1.1f%%', colors=colors,
            startangle=90, textprops={'fontsize': 9})
axes[0].set_title('Topic Distribution Across All Articles', fontsize=12, fontweight='bold')

# Chart 6: Sentiment distribution per cluster boxplot
data_by_cluster = [df[df['cluster'] == i]['sentiment'].values for i in range(k)]
bp = axes[1].boxplot(data_by_cluster, patch_artist=True,
                     tick_labels=[cluster_names[i] for i in range(k)])
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_title('Sentiment Distribution by Topic Cluster', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Sentiment Score')
axes[1].tick_params(axis='x', rotation=30)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('kmeans_deep_dive.png', dpi=150, bbox_inches='tight')
plt.close()
print("Deep dive chart saved!")

print("\nAll done! Files saved:")
print("- kmeans_dashboard.png")
print("- kmeans_deep_dive.png")
print("- kmeans_clusters.csv")
print("- cluster_summary.csv")