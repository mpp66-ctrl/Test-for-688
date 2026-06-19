import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from textblob import TextBlob
from scipy import stats
from scipy.stats import pearsonr
import re

# Load data
df = pd.read_csv("job_market_news.csv")
df['source_name'] = df['source'].apply(lambda x: eval(x)['name'] if isinstance(x, str) else 'Unknown')
df['publishedAt'] = pd.to_datetime(df['publishedAt'])

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def get_sentiment(text):
    if pd.isna(text):
        return 0
    return TextBlob(str(text)).sentiment.polarity

def word_count(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())

def char_count(text):
    if pd.isna(text):
        return 0
    return len(str(text))

df['title_sentiment'] = df['title'].apply(get_sentiment)
df['desc_sentiment'] = df['description'].apply(get_sentiment)
df['title_length'] = df['title'].apply(char_count)
df['title_word_count'] = df['title'].apply(word_count)
df['desc_length'] = df['description'].apply(char_count)
df['desc_word_count'] = df['description'].apply(word_count)

# Proxy for "popularity": how many other articles share similar source tier
# We'll use source frequency as a popularity/reach proxy
source_counts = df['source_name'].value_counts()
df['source_reach'] = df['source_name'].map(source_counts)

# Drop rows with missing key data
df_clean = df.dropna(subset=['title_sentiment', 'title_length', 'desc_length', 'source_reach']).copy()

print(f"Analyzing {len(df_clean)} articles")

# ============================================================
# CORRELATION CALCULATIONS
# ============================================================
corr_sent_titlelen, p1 = pearsonr(df_clean['title_sentiment'], df_clean['title_length'])
corr_sent_desclen, p2 = pearsonr(df_clean['title_sentiment'], df_clean['desc_length'])
corr_sent_reach, p3 = pearsonr(df_clean['title_sentiment'], df_clean['source_reach'])
corr_titlelen_reach, p4 = pearsonr(df_clean['title_length'], df_clean['source_reach'])

print("\n--- Correlation Results ---")
print(f"Title Sentiment vs Title Length: r={corr_sent_titlelen:.3f}, p={p1:.3f}")
print(f"Title Sentiment vs Description Length: r={corr_sent_desclen:.3f}, p={p2:.3f}")
print(f"Title Sentiment vs Source Reach: r={corr_sent_reach:.3f}, p={p3:.3f}")
print(f"Title Length vs Source Reach: r={corr_titlelen_reach:.3f}, p={p4:.3f}")

# ============================================================
# SAVE PROCESSED DATA TO CSV
# ============================================================
output_cols = ['title', 'source_name', 'title_sentiment', 'desc_sentiment',
               'title_length', 'title_word_count', 'desc_length',
               'desc_word_count', 'source_reach', 'publishedAt']
df_clean[output_cols].to_csv("correlation_analysis_data.csv", index=False)

corr_summary = pd.DataFrame({
    'comparison': ['Sentiment vs Title Length', 'Sentiment vs Description Length',
                   'Sentiment vs Source Reach', 'Title Length vs Source Reach'],
    'correlation_r': [corr_sent_titlelen, corr_sent_desclen, corr_sent_reach, corr_titlelen_reach],
    'p_value': [p1, p2, p3, p4]
})
corr_summary.to_csv("correlation_summary.csv", index=False)
print("\nCSV files saved!")

# ============================================================
# FIGURE 1: Correlation Dashboard
# ============================================================
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Sentiment vs Article Length & Reach: Correlation Analysis', fontsize=18, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Chart 1: Sentiment vs Title Length scatter
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(df_clean['title_length'], df_clean['title_sentiment'], 
            alpha=0.4, color='steelblue', s=30)
z = np.polyfit(df_clean['title_length'], df_clean['title_sentiment'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_clean['title_length'].min(), df_clean['title_length'].max(), 100)
ax1.plot(x_line, p(x_line), color='red', linewidth=2, linestyle='--')
ax1.set_title(f'Sentiment vs Title Length (r={corr_sent_titlelen:.3f})', fontsize=12, fontweight='bold')
ax1.set_xlabel('Title Length (characters)')
ax1.set_ylabel('Title Sentiment Score')
ax1.grid(True, alpha=0.3)

# Chart 2: Sentiment vs Description Length scatter
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(df_clean['desc_length'], df_clean['title_sentiment'], 
            alpha=0.4, color='coral', s=30)
z2 = np.polyfit(df_clean['desc_length'], df_clean['title_sentiment'], 1)
p2_line = np.poly1d(z2)
x_line2 = np.linspace(df_clean['desc_length'].min(), df_clean['desc_length'].max(), 100)
ax2.plot(x_line2, p2_line(x_line2), color='red', linewidth=2, linestyle='--')
ax2.set_title(f'Sentiment vs Description Length (r={corr_sent_desclen:.3f})', fontsize=12, fontweight='bold')
ax2.set_xlabel('Description Length (characters)')
ax2.set_ylabel('Title Sentiment Score')
ax2.grid(True, alpha=0.3)

# Chart 3: Sentiment vs Source Reach
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(df_clean['source_reach'], df_clean['title_sentiment'], 
            alpha=0.4, color='green', s=30)
z3 = np.polyfit(df_clean['source_reach'], df_clean['title_sentiment'], 1)
p3_line = np.poly1d(z3)
x_line3 = np.linspace(df_clean['source_reach'].min(), df_clean['source_reach'].max(), 100)
ax3.plot(x_line3, p3_line(x_line3), color='red', linewidth=2, linestyle='--')
ax3.set_title(f'Sentiment vs Source Reach (r={corr_sent_reach:.3f})', fontsize=12, fontweight='bold')
ax3.set_xlabel('Source Reach (article count)')
ax3.set_ylabel('Title Sentiment Score')
ax3.grid(True, alpha=0.3)

# Chart 4: Correlation heatmap
ax4 = fig.add_subplot(gs[1, 1])
corr_matrix = df_clean[['title_sentiment', 'title_length', 'desc_length', 'source_reach']].corr()
im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax4.set_xticks(range(len(corr_matrix.columns)))
ax4.set_yticks(range(len(corr_matrix.columns)))
labels = ['Sentiment', 'Title Len', 'Desc Len', 'Reach']
ax4.set_xticklabels(labels, rotation=45, ha='right')
ax4.set_yticklabels(labels)
ax4.set_title('Correlation Matrix Heatmap', fontsize=12, fontweight='bold')
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center',
                color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
plt.colorbar(im, ax=ax4)

plt.savefig('correlation_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Dashboard saved!")

# ============================================================
# FIGURE 2: Distribution Comparisons
# ============================================================
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('Article Length Distribution by Sentiment Category', fontsize=16, fontweight='bold')

def sentiment_cat(s):
    if s > 0.05:
        return 'Positive'
    elif s < -0.05:
        return 'Negative'
    return 'Neutral'

df_clean['sentiment_cat'] = df_clean['title_sentiment'].apply(sentiment_cat)

# Chart 5: Box plot title length by sentiment
data_to_plot = [df_clean[df_clean['sentiment_cat'] == cat]['title_length'].values 
                for cat in ['Positive', 'Neutral', 'Negative']]
bp = axes[0].boxplot(data_to_plot, labels=['Positive', 'Neutral', 'Negative'], patch_artist=True)
colors_box = ['#4CAF50', '#9E9E9E', '#F44336']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_title('Title Length by Sentiment Category', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Title Length (characters)')
axes[0].grid(True, alpha=0.3, axis='y')

# Chart 6: Average source reach by sentiment
avg_reach = df_clean.groupby('sentiment_cat')['source_reach'].mean().reindex(['Positive', 'Neutral', 'Negative'])
bars = axes[1].bar(avg_reach.index, avg_reach.values, color=colors_box, edgecolor='black')
axes[1].set_title('Average Source Reach by Sentiment', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Average Source Reach (article count)')
for bar, val in zip(bars, avg_reach.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('correlation_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Distribution chart saved!")

print("\nAll done! Files saved:")
print("- correlation_dashboard.png")
print("- correlation_distributions.png")
print("- correlation_analysis_data.csv")
print("- correlation_summary.csv")