import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
import re

# Load data
df = pd.read_csv("job_market_news.csv")

# Clean source column
df['source_name'] = df['source'].apply(lambda x: eval(x)['name'] if isinstance(x, str) else 'Unknown')

# Clean date column
df['publishedAt'] = pd.to_datetime(df['publishedAt'])
df['date'] = df['publishedAt'].dt.date
df['hour'] = df['publishedAt'].dt.hour
df['day_of_week'] = df['publishedAt'].dt.day_name()

# Clean text
def clean_text(text):
    if pd.isna(text):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

stopwords = {'the','a','an','in','of','to','and','for','on','is','are',
             'with','at','by','from','as','its','that','this','it','was',
             'be','have','has','will','not','but','or','about','more','new'}

def get_top_words(series, n=10):
    all_words = ' '.join(series.apply(clean_text)).split()
    filtered = [w for w in all_words if w not in stopwords and len(w) > 3]
    return Counter(filtered).most_common(n)

# ============================================================
# FIGURE 1: Dashboard with 4 charts
# ============================================================
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Job Market News Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Chart 1: Top 10 News Sources
ax1 = fig.add_subplot(gs[0, 0])
top_sources = df['source_name'].value_counts().head(10)
bars = ax1.bar(range(len(top_sources)), top_sources.values, color='steelblue', edgecolor='black')
ax1.set_xticks(range(len(top_sources)))
ax1.set_xticklabels(top_sources.index, rotation=45, ha='right', fontsize=8)
ax1.set_title('Top 10 News Sources', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Articles')
for bar, val in zip(bars, top_sources.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(val),
             ha='center', va='bottom', fontsize=8)

# Chart 2: Articles Over Time
ax2 = fig.add_subplot(gs[0, 1])
articles_by_date = df.groupby('date').size()
ax2.plot(range(len(articles_by_date)), articles_by_date.values,
         color='green', marker='o', markersize=4, linewidth=2)
ax2.fill_between(range(len(articles_by_date)), articles_by_date.values, alpha=0.3, color='green')
ax2.set_title('Articles Published Over Time', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Articles')
ax2.set_xlabel('Days')
ax2.grid(True, alpha=0.3)

# Chart 3: Top Words in Headlines
ax3 = fig.add_subplot(gs[1, 0])
top_words = get_top_words(df['title'], 10)
words_df = pd.DataFrame(top_words, columns=['word', 'count'])
bars3 = ax3.barh(words_df['word'], words_df['count'], color='coral', edgecolor='black')
ax3.set_title('Top 10 Words in Headlines', fontsize=12, fontweight='bold')
ax3.set_xlabel('Count')
for bar, val in zip(bars3, words_df['count']):
    ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             str(val), ha='left', va='center', fontsize=8)

# Chart 4: Publishing Hour Distribution
ax4 = fig.add_subplot(gs[1, 1])
hour_counts = df['hour'].value_counts().sort_index()
ax4.bar(hour_counts.index, hour_counts.values, color='purple', edgecolor='black', alpha=0.7)
ax4.set_title('Articles by Hour of Day (UTC)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Hour of Day')
ax4.set_ylabel('Number of Articles')
ax4.set_xticks(range(0, 24, 2))
ax4.grid(True, alpha=0.3, axis='y')

plt.savefig('dashboard_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("Dashboard saved!")

# ============================================================
# FIGURE 2: Day of Week + Description Word Analysis
# ============================================================
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('Job Market News - Extended Analysis', fontsize=16, fontweight='bold')

# Chart 5: Day of Week
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_counts = df['day_of_week'].value_counts().reindex(day_order, fill_value=0)
colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336','#795548','#607D8B']
axes[0].bar(day_counts.index, day_counts.values, color=colors, edgecolor='black')
axes[0].set_title('Articles by Day of Week', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Number of Articles')
axes[0].tick_params(axis='x', rotation=30)
for i, val in enumerate(day_counts.values):
    axes[0].text(i, val + 0.1, str(val), ha='center', va='bottom', fontsize=9)

# Chart 6: Top Words in Descriptions
top_desc_words = get_top_words(df['description'], 10)
desc_df = pd.DataFrame(top_desc_words, columns=['word', 'count'])
axes[1].barh(desc_df['word'], desc_df['count'], color='teal', edgecolor='black')
axes[1].set_title('Top 10 Words in Article Descriptions', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Count')

plt.tight_layout()
plt.savefig('extended_analysis_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("Extended analysis saved!")

print("\nAll charts saved successfully!")
print("Files: dashboard_chart.png, extended_analysis_chart.png")