import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from wordcloud import WordCloud
import re

# Load data
df = pd.read_csv("job_market_news.csv")
df['source_name'] = df['source'].apply(lambda x: eval(x)['name'] if isinstance(x, str) else 'Unknown')
df['publishedAt'] = pd.to_datetime(df['publishedAt'])

# ============================================================
# SENTIMENT ANALYSIS
# ============================================================
def get_sentiment(text):
    if pd.isna(text):
        return 0
    return TextBlob(str(text)).sentiment.polarity

def get_sentiment_label(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_score'] = df['title'].apply(get_sentiment)
df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)
df['desc_sentiment'] = df['description'].apply(get_sentiment)

print("Sentiment Analysis Complete!")
print(df['sentiment_label'].value_counts())

# ============================================================
# ML MODEL - Predict sentiment from title
# ============================================================
df_clean = df[df['title'].notna()].copy()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df_clean['clean_title'] = df_clean['title'].apply(clean_text)

X = df_clean['clean_title']
y = df_clean['sentiment_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

accuracy = (y_pred == y_test).mean()
print(f"\nModel Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================================
# FIGURE 1: Main Analysis Dashboard
# ============================================================
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Job Market News - Sentiment & ML Analysis', fontsize=18, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Chart 1: Sentiment Distribution
ax1 = fig.add_subplot(gs[0, 0])
sentiment_counts = df['sentiment_label'].value_counts()
colors = ['#4CAF50', '#9E9E9E', '#F44336']
bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, 
               color=colors, edgecolor='black')
ax1.set_title('Sentiment Distribution of Headlines', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Articles')
for bar, val in zip(bars, sentiment_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(val), ha='center', va='bottom', fontsize=10)

# Chart 2: Sentiment Score Distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df['sentiment_score'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral')
ax2.axvline(x=df['sentiment_score'].mean(), color='green', linestyle='--', 
            linewidth=2, label=f'Mean: {df["sentiment_score"].mean():.3f}')
ax2.set_title('Sentiment Score Distribution', fontsize=12, fontweight='bold')
ax2.set_xlabel('Sentiment Score')
ax2.set_ylabel('Frequency')
ax2.legend()

# Chart 3: Confusion Matrix
ax3 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, y_pred, labels=['Positive', 'Neutral', 'Negative'])
im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
ax3.set_title('ML Model Confusion Matrix', fontsize=12, fontweight='bold')
ax3.set_xticks([0, 1, 2])
ax3.set_yticks([0, 1, 2])
ax3.set_xticklabels(['Positive', 'Neutral', 'Negative'])
ax3.set_yticklabels(['Positive', 'Neutral', 'Negative'])
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')
for i in range(3):
    for j in range(3):
        ax3.text(j, i, str(cm[i, j]), ha='center', va='center', 
                fontsize=12, color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax3)

# Chart 4: Sentiment by Source
ax4 = fig.add_subplot(gs[1, 1])
source_sentiment = df.groupby('source_name')['sentiment_score'].mean().nlargest(10)
colors4 = ['#4CAF50' if x > 0 else '#F44336' for x in source_sentiment.values]
ax4.barh(source_sentiment.index, source_sentiment.values, color=colors4, edgecolor='black')
ax4.axvline(x=0, color='black', linewidth=1)
ax4.set_title('Top 10 Sources by Avg Sentiment', fontsize=12, fontweight='bold')
ax4.set_xlabel('Average Sentiment Score')

plt.savefig('ml_sentiment_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Dashboard saved!")

# ============================================================
# FIGURE 2: Word Cloud + Model Performance
# ============================================================
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('Job Market News - Word Analysis & Model Performance', fontsize=16, fontweight='bold')

# Chart 5: Word Cloud
positive_text = ' '.join(df[df['sentiment_label'] == 'Positive']['title'].dropna())
wordcloud = WordCloud(width=600, height=400, background_color='white',
                      colormap='Greens', max_words=50).generate(positive_text)
axes[0].imshow(wordcloud, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('Word Cloud - Positive Headlines', fontsize=12, fontweight='bold')

# Chart 6: Model accuracy per class
report = classification_report(y_test, y_pred, output_dict=True)
classes = ['Positive', 'Neutral', 'Negative']
precisions = [report.get(c, {}).get('precision', 0) for c in classes]
recalls = [report.get(c, {}).get('recall', 0) for c in classes]
x = np.arange(len(classes))
width = 0.35
axes[1].bar(x - width/2, precisions, width, label='Precision', color='steelblue', edgecolor='black')
axes[1].bar(x + width/2, recalls, width, label='Recall', color='coral', edgecolor='black')
axes[1].set_title(f'Model Performance (Accuracy: {accuracy:.2%})', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(classes)
axes[1].set_ylabel('Score')
axes[1].legend()
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('ml_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Performance chart saved!")

print("\nAll done! Files saved:")
print("- ml_sentiment_dashboard.png")
print("- ml_performance.png")