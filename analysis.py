import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("job_market_news.csv")

df['source_name'] = df['source'].apply(lambda x: eval(x)['name'] if isinstance(x, str) else 'Unknown')

top_sources = df['source_name'].value_counts().head(10)

plt.figure(figsize=(12, 6))
top_sources.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Top 10 News Sources for Job Market Articles', fontsize=14)
plt.xlabel('News Source', fontsize=12)
plt.ylabel('Number of Articles', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_sources.png')
plt.show()
print("Graph saved!")