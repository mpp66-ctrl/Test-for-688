import requests
import pandas as pd

API_KEY = "your_api_key_here"
url = "https://newsapi.org/v2/everything"

params = {
    "q": "job market hiring employment",
    "language": "en",
    "sortBy": "publishedAt",
    "pageSize": 100,
    "apiKey": API_KEY
}

response = requests.get(url, params=params)
data = response.json()

print("Status:", data["status"])
print("Total articles found:", data["totalResults"])

articles = data["articles"]
df = pd.DataFrame(articles)

print("Shape:", df.shape)
df.to_csv("job_market_news.csv", index=False)
print("Saved!") 
