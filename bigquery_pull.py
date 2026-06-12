import os
from google.cloud import bigquery
import pandas as pd

# Set credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\munni\Downloads\bigquery-practice-499203-5b3f13e0c226.json"

# Create client
client = bigquery.Client()

# Query
query = """
SELECT *
FROM `bigquery-public-data.usa_names.usa_1910_2013`
LIMIT 1000
"""

# Run query
print("Running query...")
df = client.query(query).to_dataframe()

# Show results
print("Shape:", df.shape)
print(df.head())

# Save to CSV
df.to_csv("usa_names_data.csv", index=False)
print("Saved to usa_names_data.csv!")