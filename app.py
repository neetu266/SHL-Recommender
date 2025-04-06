import pandas as pd
from fuzzywuzzy import fuzz

# Load the CSV correctly
data = pd.read_csv("SHL_TEST.csv")

# Input from user
query = input("Enter job description or role: ")

# Score based on keyword matching
data['score'] = data['Keywords'].apply(lambda x: fuzz.partial_ratio(query.lower(), str(x).lower()))

# Sort and get top 5 results
top_matches = data.sort_values(by='score', ascending=False).head(5)

# Show recommendations
print("\nTop SHL Assessment Recommendations:\n")
print(top_matches[['Name ', 'URL', 'Remote Support', 'Adaptive', 'Duration', 'Test Type']].to_string(index=False))