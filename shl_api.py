from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="SHL Recommender API")

@app.get("/")
def root():
    return {"status": "API is live"}


# Load dataset
df = pd.read_csv("SHL_TEST.csv")
df.fillna("", inplace=True)

# Prepare TF-IDF vectors once at startup
combined_data = df['Name '] + " " + df['Keywords']
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(combined_data)

# Define request model
class QueryInput(BaseModel):
    query: str

# Define endpoint
@app.post("/recommend/")
def recommend_assessments(input: QueryInput):
    query = input.query
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = similarity.argsort()[-10:][::-1]
    top_results = df.iloc[top_indices][['Name ', 'URL', 'Remote Support', 'Adaptive', 'Duration', 'Test Type']]
    return top_results.to_dict(orient="records")
