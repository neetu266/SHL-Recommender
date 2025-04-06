import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your SHL assessments CSV
df = pd.read_csv("SHL_TEST.csv")

# Clean and fill missing values
df.fillna("", inplace=True)

# Streamlit UI
st.title("🧠 SHL Assessment Recommender")
st.write("Enter a job description to get the most relevant SHL assessments.")

# Input from user
query = st.text_area("Paste Job Description or Role Here")

if st.button("Recommend Assessments") and query:
    # Combine keywords and name for better matching
    combined_data = df['Name '] + " " + df['Keywords']
    
    # TF-IDF + Cosine Similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(combined_data.tolist() + [query])
    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    # Get top 10 matches
    top_indices = similarity[0].argsort()[-10:][::-1]
    top_matches = df.iloc[top_indices]

    # Display in table
    st.subheader("🔍 Top SHL Recommendations")
    st.dataframe(top_matches[['Name ', 'URL', 'Remote Support', 'Adaptive', 'Duration', 'Test Type']])

    # # Optional: Save results
    # top_matches.to_csv("shl_recommendations_output.csv", index=False)