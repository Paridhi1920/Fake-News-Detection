import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection")
st.write("Enter the news article below to check if it's FAKE or REAL !!")

#Taking user input
user_input = st.text_area("Paste the article here", height=200)

if st.button("Predict"):
    if user_input.strip():
        # Transform input text
        transformed_text = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(transformed_text)[0]

        # Display result
        if prediction == "1":
            st.error("🚨 This news article is **FAKE**!")
        else:
            st.success("✅ This news article is **REAL**!")
    else:
        st.warning("⚠️ Please enter some text.")