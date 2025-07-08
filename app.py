import streamlit as st
import pickle
import string
import pandas as pd
import os
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download once
nltk.download("stopwords")

# Load model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Clean text
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Save prediction history to CSV
def save_to_history(input_text, prediction, prob):
    file_path = "prediction_history.csv"
    new_row = pd.DataFrame([{
        "Review": input_text,
        "Prediction": prediction,
        "Confidence": round(prob * 100, 2)
    }])

    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        full_df = pd.concat([old_df, new_row], ignore_index=True)
    else:
        full_df = new_row

    full_df.to_csv(file_path, index=False)

# UI
st.set_page_config(page_title="Amazon Sentiment Analyzer", layout="centered")
st.title("📦 Amazon Review Sentiment Analyzer")
st.write("Enter your review below and view your past predictions with charts.")

user_input = st.text_area("📝 Write your review here:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector).max()

        st.markdown(f"### 🎯 Prediction: `{prediction.upper()}`")
        st.markdown(f"✅ Confidence: `{prob*100:.2f}%`")

        # Save to history
        save_to_history(user_input, prediction, prob)

# Display history
st.subheader("🕓 Prediction History")
if os.path.exists("prediction_history.csv"):
    df = pd.read_csv("prediction_history.csv")
    st.dataframe(df.tail(10))  # Show last 10 entries
 # download history button
    if os.path.exists("prediction_history.csv"):
     with open("prediction_history.csv", "rb") as f:
        st.download_button("📥 Download Prediction History", f, file_name="prediction_history.csv")
      
    # clear history  button
    if st.button("🗑️ Clear History"):
     try:
        os.remove("prediction_history.csv")
        st.success("History cleared. Reloading...")
        st.rerun()  # ✅ This will now refresh the app
     except FileNotFoundError:
        st.warning("History already cleared.")

    # Chart of prediction counts
    st.subheader("📊 Prediction Summary")
    fig, ax = plt.subplots()
    df["Prediction"].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)
else:
    st.info("No prediction history yet.")
    

