# 🛒 Amazon Review Sentiment Analyzer (Streamlit)

A web-based **Sentiment Analysis** tool built with **Streamlit** that analyzes Amazon product reviews using **Natural Language Processing (NLP)** and a **Naive Bayes classifier**.

---

## 🔍 Features

✅ Classifies reviews into:
- **Positive** 😄
- **Neutral** 😐
- **Negative** 😞

✅ Built-in NLP Pipeline:
- Text cleaning
- Stopword removal
- Stemming
- TF-IDF vectorization

✅ User-Friendly UI:
- Confidence score with progress bar 📊
- Prediction history table 📝
- Interactive **pie chart** visualization 📈
- Clear & download history buttons
- Instant results in your browser

---

## 📸 Final Output Preview

| Output Examples |
|-----------------|
1. Positive Analysis 
 ![Prediction Output](https://github.com/avanish-d/Sentiment-Analysis-Streamlit/blob/main/images/Screenshot%202025-07-10%20223505.png)
2. Negative Analysis
   ![Prediction Output](https://github.com/avanish-d/Sentiment-Analysis-Streamlit/blob/main/images/Screenshot%202025-07-10%20223534.png)

3. Neutral Analysis
   ![Prediction Output](https://github.com/avanish-d/Sentiment-Analysis-Streamlit/blob/main/images/Screenshot%202025-07-10%20233350.png
)

4. Analysis in the form of pie chart

  ![Pie Chart](https://github.com/avanish-d/Sentiment-Analysis-Streamlit/blob/main/images/Screenshot%202025-07-10%20233420.png)



---

## 📽️ Live Demo

🌐 **Live Demo**: [Click here to try the app](https://avanish-d-sentiment-analysis-streamlit-app-higew5.streamlit.app/) 

---

## 💻 How to Run Locally

### 🔧 1. Clone the repository
### 🧰 2. Create and activate virtual environment (optional but recommended)
        python -m venv venv
        venv\Scripts\activate  # for Windows
###📦  3. Install dependencies
###🚀 4. Run the Streamlit app
```bash
        !pip install -r requirements.txt
        streamlit run app.py
