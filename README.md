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
   ![Prediction Output](<img width="1280" height="558" alt="Screenshot 2025-07-10 223534" src="https://github.com/user-attachments/assets/d6959a0d-92e3-4b56-81ee-4db2b2e28608" />)

3. Neutral Analysis
   ![Prediction Output](<img width="1579" height="681" alt="Screenshot 2025-07-10 233350" src="https://github.com/user-attachments/assets/51c993c1-4125-4ae3-96c5-69615c79d2f3" />
)


| ![Pie Chart](<img width="1242" height="908" alt="Screenshot 2025-07-10 233420" src="https://github.com/user-attachments/assets/3c03af85-e708-478e-bdcc-1bfd0fcf13b3" />
)


---

## 📽️ Live Demo

🌐 **Live Demo**: [Click here to try the app](https://avanish-d-sentiment-analysis-streamlit-app-higew5.streamlit.app/) 

---

## 💻 How to Run Locally

### 🔧 1. Clone the repository
### 🧰 2. Create and activate virtual environment (optional but recommended)
        python -m venv venv
        venv\Scripts\activate  # for Windows
###📦 3. Install dependencies
        pip install -r requirements.txt
###🚀 4. Run the Streamlit app
       streamlit run app.py
