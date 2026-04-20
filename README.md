# Emotion Detection System

### Applied as an Emotion Aware Shopping Customer Service Chatbot

---

## What is This Project?

This project is an NLP-based Emotion Detection system that can detect emotions from text.
The detected emotions are then applied to a Shopping Customer Service Chatbot that gives
responses based on the customer's emotional state.

---

## Main Goal

The main goal of this project is Emotion Detection — training a machine learning model
that can understand and classify emotions from written text such as anger, happiness,
sadness, worry, love, and more.

---

## Emotions Detected

The model can detect 12 emotions:

- 😡 Anger
- 😊 Happiness
- 😢 Sadness
- 😟 Worry
- 💕 Love
- 😤 Hate
- 😄 Fun
- 🎉 Enthusiasm
- 😌 Relief
- 😮 Surprise
- 😐 Neutral
- 😶 Empty

---

## Dataset

- **Source:** Emotion Sentiment Dataset (CSV)
- **Total Rows:** 839,555 (original)
- **After Balancing:** 4,800 rows (400 per emotion)
- **Columns:** text, Emotion

---

## How It Works

```
User types a message
        ↓
Text is preprocessed (lowercasing, punctuation removal, tokenization, stop words removal, lemmatization)
        ↓
TF-IDF converts text into numbers
        ↓
Logistic Regression model predicts the emotion
        ↓
Chatbot gives a response based on the detected emotion
```

---

## Preprocessing Steps

1. Lowercasing
2. URL and Punctuation Removal
3. Tokenization
4. Stop Words Removal
5. Lemmatization

---

## Machine Learning Model

- **Algorithm:** Logistic Regression
- **Feature Extraction:** TF-IDF Vectorizer (5000 features)
- **Train/Test Split:** 80% training, 20% testing
- **Accuracy:** 90.94%

---

## Evaluation

- Accuracy Score: **90.94%**
- Confusion Matrix: shows correct and incorrect predictions per emotion

---

## Project Structure

```
PAI-LAB-PROJECT/
│
├── app.py                  → Main Flask application
├── model.ipynb             → Model training notebook
├── emotion_balanced.csv    → Balanced dataset
├── emotion_model.pkl       → Saved trained model
├── tfidf_vectorizer.pkl    → Saved TF-IDF vectorizer
├── README.md               → Project documentation
│
└── templates/
        └── index.html      → Chatbot web interface
```

---

## How to Run

1. Install required libraries:

```
pip install flask scikit-learn nltk joblib pandas
```

2. Run the Flask app:

```
python app.py
```

3. Open browser and go to:

```
http://127.0.0.1:5000
```

---

## Technologies Used

- Python
- NLTK (Natural Language Processing)
- Scikit-learn (Machine Learning)
- TF-IDF (Feature Extraction)
- Logistic Regression (Classification)
- Flask (Web Framework)
- HTML & CSS (Interface)

---

## Developed By

Najia Khan (123)
