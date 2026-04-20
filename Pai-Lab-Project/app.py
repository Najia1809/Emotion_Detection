from flask import Flask, render_template, request, jsonify
import joblib
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random

app = Flask(__name__)

# load model and tfidf
model = joblib.load('emotion_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# preprocessing setup
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# emotion responses
emotion_responses = {
    "anger": [
        "We sincerely apologize for your experience! Please share your order ID and we will resolve this immediately 🙏",
        "We are really sorry to hear that! Our team will make this right for you as soon as possible 🙏",
        "We completely understand your frustration and we apologize. Let us fix this for you right away!"
    ],
    "happiness": [
        "That is so wonderful to hear! We are thrilled you are happy with your purchase 😊",
        "Thank you so much! Your happiness means everything to us 😊",
        "We are so glad you are satisfied! Do not forget to leave us a review 🌟"
    ],
    "sadness": [
        "We are really sorry you are feeling this way. How can we make your experience better? 💙",
        "That makes us sad too! Please tell us what went wrong and we will do our best to help 💙",
        "We sincerely apologize for disappointing you. Let us make it up to you 💙"
    ],
    "worry": [
        "No worries at all! We offer free returns so you can shop with confidence 😊",
        "Please do not worry! Our customer support is available 24/7 to help you 😊",
        "We completely understand your concern. Your order is safe with us and we will keep you updated!"
    ],
    "hate": [
        "We are extremely sorry for your terrible experience! Please let us know what happened so we can fix it immediately 🙏",
        "We sincerely apologize! This is not the experience we want for our customers. Let us make it right 🙏",
        "We are so sorry to hear this! Please share your details and our team will resolve this urgently 🙏"
    ],
    "love": [
        "Aww we love you too! Thank you so much for your kind words 💕",
        "That is so sweet! We are so happy you love your purchase 💕",
        "Thank you for the love! We work hard to make our customers happy 💕"
    ],
    "surprise": [
        "Oh we love surprises too! Hope it was a good one 😄",
        "We hope you were pleasantly surprised! Let us know if you need anything 😄",
        "Surprises are the best! We hope your experience exceeded your expectations 😄"
    ],
    "enthusiasm": [
        "We love your energy! We are just as excited to serve you 🎉",
        "Your enthusiasm makes us happy! Let us know how we can help you today 🎉",
        "Yay we are so excited too! Welcome to our store 🎉"
    ],
    "fun": [
        "Shopping should always be fun! Let us know if you need any help 😄",
        "We love that you are having fun! That is what we are here for 😄",
        "Fun and shopping go hand in hand! Enjoy your experience with us 😄"
    ],
    "relief": [
        "We are so relieved too! We always want our customers to be satisfied 😊",
        "That is great to hear! We are glad everything worked out for you 😊",
        "Relief is the best feeling! We are happy we could help you 😊"
    ],
    "empty": [
        "We are sorry you are feeling this way. Is there anything we can help you with today? 💙",
        "We care about how you feel! Please let us know how we can improve your experience 💙",
        "We want to make your day better! Tell us what you need and we will help 💙"
    ],
    "neutral": [
        "Thank you for reaching out! How can we assist you today? 😊",
        "Hello! Welcome to our customer service. How can we help you? 😊",
        "Hi there! Please let us know what you need and we will be happy to help 😊"
    ]
}

# preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# predict emotion function
def predict_emotion(text):
    clean = preprocess(text)
    vector = tfidf.transform([clean])
    prediction = model.predict(vector)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    emotion = predict_emotion(user_message)
    response = random.choice(emotion_responses[emotion])
    return jsonify({
        'response': response,
        'emotion': emotion
    })

if __name__ == '__main__':
    app.run(debug=True)