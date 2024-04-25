from flask import Flask, request, jsonify, render_template
import numpy as np
import random
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input
import json

app = Flask(__name__)

# Load trained model and necessary data
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

model = Sequential([
    Input(shape=(len(words),)),
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(len(output[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights("model_keras.weights.h5")

# Load intents file
with open('intents.json') as file:
    data = json.load(file)

# Define bag_of_words function
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

# Define chat function
def chat(inp):
    results = model.predict(np.array([bag_of_words(inp, words)]))
    results_index = np.argmax(results)
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    return random.choice(responses)

# Define route for chat API
@app.route('/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    message = data['message']
    response = chat(message)
    return jsonify({'response': response})

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
