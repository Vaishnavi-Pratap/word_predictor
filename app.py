import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

st.title('Text Generator')

with open('my_sentences_new.txt', 'r') as file:
    # Read the entire contents of the file
    df = file.read()

# Load the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([df])

# Load the model
model = load_model('word_generation_model.h5')

def generate_text(seed_text, next_words):
    for _ in range(next_words):
        # Tokenize
        token_text = tokenizer.texts_to_sequences([seed_text])[0]
        # Padding
        padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
        # Predict
        pos = np.argmax(model.predict(padded_token_text), axis=-1)[0]
        for word, index in tokenizer.word_index.items():
            if index == pos:
                seed_text += " " + word
                break
    return seed_text

seed_text = st.text_input('Enter seed text:', '')
next_words = st.number_input('Enter the number of words to generate:', min_value=1, max_value=100, value=1)

if st.button('Generate'):
    generated_text = generate_text(seed_text, next_words)
    st.write('Generated Text:', generated_text)
