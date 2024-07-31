import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from googletrans import Translator


# Background
background_image_url = "https://www.awf.org/sites/default/files/Website_SpeciesPage_Lion01_Hero.jpg"
background_css = """
<style>
    .stApp {{
        background: url("{image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
</style>
""".format(image=background_image_url)
st.markdown(background_css, unsafe_allow_html=True)

# Define Tokenizer
class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = []
        for text in X:
            tokens = word_tokenize(text)
            tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stopwords.words('english')]
            X_transformed.append(' '.join(tokens))
        return X_transformed

# Load dataset
ds = pd.read_csv('SA_official_languages.csv')

# Define label mapping
label_mapping = {
    'xho': 'IsiXhosa',
    'eng': 'English',
    'nso': 'Sepedi',
    'ven': 'Tshivenda',
    'tsn': 'Setswana',
    'nbl': 'isiNdebele',
    'zul': 'IsiZulu',
    'ssw': 'SiSwati',
    'tso': 'Xitsonga',
    'sot': 'Sesotho',
    'afr': 'Afrikaans'
}

# Define reverse label mapping for translation
language_code_mapping = {
    'IsiXhosa': 'xh',
    'English': 'en',
    'Afrikaans': 'af',
    'IsiZulu': 'zu',
    'Sesotho': 'st'
}

import pickle

with open("zzz.pkl", 'rb') as handle:
            pipeline = pickle.load(handle)

# Function to predict language
def predict_language(words):
    sentence = ' '.join(words.split())
    prediction = pipeline.predict([sentence])[0]
    predicted_language = label_mapping.get(prediction, 'Unknown')
    return predicted_language

# Initialize Google Translator
translator = Translator()

# Main function to run the Streamlit app
def main():
    st.title('South African Language Classifier')

    user_input = st.text_input('Enter text to classify')

    if st.button('Classify Language'):
        if user_input:
            prediction = predict_language(user_input)
            st.success(f'**This sentence is written in {prediction} language:**')
        else:
            st.warning('Please enter text to classify.')

    selected_languages = st.multiselect(
        'Select languages to translate to',
        options=['IsiXhosa', 'English', 'Afrikaans', 'IsiZulu', 'Sesotho']
    )

    if st.button('Translate'):
        if user_input:
            predicted_language = predict_language(user_input)
            if selected_languages:
                translations = []
                for lang in selected_languages:
                    lang_code = language_code_mapping.get(lang)
                    if lang_code:
                        try:
                            translated_text = translator.translate(user_input, dest=lang_code).text
                            translations.append(f'{lang}: {translated_text}')
                        except Exception as e:
                            translations.append(f'{lang}: Error in translation ({e})')
                
                if translations:
                    st.write('Translations:')
                    for translation in translations:
                        st.write(translation)
            else:
                st.warning('Please select at least one language to translate to.')
        else:
            st.warning('Please enter text to translate.')

# Run the app
if __name__ == '__main__':
    main()
