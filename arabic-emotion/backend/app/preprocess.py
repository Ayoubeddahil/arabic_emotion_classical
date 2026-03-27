import re
import pandas as pd
import logging
import nltk
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

st = ISRIStemmer()
AR_STOPWORDS = set(stopwords.words('arabic'))

def normalize_arabic(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)   
    text = re.sub("ؤ", "و", text)   
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub("ڤ", "ف", text)
    text = re.sub("چ", "ج", text)
    return text

def remove_diacritics(text):
    arabic_diacritics = re.compile(r'[\u0640\u064B-\u065F]')
    return arabic_diacritics.sub('', text) 

def remove_non_arabic(text):
    return re.sub(r'[^\u0600-\u06FF\s]', '', text)

def clean_arabic_text(text):
    if text is None or pd.isna(text) or not isinstance(text, str):
        return ""
    text = normalize_arabic(text)
    text = remove_diacritics(text)
    text = remove_non_arabic(text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if text:
        words = text.split()
        clean_words = [st.stem(w) for w in words if w not in AR_STOPWORDS]
        text = " ".join(clean_words)
    
    return text

def batch_clean_texts(texts):
    return [clean_arabic_text(t) for t in texts]
