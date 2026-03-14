import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)


try:
    from farasa.segmenter import FarasaSegmenter
    FARASA_AVAILABLE = True
    segmenter = FarasaSegmenter(interactive=True)
    logger.info(" Farasa loaded successfully")
except ImportError:
    FARASA_AVAILABLE = False
    logger.warning("Farasa not available, using basic cleaning only")
except Exception as e:
    FARASA_AVAILABLE = False
    logger.warning(f"Farasa initialization failed: {e}")

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

    original_length = len(text)

    text = normalize_arabic(text)
    text = remove_diacritics(text)
    text = remove_non_arabic(text)


    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if FARASA_AVAILABLE and text:
        try:
            text = segmenter.segment(text)
        except Exception as e:
            logger.warning(f"Farasa segmentation failed: {e}")
    
    logger.debug(f"Cleaned text: {original_length} -> {len(text)} chars")
    return text


def batch_clean_texts(texts):
    return [clean_arabic_text(t) for t in texts]



if __name__ == "__main__":
    test_texts = [
        "أنا سعيد جداً اليوم!",
        "هَذَا الْمَوْقِفُ يُحْزِنُنِي",
        "أشعر بالخوف الشديد 100%",
        None,
        ""
    ]
    
    print("🧪 Testing Arabic Preprocessing Module\n")
    
    for text in test_texts:
        print(f"📝 Original: {text}")
        print(f"🧹 Cleaned:  {clean_arabic_text(text)}")
        print("-" * 60)
    
    print(f"\n✅ Farasa available: {FARASA_AVAILABLE}")