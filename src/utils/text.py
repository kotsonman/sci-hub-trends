import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
# Загрузка необходимых ресурсов NLTK (выполнить один раз)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
def clean_text(text):
    """Очистка текста от лишних символов и приведение к нижнему регистру."""
    text = text.lower()
    text = re.sub(r'[^a-zа-я0-9\s]', '', text)
    return text

def tokenize_text(text):
    """Токенизация текста и удаление стоп-слов."""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)
