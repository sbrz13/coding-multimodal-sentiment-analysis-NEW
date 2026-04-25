import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        processed_text = " ".join(tokens)
        return processed_text


class ImagePreprocessor:
    def __init__(self, image_size=224):
        self.image_size = image_size

    def preprocess(self, image):
        image = image.resize((self.image_size, self.image_size))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image


def preprocess_dataset(data, text_preprocessor=None, image_preprocessor=None):
    processed_data = data.copy()
    if text_preprocessor and "text" in processed_data.columns:
        processed_data["text"] = processed_data["text"].apply(text_preprocessor.preprocess)
    return processed_data
