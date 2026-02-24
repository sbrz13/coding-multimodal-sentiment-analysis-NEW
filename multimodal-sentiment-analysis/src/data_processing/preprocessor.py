import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 下载NLTK资源
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    def __init__(self):
        """文本预处理类"""
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text):
        """
        预处理文本
        
        Args:
            text: 原始文本
        
        Returns:
            预处理后的文本
        """
        # 转换为小写
        text = text.lower()
        
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 移除提及（@username）
        text = re.sub(r'@\w+', '', text)
        
        # 移除标签（#hashtag）
        text = re.sub(r'#\w+', '', text)
        
        # 移除特殊字符和数字
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 分词
        tokens = word_tokenize(text)
        
        # 移除停用词
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # 重新组合为文本
        processed_text = ' '.join(tokens)
        
        return processed_text

class ImagePreprocessor:
    def __init__(self, image_size=224):
        """图像预处理类"""
        self.image_size = image_size
    
    def preprocess(self, image):
        """
        预处理图像
        
        Args:
            image: PIL图像对象
        
        Returns:
            预处理后的图像
        """
        # 调整图像大小
        image = image.resize((self.image_size, self.image_size))
        
        # 转换为RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image

def preprocess_dataset(data, text_preprocessor=None, image_preprocessor=None):
    """
    预处理整个数据集
    
    Args:
        data: 数据集（pandas DataFrame）
        text_preprocessor: 文本预处理对象
        image_preprocessor: 图像预处理对象
    
    Returns:
        预处理后的数据集
    """
    processed_data = data.copy()
    
    # 预处理文本
    if text_preprocessor and 'text' in processed_data.columns:
        processed_data['text'] = processed_data['text'].apply(text_preprocessor.preprocess)
    
    # 注意：图像预处理通常在数据加载时进行，这里不直接处理图像文件
    
    return processed_data
