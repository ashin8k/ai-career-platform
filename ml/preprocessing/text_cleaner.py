import re
import unicodedata
from typing import List, Dict, Any


class TextCleaner:
    """
    Domain-Aware NLP Text Cleaning Pipeline for Technical Resumes and Job Descriptions.
    Preserves programming language symbols (C++, C#, .NET, Node.js, scikit-learn, CI/CD).
    """

    # Tech-specific regex protection mapping to temporary placeholders
    TECH_PROTECTION_MAP = {
        r"\bc\+\+\b": "__TECH_CPP__",
        r"\bc#\b": "__TECH_CSHARP__",
        r"\b\.net\b": "__TECH_DOTNET__",
        r"\bnode\.js\b": "__TECH_NODEJS__",
        r"\breact\.js\b": "__TECH_REACTJS__",
        r"\bvue\.js\b": "__TECH_VUEJS__",
        r"\bscikit-learn\b": "__TECH_SCIKITLEARN__",
        r"\bci/cd\b": "__TECH_CICD__",
        r"\btcp/ip\b": "__TECH_TCPIP__",
        r"\bai/ml\b": "__TECH_AIML__",
    }

    REVERSE_PROTECTION_MAP = {
        "__TECH_CPP__": "c++",
        "__TECH_CSHARP__": "c#",
        "__TECH_DOTNET__": ".net",
        "__TECH_NODEJS__": "node.js",
        "__TECH_REACTJS__": "react.js",
        "__TECH_VUEJS__": "vue.js",
        "__TECH_SCIKITLEARN__": "scikit-learn",
        "__TECH_CICD__": "ci/cd",
        "__TECH_TCPIP__": "tcp/ip",
        "__TECH_AIML__": "ai/ml",
    }

    # Common generic English stopwords (avoiding single-letter languages like 'c', 'r')
    DEFAULT_STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
        "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
        "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
        "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
        "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
        "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
    }

    def __init__(self, custom_stopwords: Optional[set] = None):
        self.stopwords = custom_stopwords if custom_stopwords is not None else self.DEFAULT_STOPWORDS

    def clean_text(self, text: str) -> str:
        """
        Full text cleaning pipeline:
        1. Normalize unicodes & strip bullet symbols.
        2. Lowercase text.
        3. Protect technical keywords (C++, C#, Node.js).
        4. Remove noisy non-alphanumeric punctuation.
        5. Restore protected technical keywords.
        6. Collapse multiple spaces & linebreaks.
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Unicode normalization (NFKD)
        text = unicodedata.normalize("NFKD", text)

        # Step 2: Remove bullet points and special bullet characters
        text = re.sub(r"[\u2022\u2023\u25b6\u25c0\u25e6\u25a0\u25a1\u25ca\u25cb\u25cf\u25fe\u25ff\u2212\-•▪*]", " ", text)

        # Step 3: Convert to lowercase
        text = text.lower()

        # Step 4: Protect special technical terms
        for pattern, placeholder in self.TECH_PROTECTION_MAP.items():
            text = re.sub(pattern, placeholder, text)

        # Step 5: Replace unwanted special characters (keep alphanumeric, spaces, and protected placeholders)
        # Note: We keep underscores temporarily so placeholders like __TECH_CPP__ are preserved
        text = re.sub(r"[^\w\s]", " ", text)

        # Step 6: Restore technical keywords
        for placeholder, original in self.REVERSE_PROTECTION_MAP.items():
            text = text.replace(placeholder.lower(), original)

        # Step 7: Collapse extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenizes cleaned text into a list of normalized words.
        Optionally filters out standard stopwords.
        """
        cleaned = self.clean_text(text)
        tokens = cleaned.split()
        
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1 or t in {"c", "r"}]
            
        return tokens

    def get_corpus_statistics(self, raw_text: str) -> Dict[str, Any]:
        """
        Calculates NLP statistics before and after cleaning.
        Useful for logging, debugging, and resume analytics.
        """
        cleaned = self.clean_text(raw_text)
        raw_tokens = raw_text.split()
        cleaned_tokens = self.tokenize(raw_text, remove_stopwords=True)

        return {
            "raw_char_count": len(raw_text),
            "cleaned_char_count": len(cleaned),
            "raw_word_count": len(raw_tokens),
            "cleaned_word_count": len(cleaned_tokens),
            "vocabulary_size": len(set(cleaned_tokens))
        }
