"""
Wrestling-specific text preprocessing pipeline for Phase 2 training.
Handles wrestling terminology, slang, and context-aware cleaning.
"""
import re
import string
from typing import List, Dict, Tuple
from dataclasses import dataclass
import unicodedata


@dataclass
class PreprocessedText:
    """Container for preprocessed text with metadata."""
    original: str
    cleaned: str
    tokens: List[str]
    wrestling_terms: List[str]
    mentions: List[str]
    hashtags: List[str]
    urls: List[str]
    length: int


class WrestlingTextPreprocessor:
    """Wrestling-specific text preprocessor for training data."""
    
    def __init__(self):
        # Wrestling terminology that should be preserved
        self.wrestling_terms = {
            # Performance terms
            'heel', 'face', 'babyface', 'tweener', 'turn', 'gimmick', 'character',
            'work', 'shoot', 'kayfabe', 'booking', 'creative', 'storyline', 'angle',
            'promo', 'segment', 'vignette', 'squash', 'jobber', 'enhancement',
            
            # Match terms
            'botch', 'spot', 'highspot', 'nearfall', 'finisher', 'signature',
            'submission', 'pin', 'count', 'kickout', 'interference', 'dq',
            
            # Backstage terms
            'push', 'burial', 'bury', 'buried', 'heat', 'pop', 'cheap heat',
            'mark', 'smark', 'insider', 'dirt sheet', 'backstage', 'locker room',
            
            # Fan reactions
            'over', 'nuclear heat', 'x-pac heat', 'go away heat', 'what chant',
            'this is awesome', 'holy shit', 'fight forever', 'boring',
            
            # Wrestling moves (common ones)
            'suplex', 'piledriver', 'powerbomb', 'ddt', 'clothesline', 'chokeslam',
            'stunner', 'spear', 'rko', 'f5', 'aa', 'gts', 'sweet chin music',
            
            # Companies/Shows
            'wwe', 'aew', 'njpw', 'tna', 'impact', 'roh', 'nxt', 'raw', 'smackdown',
            'dynamite', 'rampage', 'collision', 'wrestlemania', 'summerslam',
            'royal rumble', 'survivor series', 'money in the bank', 'hell in a cell'
        }
        
        # Convert to lowercase for matching
        self.wrestling_terms_lower = {term.lower() for term in self.wrestling_terms}
        
        # Emojis commonly used in wrestling context
        self.wrestling_emojis = {
            '🔥': 'fire',
            '💯': 'hundred',
            '👑': 'crown',
            '💪': 'muscle',
            '🏆': 'trophy',
            '⭐': 'star',
            '💥': 'boom',
            '👏': 'clap',
            '😴': 'sleep',
            '😡': 'angry',
            '🙄': 'eyeroll',
            '👎': 'thumbsdown',
            '👍': 'thumbsup'
        }
        
        # Contractions to expand
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "'s": " is"  # Note: Could be possessive, but context usually clear
        }
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Replace curly quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('—', ' - ').replace('–', ' - ')
        
        return text
    
    def extract_metadata(self, text: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract mentions, hashtags, and URLs from text."""
        # Extract mentions
        mentions = re.findall(r'@(\w+)', text)
        
        # Extract hashtags
        hashtags = re.findall(r'#(\w+)', text)
        
        # Extract URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        
        return mentions, hashtags, urls
    
    def preserve_wrestling_terms(self, text: str) -> str:
        """Preserve wrestling-specific terms during cleaning."""
        # Replace wrestling terms with placeholders temporarily
        placeholders = {}
        placeholder_counter = 0
        
        # Sort by length (longest first) to avoid partial matches
        sorted_terms = sorted(self.wrestling_terms, key=len, reverse=True)
        
        for term in sorted_terms:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(text):
                placeholder = f"__WRESTLING_TERM_{placeholder_counter}__"
                placeholders[placeholder] = term
                text = pattern.sub(placeholder, text)
                placeholder_counter += 1
        
        return text, placeholders
    
    def restore_wrestling_terms(self, text: str, placeholders: Dict[str, str]) -> str:
        """Restore wrestling terms from placeholders."""
        for placeholder, term in placeholders.items():
            text = text.replace(placeholder, term)
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean text while preserving wrestling context."""
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize unicode
        text = self.normalize_unicode(text)
        
        # Preserve wrestling terms
        text, wrestling_placeholders = self.preserve_wrestling_terms(text)
        
        # Remove URLs (keep for metadata but remove from text)
        text = re.sub(r'http[s]?://\S+', ' ', text)
        
        # Replace emojis with text equivalents for wrestling context
        for emoji, replacement in self.wrestling_emojis.items():
            text = text.replace(emoji, f' {replacement} ')
        
        # Remove other emojis
        text = re.sub(r'[^\w\s@#\'".,!?-]', ' ', text)
        
        # Expand contractions
        for contraction, expansion in self.contractions.items():
            text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
        
        # Handle mentions and hashtags (preserve but normalize)
        text = re.sub(r'@(\w+)', r' mention_\1 ', text)
        text = re.sub(r'#(\w+)', r' hashtag_\1 ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra punctuation but preserve sentence structure
        text = re.sub(r'[.]{2,}', '.', text)  # Multiple periods
        text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamations
        text = re.sub(r'[?]{2,}', '?', text)  # Multiple questions
        
        # Restore wrestling terms
        text = self.restore_wrestling_terms(text, wrestling_placeholders)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization that preserves wrestling terms."""
        # Split on whitespace and punctuation, but keep important punctuation
        tokens = []
        
        # Split text while preserving contractions and wrestling terms
        words = re.findall(r"\b\w+(?:'\w+)?\b|[.!?]", text.lower())
        
        for word in words:
            # Skip very short tokens unless they're punctuation
            if len(word) < 2 and word not in '.!?':
                continue
            tokens.append(word)
        
        return tokens
    
    def extract_wrestling_terms(self, text: str) -> List[str]:
        """Extract wrestling-specific terms found in text."""
        text_lower = text.lower()
        found_terms = []
        
        for term in self.wrestling_terms_lower:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def preprocess(self, text: str) -> PreprocessedText:
        """Complete preprocessing pipeline."""
        if not text:
            return PreprocessedText("", "", [], [], [], [], [], 0)
        
        # Extract metadata before cleaning
        mentions, hashtags, urls = self.extract_metadata(text)
        
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Extract wrestling terms
        wrestling_terms = self.extract_wrestling_terms(cleaned)
        
        return PreprocessedText(
            original=text,
            cleaned=cleaned,
            tokens=tokens,
            wrestling_terms=wrestling_terms,
            mentions=mentions,
            hashtags=hashtags,
            urls=urls,
            length=len(cleaned)
        )
    
    def preprocess_batch(self, texts: List[str]) -> List[PreprocessedText]:
        """Preprocess a batch of texts."""
        return [self.preprocess(text) for text in texts]
    
    def prepare_for_training(self, preprocessed_texts: List[PreprocessedText]) -> List[str]:
        """Prepare preprocessed texts for model training."""
        training_texts = []
        
        for preprocessed in preprocessed_texts:
            # Use cleaned text for training
            text = preprocessed.cleaned
            
            # Filter out very short or very long texts
            if 10 <= len(text) <= 2000:
                training_texts.append(text)
        
        return training_texts
    
    def get_preprocessing_stats(self, preprocessed_texts: List[PreprocessedText]) -> Dict:
        """Get statistics about preprocessed texts."""
        if not preprocessed_texts:
            return {}
        
        total_texts = len(preprocessed_texts)
        avg_length = sum(p.length for p in preprocessed_texts) / total_texts
        
        wrestling_term_counts = {}
        for preprocessed in preprocessed_texts:
            for term in preprocessed.wrestling_terms:
                wrestling_term_counts[term] = wrestling_term_counts.get(term, 0) + 1
        
        return {
            "total_texts": total_texts,
            "average_length": round(avg_length, 2),
            "wrestling_terms_found": len(wrestling_term_counts),
            "most_common_wrestling_terms": sorted(
                wrestling_term_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }


# Singleton instance
wrestling_preprocessor = WrestlingTextPreprocessor()


# Convenience functions
def preprocess_wrestling_text(text: str) -> PreprocessedText:
    """Preprocess a single wrestling text."""
    return wrestling_preprocessor.preprocess(text)


def preprocess_wrestling_texts(texts: List[str]) -> List[PreprocessedText]:
    """Preprocess multiple wrestling texts."""
    return wrestling_preprocessor.preprocess_batch(texts)


def prepare_texts_for_training(texts: List[str]) -> List[str]:
    """Preprocess and prepare texts for model training."""
    preprocessed = wrestling_preprocessor.preprocess_batch(texts)
    return wrestling_preprocessor.prepare_for_training(preprocessed)