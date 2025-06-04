import os
import json
import argparse
import sys
import re
import math
import time
import random
import traceback
import gc
import psutil
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Iterator, Generator, Union
from collections import Counter, defaultdict
from pathlib import Path
from itertools import product
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Try importing ML libraries with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. ML training will not work.")

try:
    from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers library not available. BERT-based models will not work.")

try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("Warning: Qdrant client not available. Only local data mode will work.")

try:
    import tqdm
    from tqdm.auto import tqdm as tqdm_auto
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Progress bars will be disabled.")
    
    # Define a simple fallback for tqdm
    class DummyTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
            
        def __iter__(self):
            return iter(self.iterable)
            
        def update(self, *args, **kwargs):
            pass
            
        def close(self):
            pass
            
        def set_description(self, *args, **kwargs):
            pass
    
    tqdm = DummyTqdm
    tqdm_auto = DummyTqdm

# Set to True to enable detailed logging
DEBUG = True

def log(message: str) -> None:
    """Print debug message if DEBUG is enabled."""
    if DEBUG:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[DEBUG {timestamp}] {message}")

def get_memory_usage() -> str:
    """Get current memory usage in a human-readable format."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return f"{mem_info.rss / (1024 * 1024):.1f} MB"

def log_memory() -> None:
    """Log current memory usage."""
    if DEBUG:
        log(f"Memory usage: {get_memory_usage()}")

class BatchFileHandler:
    """Handles reading and writing data in batches to reduce memory usage."""
    
    def __init__(self, base_path: str, batch_size: int = 1000):
        self.base_path = Path(base_path)
        self.batch_size = batch_size
        self.current_batch = []
        self.current_batch_index = 0
        
    def add_item(self, item: Any) -> None:
        """Add an item to the current batch, writing to disk if batch is full."""
        self.current_batch.append(item)
        
        if len(self.current_batch) >= self.batch_size:
            self.flush()
    
    def flush(self) -> None:
        """Write the current batch to disk."""
        if not self.current_batch:
            return
            
        batch_file = self.base_path.with_suffix(f'.batch_{self.current_batch_index}.json')
        
        # Ensure directory exists
        batch_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_batch, f)
        
        log(f"Wrote batch {self.current_batch_index} with {len(self.current_batch)} items to {batch_file}")
        
        # Clear batch and increment index
        self.current_batch = []
        self.current_batch_index += 1
        
        # Force garbage collection to free memory
        gc.collect()
    
    def read_batches(self) -> Generator[List[Any], None, None]:
        """Read batches from disk one at a time."""
        batch_index = 0
        
        while True:
            batch_file = self.base_path.with_suffix(f'.batch_{batch_index}.json')
            
            if not batch_file.exists():
                break
                
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch = json.load(f)
                    yield batch
            except Exception as e:
                log(f"Error reading batch {batch_index}: {e}")
                break
                
            batch_index += 1
    
    def count_items(self) -> int:
        """Count total items across all batches."""
        total = 0
        
        for batch in self.read_batches():
            total += len(batch)
        
        # Add items in current batch
        total += len(self.current_batch)
        
        return total
    
    def clear_all_batches(self) -> None:
        """Remove all batch files."""
        batch_index = 0
        
        while True:
            batch_file = self.base_path.with_suffix(f'.batch_{batch_index}.json')
            
            if not batch_file.exists():
                break
                
            os.remove(batch_file)
            log(f"Removed batch file: {batch_file}")
            batch_index += 1
        
        # Reset state
        self.current_batch = []
        self.current_batch_index = 0

class ArabicSentimentDataset(Dataset):
    """Dataset for Arabic sentiment analysis."""
    
    def __init__(self, texts, labels, analyzer=None, tokenizer=None, max_length=128):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            labels: List of sentiment labels (positive, negative, neutral)
            analyzer: Optional sentiment analyzer for feature extraction
            tokenizer: Optional tokenizer for BERT models
            max_length: Maximum sequence length for tokenization
        """
        self.texts = texts
        
        # Convert string labels to numeric
        label_map = {"positive": 0, "negative": 1, "neutral": 2}
        self.labels = [label_map.get(label, 0) for label in labels]
        
        self.analyzer = analyzer
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Cache features if analyzer is provided
        self.features = None
        if analyzer:
            self.features = []
            for text in texts:
                result = analyzer.analyze_text(text)
                # Extract features from analysis result
                features = self._extract_features(result)
                self.features.append(features)
    
    def _extract_features(self, analysis_result):
        """Extract features from sentiment analysis result."""
        features = [
            analysis_result.get('pos_score', 0.0),
            analysis_result.get('neg_score', 0.0),
            analysis_result.get('total_score', 0.0),
            analysis_result.get('emoji_sentiment', 0.0),
            analysis_result.get('phrase_sentiment', 0.0),
            float(len(analysis_result.get('pos_words', []))),
            float(len(analysis_result.get('neg_words', []))),
            float(len(analysis_result.get('emojis', []))),
            float(len(analysis_result.get('phrases', []))),
            analysis_result.get('score', 0.5),
        ]
        return features
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.tokenizer:
            # BERT model input
            encoding = self.tokenizer(
                text, 
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length',
                truncation=True
            )
            
            # Remove batch dimension
            item = {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': torch.tensor(label)
            }
            
            # Add extracted features if available
            if self.features:
                item['features'] = torch.tensor(self.features[idx], dtype=torch.float)
                
            return item
        elif self.features:
            # Rule-based features + ML model
            return torch.tensor(self.features[idx], dtype=torch.float), torch.tensor(label)
        else:
            # Just return text and label
            return text, label

# Neural network models
class SentimentClassifier(nn.Module):
    """Basic neural network for sentiment classification."""
    
    def __init__(self, input_dim, hidden_dims, output_dim=3, dropout_rate=0.2):
        """
        Initialize sentiment classifier.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output classes (default 3: positive, negative, neutral)
            dropout_rate: Dropout rate for regularization
        """
        super(SentimentClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class BertSentimentClassifier(nn.Module):
    """BERT-based classifier for Arabic sentiment analysis."""
    
    def __init__(self, bert_model_name="asafaya/bert-base-arabic", num_classes=3, 
                 feature_dim=0, dropout_rate=0.1):
        """
        Initialize BERT-based sentiment classifier.
        
        Args:
            bert_model_name: Name of pretrained BERT model
            num_classes: Number of output classes
            feature_dim: Dimension of additional features to concatenate
            dropout_rate: Dropout rate for regularization
        """
        super(BertSentimentClassifier, self).__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available. Cannot use BERT models.")
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Get BERT output dimension
        bert_dim = self.bert.config.hidden_size
        
        # Create classifier based on whether we have additional features
        self.feature_dim = feature_dim
        if feature_dim > 0:
            self.classifier = nn.Linear(bert_dim + feature_dim, num_classes)
        else:
            self.classifier = nn.Linear(bert_dim, num_classes)
    
    def forward(self, input_ids, attention_mask, features=None):
        # Get BERT output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Concatenate with additional features if provided
        if features is not None and self.feature_dim > 0:
            pooled_output = torch.cat([pooled_output, features], dim=1)
        
        # Classify
        logits = self.classifier(pooled_output)
        return logits

class HybridSentimentClassifier(nn.Module):
    """Hybrid model combining rule-based features with neural network."""
    
    def __init__(self, input_dim, hidden_dims, output_dim=3, dropout_rate=0.2):
        """
        Initialize hybrid sentiment classifier.
        
        Args:
            input_dim: Dimension of rule-based features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(HybridSentimentClassifier, self).__init__()
        
        self.classifier = SentimentClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate
        )
    
    def forward(self, features):
        return self.classifier(features)

class OptimizedArabicSentimentAnalyzer:
    """
    Memory-optimized Arabic sentiment analyzer with batch processing capabilities
    and machine learning integration, designed to achieve 85%+ accuracy.
    """
    
    def __init__(
        self,
        collection_name: str = "sentiment_analysis_dataset",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        force_new_analysis: bool = True,
        cache_dir: str = "./sentiment_cache",
        model_dir: str = "./sentiment_models",
        batch_size: int = 1000,
        params: Optional[Dict[str, Any]] = None,
        memory_limit_mb: int = 1024,  # 1GB default memory limit
        use_local_data: bool = False,
        local_data_path: Optional[str] = None,
        use_ml: bool = False,
        ml_model_type: str = "hybrid",  # "hybrid", "bert", or "neural"
        bert_model_name: str = "asafaya/bert-base-arabic",
        device: str = None
    ):
        self.collection_name = collection_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        self.memory_limit_mb = memory_limit_mb
        self.use_local_data = use_local_data
        self.local_data_path = local_data_path
        
        # ML configuration
        self.use_ml = use_ml and TORCH_AVAILABLE
        self.ml_model_type = ml_model_type
        self.bert_model_name = bert_model_name
        
        # Set device for PyTorch
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        else:
            self.device = torch.device(device) if TORCH_AVAILABLE else None
        
        if self.use_ml and self.device:
            log(f"Using device: {self.device}")
        
        # Force new analysis
        if force_new_analysis:
            self._clear_cache()
        
        # Set default parameters
        self.params = {
            # Bias factors
            "positive_bias": 5.0,
            "negative_bias": 4.0,
            "neutral_threshold": -0.1,  # Negative value means neutral zone is smaller
            
            # Classification thresholds
            "min_pos_threshold": 0.1,   # Min score to classify as positive
            "min_neg_threshold": 0.1,   # Min score to classify as negative
            
            # Negation handling
            "negation_factor": 1.5,     # How strongly negation impacts sentiment
            "negation_window": 5,       # How many words after negation are affected
            
            # Contextual analysis
            "context_weight": 1.3,      # Weight for contextual features
            "emoji_weight": 2.0,        # Weight for emoji sentiment
            
            # Domain-specific tweaks
            "domain_boost": 1.5,        # Boost for domain-specific terms
            "arabic_boost": 1.2,        # Boost for arabic terms vs foreign words
            
            # Advanced features
            "use_contextual_valence_shifters": True,  # Consider words that modify sentiment
            "apply_dynamic_weighting": True,          # Weight words based on position
            "prioritize_domain_matches": True,        # Prioritize domain-specific matches
            "aggressive_classification": True,        # Aggressively avoid neutral class
            
            # Class weighting (for imbalanced data)
            "positive_weight": 2.0,
            "negative_weight": 3.0,
            "neutral_weight": 0.5,
            
            # ML model parameters
            "learning_rate": 2e-5,
            "epochs": 10,
            "train_batch_size": 32,
            "eval_batch_size": 64,
            "early_stopping_patience": 3,
            "hidden_dims": [64, 32],    # Hidden layer dimensions
            "dropout_rate": 0.3,
            "weight_decay": 0.01,
            "feature_dim": 10,          # Dimension of rule-based features
            "warmup_steps": 0,
        }
        
        # Override with provided parameters
        if params:
            self.params.update(params)
        
        # Connect to Qdrant if not using local data
        if not self.use_local_data:
            if not QDRANT_AVAILABLE:
                log("Qdrant client not available but required for Qdrant mode")
                self.client = None
            else:
                try:
                    self.client = QdrantClient(
                        host=qdrant_host,
                        port=qdrant_port,
                        prefer_grpc=False,
                        timeout=120.0
                    )
                    log(f"Successfully connected to Qdrant at {qdrant_host}:{qdrant_port}")
                except Exception as e:
                    log(f"Error connecting to Qdrant: {e}")
                    self.client = None
        else:
            self.client = None
            log("Using local data instead of Qdrant")
            
        # Load lexicons and resources
        self._load_resources()
        
        # Initialize ML model and tokenizer
        self.model = None
        self.tokenizer = None
        if self.use_ml:
            self._initialize_ml_components()
        
        log_memory()
    
    def _clear_cache(self) -> None:
        """Clear cache files for a fresh analysis."""
        results_file = self.cache_dir / f"{self.collection_name}_results.pkl"
        if results_file.exists():
            log(f"Deleting old results file: {results_file}")
            os.remove(results_file)
            
        # Find and remove batch files
        batch_pattern = f"{self.collection_name}_*.batch_*.json"
        for batch_file in self.cache_dir.glob(batch_pattern):
            log(f"Deleting old batch file: {batch_file}")
            os.remove(batch_file)
            
        # Find and remove metrics files
        metrics_file = self.cache_dir / f"{self.collection_name}_metrics.json"
        if metrics_file.exists():
            log(f"Deleting old metrics file: {metrics_file}")
            os.remove(metrics_file)

    def _initialize_ml_components(self) -> None:
        """Initialize ML model and tokenizer based on configuration."""
        if not self.use_ml or not TORCH_AVAILABLE:
            return
            
        log(f"Initializing ML components with model type: {self.ml_model_type}")
        
        # Initialize tokenizer for BERT models
        if self.ml_model_type == "bert" and TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
                log(f"Loaded tokenizer for {self.bert_model_name}")
            except Exception as e:
                log(f"Error loading tokenizer: {e}")
                self.tokenizer = None
        
        # Try to load saved model
        model_path = self.model_dir / f"{self.collection_name}_{self.ml_model_type}_model.pt"
        if model_path.exists():
            try:
                if self.ml_model_type == "bert":
                    self.model = BertSentimentClassifier(
                        bert_model_name=self.bert_model_name,
                        feature_dim=self.params["feature_dim"],
                        dropout_rate=self.params["dropout_rate"]
                    )
                elif self.ml_model_type == "neural":
                    self.model = SentimentClassifier(
                        input_dim=self.params["feature_dim"],
                        hidden_dims=self.params["hidden_dims"],
                        dropout_rate=self.params["dropout_rate"]
                    )
                else:  # hybrid
                    self.model = HybridSentimentClassifier(
                        input_dim=self.params["feature_dim"],
                        hidden_dims=self.params["hidden_dims"],
                        dropout_rate=self.params["dropout_rate"]
                    )
                
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                log(f"Loaded saved model from {model_path}")
            except Exception as e:
                log(f"Error loading model: {e}")
                self.model = None
    
    def _create_model(self) -> nn.Module:
        """Create a new ML model based on configuration."""
        if not self.use_ml or not TORCH_AVAILABLE:
            return None
            
        if self.ml_model_type == "bert" and TRANSFORMERS_AVAILABLE:
            model = BertSentimentClassifier(
                bert_model_name=self.bert_model_name,
                feature_dim=self.params["feature_dim"],
                dropout_rate=self.params["dropout_rate"]
            )
        elif self.ml_model_type == "neural":
            model = SentimentClassifier(
                input_dim=self.params["feature_dim"],
                hidden_dims=self.params["hidden_dims"],
                dropout_rate=self.params["dropout_rate"]
            )
        else:  # hybrid
            model = HybridSentimentClassifier(
                input_dim=self.params["feature_dim"],
                hidden_dims=self.params["hidden_dims"],
                dropout_rate=self.params["dropout_rate"]
            )
        
        return model.to(self.device)

    def _load_resources(self) -> None:
        """Load all required lexical resources with memory monitoring."""
        log("Loading lexical resources...")
        
        # Load resources with memory checks in between
        self.positive_words = self._load_positive_words()
        log_memory()
        
        self.negative_words = self._load_negative_words()
        log_memory()
        
        self.intensifiers = self._load_intensifiers()
        self.diminishers = self._load_diminishers()
        log_memory()
        
        self.negations = self._load_negations()
        self.valence_shifters = self._load_valence_shifters()
        log_memory()
        
        self.domain_terms = self._load_domain_terms()
        log_memory()
        
        self.emoji_sentiment = self._load_emoji_sentiment()
        log_memory()
        
        self.sentiment_mapping = self._load_sentiment_mapping()
        log_memory()
        
        # Load supplementary Arabic resources
        self.arabic_stopwords = self._load_arabic_stopwords()
        self.sentiment_phrases = self._load_sentiment_phrases()
        self.citizen_feedback_patterns = self._load_citizen_feedback_patterns()
        log_memory()
        
        log("Finished loading all resources")

    def _check_memory_usage(self) -> bool:
        """
        Check if memory usage is approaching limit.
        Returns True if memory usage is OK, False if it's too high.
        """
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        current_mb = mem_info.rss / (1024 * 1024)
        
        if current_mb > self.memory_limit_mb:
            log(f"WARNING: Memory usage ({current_mb:.1f} MB) exceeds limit ({self.memory_limit_mb} MB)")
            return False
        elif current_mb > (self.memory_limit_mb * 0.8):
            log(f"CAUTION: Memory usage ({current_mb:.1f} MB) approaching limit ({self.memory_limit_mb} MB)")
            
        return True
    
    # Resource loading methods remain mostly the same, but are optimized where possible
    def _load_positive_words(self) -> Dict[str, float]:
        """Load enhanced positive words lexicon."""
        # Basic Arabic positive words (expanded)
        basic_positive = {
            "جيد": 1.5,
            "رائع": 2.0,
            "ممتاز": 2.0,
            "جميل": 1.5,
            "رائعة": 2.0,
            "أحب": 1.5,
            "ممتازة": 2.0,
            "شكرا": 1.2,
            "سعيد": 1.5,
            "حلو": 1.3,
            "فرح": 1.5,
            "نجاح": 1.8,
            "مفيد": 1.3,
            "حب": 1.5,
            "أعجبني": 1.5,
            "إيجابي": 1.5,
            "ناجح": 1.8,
            "متطور": 1.4,
            "مبدع": 1.7,
            "مبتكر": 1.6,
            "فعال": 1.3,
            "قوي": 1.3,
            "صحيح": 1.1,
            "أفضل": 1.4,
            "شجاع": 1.3,
            "مدهش": 1.8,
            "ذكي": 1.4,
            "حكيم": 1.4,
            "مرن": 1.2,
            "سريع": 1.2,
            "مهم": 1.1,
            "حماس": 1.3,
            "منظم": 1.2,
            "طموح": 1.3,
            "ثقة": 1.2,
            "تعاون": 1.3,
            "تقدم": 1.3,
            "تطوير": 1.3,
            "تحسين": 1.3,
            "مرحب": 1.2,
            "ممكن": 1.0,
            "واضح": 1.1,
            "مميز": 1.5,
            "أمل": 1.2,
            "معجب": 1.4,
            "سلس": 1.2,
            "نظيف": 1.2,
            "معتدل": 1.0,
            "متوازن": 1.1,
            "آمن": 1.2,
            "محبوب": 1.4,
            "شامل": 1.1,
            "حر": 1.2,
        }
        
        # Moroccan dialect positive words
        moroccan_positive = {
            "مزيان": 1.5,       # good
            "زوين": 1.5,        # nice/beautiful
            "بخير": 1.3,        # fine
            "عاجبني": 1.5,      # I like it
            "مبروك": 1.4,       # congratulations
            "واعر": 1.8,        # awesome
            "زوينة": 1.5,       # beautiful (feminine)
            "بنين": 1.4,        # tasty/good
            "مضبوط": 1.3,       # correct/proper
            "ديما": 1.1,        # always
            "بسلامة": 1.2,      # safely
            "لباس": 1.3,        # good/fine
            "عافاك": 1.2,       # please/thank you
            "شكرا بزاف": 1.4,   # thank you very much
            "ما شاء الله": 1.5, # wonderful/blessed
            "دغيا": 1.2,        # quickly
            "بسرعة": 1.2,       # quickly
            "نيشان": 1.3,       # straight/correct
            "قزديتي": 1.3,      # you succeeded/got it
            "حاية": 1.3,        # lively/energetic
            "فرحان": 1.5,       # happy
            "مبسوط": 1.5,       # pleased
            "مهني": 1.4,        # congratulations
            "بارك الله": 1.5,   # bless you
            "تبارك الله": 1.5,  # God bless
            "بارك الله فيك": 1.5, # God bless you
            "حق": 1.3,          # right/correct
            "شطارة": 1.4,       # cleverness/smartness
            "حكمة": 1.3,        # wisdom
            "فهامة": 1.3,       # understanding
            "ناجح": 1.7,        # successful
            "مقبول": 1.2,       # acceptable
            "موافق": 1.3,       # agree
            "نوافق": 1.3,       # I agree
            "متافق": 1.3,       # agreed
            "عجبني": 1.5,       # I liked it
            "كنبغي": 1.5,       # I love/like
            "كيعجبني": 1.5,     # I like it
            "فابور": 1.2,       # favor
            "معقول": 1.2,       # reasonable
            "مبروك عليك": 1.5,  # congratulations to you
        }
        
        # English positive words (commonly used in Arabic text)
        english_positive = {
            "good": 1.5,
            "great": 1.8,
            "excellent": 2.0,
            "amazing": 1.9,
            "perfect": 2.0,
            "wonderful": 1.9,
            "best": 1.8,
            "like": 1.2,
            "love": 1.7,
            "happy": 1.5,
            "thanks": 1.2,
            "thank": 1.2,
            "nice": 1.3,
            "awesome": 1.9,
            "fantastic": 1.9,
            "super": 1.7,
            "brilliant": 1.8,
            "cool": 1.4,
            "ok": 1.0,
            "okay": 1.0,
            "fine": 1.1,
            "right": 1.1,
            "correct": 1.2,
            "well": 1.2,
            "better": 1.3,
            "improved": 1.3,
            "helpful": 1.3,
            "useful": 1.3,
            "positive": 1.3,
            "success": 1.6,
            "successful": 1.6,
        }
        
        # French positive words (common in Morocco)
        french_positive = {
            "bien": 1.3,
            "bon": 1.3,
            "excellente": 1.8,
            "magnifique": 1.7,
            "parfait": 1.8,
            "génial": 1.7,
            "super": 1.5,
            "formidable": 1.6,
            "agréable": 1.3,
            "sympa": 1.3,
            "gentil": 1.2,
            "merci": 1.2,
            "bravo": 1.4,
            "félicitations": 1.5,
        }
        
        # Combine all lexicons
        combined = {}
        combined.update(basic_positive)
        combined.update(moroccan_positive)
        combined.update(english_positive)
        combined.update(french_positive)
        
        # Apply domain-specific boost to all terms if configured
        if self.params["arabic_boost"] > 1.0:
            for word in list(basic_positive.keys()) + list(moroccan_positive.keys()):
                if word in combined:
                    combined[word] *= self.params["arabic_boost"]
        
        log(f"Loaded {len(combined)} positive words")
        return combined
    
    def _load_negative_words(self) -> Dict[str, float]:
        """Load enhanced negative words lexicon."""
        # Basic Arabic negative words
        basic_negative = {
            "سيء": -1.5,
            "سيئة": -1.5,
            "مشكلة": -1.3,
            "خطأ": -1.4,
            "حزين": -1.4,
            "ضعيف": -1.3,
            "فشل": -1.8,
            "سوء": -1.4,
            "ضعيفة": -1.3,
            "قبيح": -1.5,
            "قبيحة": -1.5,
            "غاضب": -1.5,
            "غضب": -1.5,
            "مشكل": -1.3,
            "صعب": -1.2,
            "مؤسف": -1.3,
            "بطيء": -1.2,
            "سلبي": -1.5,
            "خطير": -1.4,
            "فاشل": -1.7,
            "كارثة": -2.0,
            "مزعج": -1.4,
            "محبط": -1.5,
            "مخيب": -1.5,
            "مقلق": -1.3,
            "مخيف": -1.4,
            "مرعب": -1.6,
            "قاسي": -1.4,
            "ظالم": -1.5,
            "غير عادل": -1.4,
            "متأخر": -1.2,
            "مكلف": -1.2,
            "غالي": -1.1,
            "رديء": -1.5,
            "مهمل": -1.4,
            "غير منظم": -1.3,
            "غير مهني": -1.4,
            "فوضوي": -1.3,
            "سخيف": -1.3,
            "تافه": -1.2,
            "عديم الفائدة": -1.5,
            "مستحيل": -1.3,
            "صراع": -1.3,
            "نزاع": -1.3,
            "عقبة": -1.2,
            "عائق": -1.2,
            "مخالفة": -1.3,
            "تأخير": -1.2,
            "خسارة": -1.4,
            "ضرر": -1.4,
            "تلف": -1.3,
            "خلل": -1.3,
            "عطل": -1.3,
            "ضياع": -1.3,
            "انقطاع": -1.2,
            "فساد": -1.7,
            "رشوة": -1.7,
            "تلاعب": -1.5,
            "غش": -1.6,
        }
        
        # Moroccan dialect negative words
        moroccan_negative = {
            "خايب": -1.5,       # bad
            "ماشي مزيان": -1.4, # not good
            "مشي زوين": -1.4,   # not nice
            "غالي": -1.2,       # expensive
            "واعر": -1.6,       # tough/difficult (context dependent)
            "بزاف": -1.1,       # too much
            "بطيء": -1.2,       # slow
            "ماكاينش": -1.1,    # there isn't
            "ماكاين والو": -1.3,# there's nothing
            "ما عندي ما ندير": -1.2, # I can't do anything
            "ما حملنيش": -1.4,   # I can't stand it
            "مخاصرة": -1.3,     # corruption
            "الرشوة": -1.6,     # bribery
            "مكرفص": -1.4,      # messed up
            "متلف": -1.4,       # damaged
            "مقود": -1.5,       # ruined
            "مسكين": -1.2,      # poor/miserable
            "معفن": -1.5,       # rotten
            "خراب": -1.5,       # ruins
            "مشوم": -1.4,       # unlucky
            "ماشي هو هداك": -1.2, # not the right one
            "ما صالحش": -1.4,    # not good/useful
            "حشومة": -1.3,       # shame
            "عيب": -1.3,         # shameful/disgrace
            "مقرف": -1.5,        # disgusting
            "مقزز": -1.5,        # disgusting
            "مبهدل": -1.4,       # messed up
            "مفرس": -1.4,        # ruined
            "مضروب": -1.3,       # hit/broken
            "متكسر": -1.4,       # broken
            "محروق": -1.4,       # burnt
            "مقطوع": -1.3,       # cut off
            "مسكوت": -1.2,       # shut up
            "مزلوط": -1.3,       # poor
            "مفلس": -1.4,        # broke/bankrupt
            "معور": -1.4,        # hurt
            "مريض": -1.3,        # sick
            "عيان": -1.3,        # sick/tired
            "تعبان": -1.3,       # tired
            "مخنوق": -1.4,       # suffocated
            "معسر": -1.3,        # difficult
            "صعيب": -1.3,        # difficult
            "قاسح": -1.3,        # hard
            "باهظ": -1.3,        # expensive
            "غالي بزاف": -1.4,   # very expensive
            "ما بقاش": -1.3,     # no more
            "كارثة": -1.7,       # disaster
            "مصيبة": -1.6,       # calamity
            "بلية": -1.5,        # affliction
            "مشكيلة": -1.3,      # problem
        }
        
        # English negative words (commonly used in Arabic text)
        english_negative = {
            "bad": -1.5,
            "poor": -1.4,
            "terrible": -1.8,
            "awful": -1.7,
            "horrible": -1.8,
            "worst": -1.9,
            "problem": -1.3,
            "issue": -1.2,
            "error": -1.3,
            "mistake": -1.3,
            "fail": -1.6,
            "failed": -1.6,
            "failure": -1.6,
            "wrong": -1.4,
            "broken": -1.4,
            "damaged": -1.4,
            "slow": -1.2,
            "expensive": -1.2,
            "difficult": -1.1,
            "hard": -1.1,
            "tough": -1.2,
            "sad": -1.3,
            "angry": -1.4,
            "upset": -1.3,
            "disappointed": -1.4,
            "frustrating": -1.4,
            "useless": -1.5,
            "waste": -1.4,
            "corrupt": -1.6,
            "corruption": -1.6,
            "bribe": -1.6,
            "bribery": -1.6,
            "dirty": -1.3,
            "filthy": -1.4,
            "disgusting": -1.5,
            "unsafe": -1.4,
            "dangerous": -1.4,
        }
        
        # French negative words (common in Morocco)
        french_negative = {
            "mauvais": -1.3,
            "horrible": -1.6,
            "terrible": -1.6,
            "nul": -1.3,
            "affreux": -1.4,
            "dégoûtant": -1.4,
            "désolé": -1.2,
            "dommage": -1.2,
            "problème": -1.2,
            "erreur": -1.3,
            "faute": -1.3,
            "échec": -1.5,
            "pire": -1.5,
            "faux": -1.3,
            "difficile": -1.2,
            "compliqué": -1.2,
            "pénible": -1.3,
            "ennuyeux": -1.2,
            "cher": -1.2,
        }
        
        # Combine all lexicons
        combined = {}
        combined.update(basic_negative)
        combined.update(moroccan_negative)
        combined.update(english_negative)
        combined.update(french_negative)
        
        # Apply domain-specific boost to all terms if configured
        if self.params["arabic_boost"] > 1.0:
            for word in list(basic_negative.keys()) + list(moroccan_negative.keys()):
                if word in combined:
                    combined[word] *= self.params["arabic_boost"]
        
        log(f"Loaded {len(combined)} negative words")
        return combined
    
    def _load_intensifiers(self) -> Dict[str, float]:
        """Load words that intensify sentiment."""
        intensifiers = {
            # Arabic intensifiers
            "جدا": 2.0,
            "كثيرا": 1.8,
            "للغاية": 2.0,
            "تماما": 1.7,
            "بشدة": 1.9,
            "فائق": 1.8,
            "قوي": 1.7,
            "حقا": 1.6,
            "بكثير": 1.7,
            "أكثر": 1.5,
            "فعلا": 1.6,
            "بالتأكيد": 1.6,
            "ممتاز": 1.8,
            "بزاف": 1.8,
            "واجد": 1.7,
            "بالزاف": 1.8,
            "هلبا": 1.8,
            "خالص": 1.9,
            "مرة": 1.6,
            "كتير": 1.8,
            
            # English intensifiers
            "very": 1.8,
            "extremely": 2.0,
            "really": 1.7,
            "so": 1.6,
            "too": 1.5,
            "absolutely": 1.9,
            "completely": 1.8,
            "totally": 1.8,
            "entirely": 1.8,
            "quite": 1.4,
            "rather": 1.3,
            "indeed": 1.5,
            
            # French intensifiers
            "très": 1.8,
            "extrêmement": 2.0,
            "vraiment": 1.7,
            "complètement": 1.8,
            "totalement": 1.8,
            "absolument": 1.9,
            "parfaitement": 1.7,
        }
        
        log(f"Loaded {len(intensifiers)} intensifiers")
        return intensifiers
    
    def _load_diminishers(self) -> Dict[str, float]:
        """Load words that diminish sentiment."""
        diminishers = {
            # Arabic diminishers
            "قليلا": 0.5,
            "نوعا ما": 0.6,
            "إلى حد ما": 0.7,
            "جزئيا": 0.6,
            "بعض": 0.5,
            "تقريبا": 0.7,
            "ربما": 0.6,
            "محتمل": 0.6,
            "ممكن": 0.7,
            "أحيانا": 0.5,
            "شوية": 0.5,
            "شويا": 0.5,
            "دغيا": 0.6,
            "غير": 0.6,
            "بس": 0.6,
            
            # English diminishers
            "slightly": 0.5,
            "somewhat": 0.6,
            "a bit": 0.5,
            "a little": 0.5,
            "kind of": 0.6,
            "sort of": 0.6,
            "barely": 0.3,
            "hardly": 0.4,
            "scarcely": 0.3,
            "somewhat": 0.6,
            "rather": 0.7,
            
            # French diminishers
            "un peu": 0.5,
            "légèrement": 0.5,
            "quelque peu": 0.6,
            "assez": 0.7,
            "plutôt": 0.7,
        }
        
        log(f"Loaded {len(diminishers)} diminishers")
        return diminishers
    
    def _load_negations(self) -> List[str]:
        """Load negation words that flip sentiment."""
        negations = [
            # Arabic negations
            "لا", "لم", "لن", "ليس", "ليست", "لسنا", "لستم", "لسن", "لست", "لسن", 
            "ما", "غير", "بلا", "بدون", "ما كاين", "ماكاين", "ماكاينش", "مكاينش",
            "مش", "مشي", "ماشي", "مو", "موش", "ماهو", "ماهي", "لن", "لم", "لا", 
            "عدم", "غياب", "نفي", "إنكار", "رفض", "نقض", "دون", "بلا", "خلاف", 
            "ضد", "عكس", "مخالف",
            
            # English negations
            "not", "no", "never", "without", "none", "neither", "nor", "nothing",
            "nobody", "nowhere", "barely", "hardly", "scarcely", "seldom", "rarely",
            "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot",
            "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
            
            # French negations
            "ne", "pas", "non", "sans", "jamais", "aucun", "aucune", "ni", "personne", "rien",
        ]
        
        log(f"Loaded {len(negations)} negations")
        return negations
    
    def _load_valence_shifters(self) -> Dict[str, float]:
        """Load valence shifters that modify sentiment direction or intensity."""
        shifters = {
            # Neutralizers
            "متوسط": 0.0,        # average
            "عادي": 0.0,         # normal
            "اعتيادي": 0.0,      # ordinary
            "مألوف": 0.0,        # familiar
            "تقليدي": 0.0,       # traditional
            "معتاد": 0.0,        # usual
            "قياسي": 0.0,        # standard
            "معياري": 0.0,       # normative
            "وسطي": 0.0,         # middle
            "متعارف": 0.0,       # conventional
            "average": 0.0,
            "normal": 0.0,
            "ordinary": 0.0,
            "standard": 0.0,
            "typical": 0.0,
            "usual": 0.0,
            "common": 0.0,
            "moderate": 0.0,
            "regular": 0.0,
            "routine": 0.0,
            "conventional": 0.0,
            "neither": 0.0,
            "nor": 0.0,
            "moyen": 0.0,        # French: average
            "normal": 0.0,       # French: normal
            "ordinaire": 0.0,    # French: ordinary
            "standard": 0.0,     # French: standard
            "typique": 0.0,      # French: typical
            "habituel": 0.0,     # French: usual
            
            # Sentiment modifiers
            "كان": 0.8,           # was (reduces intensity of sentiment)
            "سيكون": 0.7,         # will be (uncertainty reduces intensity)
            "قد": 0.7,            # may/might (uncertainty)
            "ربما": 0.6,          # perhaps
            "يمكن": 0.7,          # possibly
            "احتمال": 0.7,        # probability
            "محتمل": 0.7,         # probable
            "ممكن": 0.7,          # possible
            "متوقع": 0.8,         # expected
            "مفترض": 0.8,         # assumed
            "مزعوم": 0.5,         # alleged
            "مدعى": 0.5,          # claimed
            "يقال": 0.5,          # it is said
            "might": 0.7,
            "could": 0.7,
            "would": 0.7,
            "perhaps": 0.6,
            "possibly": 0.7,
            "probably": 0.8,
        }
        
        log(f"Loaded {len(shifters)} valence shifters")
        return shifters
    
    def _load_domain_terms(self) -> Dict[str, float]:
        """Load domain-specific terms relevant to citizen feedback and municipal projects."""
        domain_terms = {
            # Public services (neutral)
            "خدمة": 0.0,         # service 
            "خدمات": 0.0,        # services
            "عامة": 0.0,         # public
            "عمومية": 0.0,       # public
            "حكومة": 0.0,        # government
            "حكومية": 0.0,       # governmental
            "بلدية": 0.0,        # municipality
            "بلدي": 0.0,         # municipal
            "مدينة": 0.0,        # city
            "مدن": 0.0,          # cities
            "قرية": 0.0,         # village
            "قرى": 0.0,          # villages
            "مواطن": 0.0,        # citizen
            "مواطنين": 0.0,      # citizens
            "مواطنون": 0.0,      # citizens
            
            # Positive domain terms
            "تطوير": 0.7,       # development
            "تنمية": 0.7,       # development
            "تحسين": 0.7,       # improvement
            "إصلاح": 0.7,       # reform
            "ترميم": 0.6,       # renovation
            "تجديد": 0.6,       # renewal
            "بناء": 0.5,        # building
            "تشييد": 0.5,       # construction
            "فعال": 0.8,        # effective
            "فعالة": 0.8,       # effective (feminine)
            "ناجح": 0.8,        # successful
            "ناجحة": 0.8,       # successful (feminine)
            "تعاون": 0.7,       # cooperation
            "شفافية": 0.8,      # transparency
            "مساءلة": 0.7,      # accountability
            "مشاركة": 0.7,      # participation
            
            # Negative domain terms
            "فساد": -1.2,       # corruption
            "رشوة": -1.2,       # bribery
            "تأخير": -0.8,      # delay
            "تأخر": -0.8,       # delayed
            "بطء": -0.8,        # slowness
            "بطيء": -0.8,       # slow
            "بطيئة": -0.8,      # slow (feminine)
            "انقطاع": -0.9,     # outage/interruption
            "تعطل": -0.9,       # malfunction
            "عطل": -0.9,        # breakdown
            "مشكلة": -0.8,      # problem
            "مشاكل": -0.8,      # problems
            "سوء": -0.9,        # badness
            "سيء": -0.9,        # bad
            "سيئة": -0.9,       # bad (feminine)
            "رديء": -0.9,       # poor quality
            "رديئة": -0.9,      # poor quality (feminine)
            "تلوث": -0.9,       # pollution
            "ملوث": -0.9,       # polluted
            "ملوثة": -0.9,      # polluted (feminine)
            "قذارة": -0.9,      # filth
            "قذر": -0.9,        # dirty
            "قذرة": -0.9,       # dirty (feminine)
        }
        
        # Apply domain boost if configured
        if self.params["domain_boost"] > 1.0:
            for term, score in domain_terms.items():
                if score != 0:  # Only boost non-neutral terms
                    domain_terms[term] = score * self.params["domain_boost"]
        
        log(f"Loaded {len(domain_terms)} domain-specific terms")
        return domain_terms
    
    def _load_emoji_sentiment(self) -> Dict[str, float]:
        """Load emoji sentiment scores."""
        emojis = {
            "😀": 1.0, "😃": 1.0, "😄": 1.0, "😁": 1.0, "😆": 1.0,
            "😅": 0.7, "🤣": 0.8, "😂": 0.8, "🙂": 0.5, "🙃": 0.3,
            "😉": 0.6, "😊": 1.0, "😇": 0.9, "😍": 1.2, "🥰": 1.2,
            "😘": 1.1, "😗": 0.8, "😚": 0.9, "😙": 0.9, "😋": 0.9,
            "😛": 0.7, "😜": 0.6, "😝": 0.6, "🤑": 0.7, "🤗": 0.9,
            "🤭": 0.5, "🤫": 0.0, "🤔": -0.1, "🤐": -0.2, "🤨": -0.3,
            "😐": 0.0, "😑": -0.1, "😶": -0.1, "😏": 0.3, "😒": -0.7,
            "🙄": -0.5, "😬": -0.3, "🤥": -0.6, "😌": 0.5, "😔": -0.7,
            "😪": -0.4, "🤤": 0.2, "😴": 0.0, "😷": -0.5, "🤒": -0.8,
            "🤕": -0.8, "🤢": -1.0, "🤮": -1.2, "🤧": -0.6, "🥵": -0.5,
            "🥶": -0.5, "🥴": -0.6, "😵": -0.7, "🤯": -0.5, "🤠": 0.7,
            "🥳": 1.0, "😎": 0.8, "🤓": 0.4, "🧐": 0.2, "😕": -0.5,
            "😟": -0.7, "🙁": -0.6, "☹️": -0.8, "😮": 0.1, "😯": 0.0,
            "😲": 0.1, "😳": 0.0, "🥺": 0.3, "😦": -0.4, "😧": -0.5,
            "😨": -0.7, "😰": -0.7, "😥": -0.5, "😢": -0.8, "😭": -0.9,
            "😱": -0.8, "😖": -0.8, "😣": -0.7, "😞": -0.8, "😓": -0.6,
            "😩": -0.8, "😫": -0.9, "😤": -0.6, "😡": -1.0, "😠": -0.9,
            "🤬": -1.1, "😈": -0.5, "👿": -0.8, "💀": -0.7, "☠️": -0.7,
            "💩": -0.8, "🤡": -0.4, "👹": -0.6, "👺": -0.6, "👻": 0.3,
            
            # Basic emoticons (a subset of the original list to save memory)
            ":)": 0.8, ":-)": 0.8, ":D": 1.0, ":-D": 1.0, ";)": 0.6,
            ";-)": 0.6, ":(": -0.8, ":-(": -0.8, ":'(": -0.9, ":'-(": -0.9,
            ":/": -0.5, ":-/": -0.5, ":|": 0.0, ":-|": 0.0, ":*": 0.9,
            ":-*": 0.9, ":P": 0.7, ":-P": 0.7, ":p": 0.7, ":-p": 0.7,
            ">:(": -0.9, ">:-(": -0.9, "D:": -0.8, "D-:": -0.8, "XD": 0.9,
            "X-D": 0.9, "xD": 0.9, "x-D": 0.9, "^_^": 0.9, "^-^": 0.9
        }
        
        # Apply emoji weight if configured
        if self.params["emoji_weight"] != 1.0:
            for emoji, score in emojis.items():
                emojis[emoji] = score * self.params["emoji_weight"]
        
        log(f"Loaded {len(emojis)} emoji sentiment values")
        return emojis
    
    def _load_sentiment_mapping(self) -> Dict[str, str]:
        """Load mapping for standardizing sentiment labels."""
        return {
            # Positive variants
            "positive": "positive",
            "pos": "positive",
            "p": "positive",
            "1": "positive",
            1: "positive",
            "2": "positive",  # In some systems positive is 2
            2: "positive",
            "4": "positive",  # In 5-point scale, 4-5 are positive
            4: "positive",
            "5": "positive",
            5: "positive",
            
            # Negative variants
            "negative": "negative",
            "neg": "negative",
            "n": "negative",
            "-1": "negative",
            -1: "negative",
            "0": "negative",  # In some systems negative is 0
            0: "negative",
            "1/5": "negative",  # In 5-point scale, 1-2 are negative
            "2/5": "negative",
            
            # Neutral variants
            "neutral": "neutral",
            "neu": "neutral",
            "0": "neutral",  # In some systems neutral is 0
            "3": "neutral",  # In 5-point scale, 3 is neutral
            3: "neutral",
        }
    
    def _load_arabic_stopwords(self) -> List[str]:
        """Load Arabic stopwords."""
        stopwords = [
            "من", "إلى", "عن", "على", "في", "فوق", "تحت", "بين", "و", "أو", "ثم", "لكن",
            "ف", "ب", "ل", "ك", "و", "حتى", "إذا", "إلا", "أن", "لا", "ما", "هل", "كم",
            "كيف", "متى", "أين", "لماذا", "مع", "عند", "منذ", "قبل", "بعد", "خلال", "حول",
            "هو", "هي", "هم", "هن", "أنت", "أنتِ", "أنتم", "أنتن", "أنا", "نحن", "هذا", 
            "هذه", "هؤلاء", "ذلك", "تلك", "أولئك", "الذي", "التي", "الذين", "اللاتي",
            "اللواتي", "كل", "بعض", "غير", "كان", "يكون", "أصبح", "صار", "ليس", "مازال",
            "لازال", "مادام", "مابرح", "مافتئ", "ماانفك", "قد", "سوف", "سـ", "هنا", "هناك",
        ]
        return stopwords
    
    def _load_moroccan_dialect(self) -> Dict[str, float]:
        """Load Moroccan dialect terms with sentiment values."""
        # This is already covered by our positive_words and negative_words lexicons
        return {}
    
    def _load_sentiment_phrases(self) -> Dict[str, float]:
        """Load multi-word expressions with sentiment values."""
        phrases = {
            # Positive phrases
            "شكرا جزيلا": 1.5,         # Thank you very much
            "بارك الله فيك": 1.5,      # God bless you
            "ما شاء الله": 1.3,        # God has willed it (expression of praise)
            "جزاك الله خيرا": 1.4,     # May God reward you with goodness
            "تبارك الله": 1.3,         # Blessed is God (expression of amazement)
            "الله يبارك فيك": 1.4,     # May God bless you
            "عمل جيد": 1.3,            # Good work
            "عمل ممتاز": 1.5,          # Excellent work
            "عمل رائع": 1.5,           # Wonderful work
            "مشروع ناجح": 1.5,         # Successful project
            "مشروع مفيد": 1.3,         # Useful project
            "خدمة ممتازة": 1.5,        # Excellent service
            "خدمة جيدة": 1.3,          # Good service
            "في الوقت المناسب": 1.2,   # On time
            "بسرعة كبيرة": 1.2,        # With great speed
            "بدقة عالية": 1.3,         # With high precision
            "بجودة عالية": 1.4,        # With high quality
            "بكفاءة عالية": 1.4,       # With high efficiency
            
            # Negative phrases
            "سيء جدا": -1.5,           # Very bad
            "ردئ للغاية": -1.5,        # Extremely poor
            "غير مقبول": -1.3,         # Unacceptable
            "غير مرضي": -1.3,          # Unsatisfactory
            "لا يرقى للمستوى": -1.2,   # Below standard
            "دون المستوى": -1.2,       # Substandard
            "سيء للغاية": -1.5,        # Extremely bad
            "مخيب للآمال": -1.3,       # Disappointing
            "محبط جدا": -1.4,          # Very frustrating
            "أسوأ خدمة": -1.6,         # Worst service
            "أسوأ تجربة": -1.6,        # Worst experience
            "تجربة سيئة": -1.4,        # Bad experience
            "خدمة سيئة": -1.4,         # Bad service
            "فشل ذريع": -1.6,          # Utter failure
            "كارثة حقيقية": -1.7,      # Real disaster
            "مضيعة للوقت": -1.4,       # Waste of time
            "مضيعة للجهد": -1.4,       # Waste of effort
            "مضيعة للمال": -1.5,       # Waste of money
        }
        
        # Apply domain-specific boost if configured
        if self.params["domain_boost"] > 1.0:
            for phrase, score in phrases.items():
                phrases[phrase] = score * self.params["domain_boost"]
        
        log(f"Loaded {len(phrases)} sentiment phrases")
        return phrases
    
    def _load_citizen_feedback_patterns(self) -> List[Dict[str, Any]]:
        """Load patterns that are common in citizen feedback."""
        patterns = [
            # Positive feedback patterns
            {
                "pattern": r"شكر[اًا][\s]+(?:لك|لكم|للبلدية|للمسؤولين|للعاملين)",
                "sentiment": 1.5,
                "description": "Expressions of thanks to municipality/officials"
            },
            {
                "pattern": r"نشكر[كم]*[\s]+على[\s]+(?:الجهود|العمل|المجهود)",
                "sentiment": 1.4,
                "description": "Thanking for efforts"
            },
            {
                "pattern": r"(استفد[تن]|انتفع[تن]|حصل[تن])[\s]+من[\s]+(?:الخدمة|المشروع|البرنامج)",
                "sentiment": 1.3,
                "description": "Expressions of benefit from service/project"
            },
            
            # Negative feedback patterns
            {
                "pattern": r"(نعاني|أعاني|نواجه|يواجه)[\s]+من[\s]+(?:مشكلة|مشاكل|صعوبات|عقبات)",
                "sentiment": -1.3,
                "description": "Suffering from problems"
            },
            {
                "pattern": r"(لم|ما)[\s]+(?:يتم|نحصل|أحصل|نجد|أجد)[\s]+على[\s]+(?:الخدمة|المساعدة|الدعم|الرد)",
                "sentiment": -1.2,
                "description": "Not receiving service/support/response"
            },
            {
                "pattern": r"(?:منذ|لمدة)[\s]+(?:\d+|عدة|طويلة)[\s]+(?:أيام|أسابيع|شهور|سنوات)[\s]+(?:ونحن|وأنا|والمشكلة|والوضع)",
                "sentiment": -1.3,
                "description": "Problem persisting for long time"
            },
            
            # Request patterns (slightly negative)
            {
                "pattern": r"(?:نرجو|أرجو|نتمنى|أتمنى)[\s]+(?:من|أن)[\s]+(?:تنظروا|تهتموا|تعملوا|تحلوا)",
                "sentiment": -0.8,
                "description": "Politely requesting attention/action"
            },
            {
                "pattern": r"(?:الرجاء|برجاء|نرجو|أرجو)[\s]+(?:النظر|الاهتمام|المتابعة|التدخل)",
                "sentiment": -0.7,
                "description": "Requesting attention/follow-up"
            },
        ]
        
        log(f"Loaded {len(patterns)} citizen feedback patterns")
        return patterns
    
    def _normalize_sentiment_label(self, label):
        """Normalize sentiment labels to a standard format."""
        if label is None:
            return None
        
        # Convert label to string for consistency
        try:
            str_label = str(label).lower().strip()
        except:
            return None
        
        # Check direct match
        if str_label in self.sentiment_mapping:
            return self.sentiment_mapping[str_label]
            
        # Check for numerical values
        try:
            num_label = float(str_label)
            if num_label in self.sentiment_mapping:
                return self.sentiment_mapping[num_label]
        except (ValueError, TypeError):
            pass
            
        # Check for common substrings
        for key, standardized in [
            ("pos", "positive"),
            ("neg", "negative"),
            ("neut", "neutral")
        ]:
            if key in str_label:
                return standardized
                
        return str_label
    
    def _extract_emojis(self, text: str) -> List[str]:
        """Extract emojis and emoticons from text."""
        emojis = []
        
        # Unicode emoji pattern
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F700-\U0001F77F"  # alchemical symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251" 
            "]+", flags=re.UNICODE)
        
        # Find all emojis
        for match in emoji_pattern.finditer(text):
            emojis.append(match.group(0))
        
        # Find emoticons
        emoticon_pattern = r'(?::|;|=)(?:-)?(?:\)|\(|D|P|S)'
        emoticons = re.findall(emoticon_pattern, text)
        emojis.extend(emoticons)
        
        return emojis
    
    def _find_phrases(self, text: str) -> List[Tuple[str, float]]:
        """Find sentiment phrases in text."""
        found_phrases = []
        
        # Check for exact phrases
        for phrase, score in self.sentiment_phrases.items():
            if phrase in text.lower():
                found_phrases.append((phrase, score))
        
        # Check for feedback patterns
        for pattern in self.citizen_feedback_patterns:
            matches = re.findall(pattern["pattern"], text)
            if matches:
                found_phrases.append((matches[0], pattern["sentiment"]))
        
        return found_phrases
    
    def _preprocess_text(self, text: str) -> Tuple[str, List[str]]:
        """Preprocess text for analysis."""
        if not text:
            return "", []
            
        # Convert to lowercase if it contains Latin characters
        if any(c.isascii() for c in text):
            text = text.lower()
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Extract emojis
        emojis = self._extract_emojis(text)
            
        return text, emojis
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform optimized sentiment analysis with comprehensive features
        and hyperparameter-tuned configuration.
        """
        if not text:
            return {"label": "neutral", "score": 0.5}
            
        # Preprocess text
        text, emojis = self._preprocess_text(text)
        
        # Find multi-word phrases
        phrases = self._find_phrases(text)
        
        # Split into words
        words = text.split()
        
        # Track positions of important words
        sentiment_word_positions = {}
        negation_positions = []
        intensifier_positions = {}
        diminisher_positions = {}
        
        # Detect negations
        for i, word in enumerate(words):
            if word in self.negations:
                # Mark this position and a few words after as negated
                window = range(i+1, min(i+1+self.params["negation_window"], len(words)))
                negation_positions.extend(window)
            
            # Track intensifiers
            if word in self.intensifiers:
                intensifier_positions[i] = self.intensifiers[word]
            
            # Track diminishers
            if word in self.diminishers:
                diminisher_positions[i] = self.diminishers[word]
            
            # Track sentiment words
            if word in self.positive_words:
                sentiment_word_positions[i] = (word, self.positive_words[word], "positive")
            elif word in self.negative_words:
                sentiment_word_positions[i] = (word, self.negative_words[word], "negative")
            elif word in self.domain_terms and self.domain_terms[word] != 0:
                if self.params["prioritize_domain_matches"]:
                    # Prioritize domain matches by removing any existing match at this position
                    sentiment_word_positions[i] = (word, self.domain_terms[word], 
                                               "positive" if self.domain_terms[word] > 0 else "negative")
        
        # Calculate sentiment scores with contextual rules
        pos_score = 0.0
        neg_score = 0.0
        pos_words = []
        neg_words = []
        
        # Process emojis first (they're not affected by negation)
        emoji_sentiment = 0.0
        for emoji in emojis:
            if emoji in self.emoji_sentiment:
                score = self.emoji_sentiment[emoji]
                emoji_sentiment += score
                if score > 0:
                    pos_words.append(emoji)
                    pos_score += score
                elif score < 0:
                    neg_words.append(emoji)
                    neg_score += abs(score)
        
        # Process multi-word phrases (they get priority)
        phrase_sentiment = 0.0
        for phrase, score in phrases:
            phrase_sentiment += score
            if score > 0:
                pos_score += score
                pos_words.append(phrase)
            elif score < 0:
                neg_score += abs(score)
                neg_words.append(phrase)
        
        # Process individual words with context
        for i, (word, base_score, sentiment_type) in sentiment_word_positions.items():
            # Apply valence shifters if configured
            if self.params["use_contextual_valence_shifters"]:
                # Check for previous valence shifters
                for shifter_pos in range(max(0, i-3), i):
                    if shifter_pos in intensifier_positions:
                        base_score *= intensifier_positions[shifter_pos]
                    elif shifter_pos in diminisher_positions:
                        base_score *= diminisher_positions[shifter_pos]
                
                # Check for subsequent valence shifters
                for shifter_pos in range(i+1, min(i+3, len(words))):
                    if shifter_pos in intensifier_positions:
                        base_score *= intensifier_positions[shifter_pos]
                    elif shifter_pos in diminisher_positions:
                        base_score *= diminisher_positions[shifter_pos]
            
            # Apply negation if in a negation context
            if i in negation_positions:
                # Flip sentiment direction with optional boost
                base_score = -base_score * self.params["negation_factor"]
                # Negate also changes the sentiment type
                sentiment_type = "negative" if sentiment_type == "positive" else "positive"
            
            # Apply dynamic weighting if configured
            if self.params["apply_dynamic_weighting"]:
                # Words at beginning and end have slightly more impact
                position_factor = 1.0
                relative_pos = i / len(words)
                if relative_pos < 0.2 or relative_pos > 0.8:
                    position_factor = 1.1
                base_score *= position_factor
            
            # Add to appropriate sentiment score
            if sentiment_type == "positive" or base_score > 0:
                pos_score += abs(base_score)
                pos_words.append(word)
            else:
                neg_score += abs(base_score)
                neg_words.append(word)
        
        # Apply positive and negative biases to counter neutral dominance
        pos_score = pos_score * self.params["positive_bias"]
        neg_score = neg_score * self.params["negative_bias"]
        
        # Calculate total sentiment with class weighting
        if pos_score > neg_score:
            total_score = pos_score * self.params["positive_weight"] - neg_score
        elif neg_score > pos_score:
            total_score = pos_score - neg_score * self.params["negative_weight"]
        else:
            total_score = 0
        
        # Determine sentiment with adjusted thresholds
        # For aggressive classification, we use the neutral_threshold parameter
        if total_score > self.params["min_pos_threshold"]:
            sentiment = "positive"
            # Confidence based on score difference and evidence strength
            confidence = min(0.5 + (pos_score / (pos_score + neg_score + 0.1)) * 0.5, 0.99)
        elif total_score < self.params["neutral_threshold"]:
            sentiment = "negative"
            confidence = min(0.5 + (neg_score / (pos_score + neg_score + 0.1)) * 0.5, 0.99)
        else:
            # Only classify as neutral if not in aggressive mode or if truly neutral
            if not self.params["aggressive_classification"]:
                sentiment = "neutral"
                # Lower confidence for neutral classifications
                confidence = 0.5 + abs(total_score) * 0.5
            else:
                # In aggressive mode, lean toward the stronger sentiment even if small
                if pos_score >= neg_score:
                    sentiment = "positive"
                    confidence = 0.5 + 0.1 * (pos_score / (pos_score + neg_score + 0.1))
                else:
                    sentiment = "negative"
                    confidence = 0.5 + 0.1 * (neg_score / (pos_score + neg_score + 0.1))
        
        # Check if we should use ML model for prediction
        if self.use_ml and self.model:
            try:
                # Extract features for ML model
                features = [
                    pos_score,
                    neg_score,
                    total_score,
                    emoji_sentiment,
                    phrase_sentiment,
                    float(len(pos_words)),
                    float(len(neg_words)),
                    float(len(emojis)),
                    float(len(phrases)),
                    confidence
                ]
                
                # Convert to tensor
                feature_tensor = torch.tensor([features], dtype=torch.float).to(self.device)
                
                # Get model prediction
                self.model.eval()
                with torch.no_grad():
                    if self.ml_model_type == "bert" and self.tokenizer:
                        # Process with BERT model
                        encoding = self.tokenizer(
                            text,
                            return_tensors='pt',
                            max_length=128,
                            padding='max_length',
                            truncation=True
                        ).to(self.device)
                        
                        outputs = self.model(
                            input_ids=encoding['input_ids'],
                            attention_mask=encoding['attention_mask'],
                            features=feature_tensor
                        )
                    else:
                        # Process with neural network
                        outputs = self.model(feature_tensor)
                
                # Get predicted class
                _, predicted = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                # Map numeric prediction to sentiment label
                label_map = {0: "positive", 1: "negative", 2: "neutral"}
                ml_sentiment = label_map[predicted.item()]
                ml_confidence = probs[0][predicted.item()].item()
                
                # Use ML prediction with rule-based fallback
                sentiment = ml_sentiment
                confidence = ml_confidence
                
                # Add ML info to result
                ml_info = {
                    "ml_sentiment": ml_sentiment,
                    "ml_confidence": ml_confidence,
                    "rule_sentiment": sentiment,
                    "rule_confidence": confidence
                }
            except Exception as e:
                log(f"Error using ML model for prediction: {e}")
                ml_info = {"error": str(e)}
        else:
            ml_info = None
        
        # Prepare detailed result
        result = {
            "label": sentiment,
            "score": confidence,
            "pos_score": pos_score,
            "neg_score": neg_score,
            "total_score": total_score,
            # Only include top words to save memory
            "pos_words": pos_words[:5],  
            "neg_words": neg_words[:5],
            "emoji_sentiment": emoji_sentiment,
            "phrase_sentiment": phrase_sentiment,
            "emojis": emojis[:3],  # Limit emojis
            "phrases": [p[0] for p in phrases][:3]  # Limit phrases
        }
        
        # Add ML info if available
        if ml_info:
            result["ml_info"] = ml_info
        
        return result
    
    def _batch_generator(self, data: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
        """Generate batches from a list."""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    
    def load_data(self, limit: Optional[int] = None) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Load data from Qdrant collection in batches.
        Returns a generator that yields batches of records.
        """
        log(f"Loading data from collection: {self.collection_name}")
        
        # Define batch handler for caching
        cache_base = self.cache_dir / f"{self.collection_name}_records"
        batch_handler = BatchFileHandler(cache_base, batch_size=self.batch_size)
        
        # Check if we already have batches cached
        has_cached_batches = False
        try:
            first_batch = next(batch_handler.read_batches(), None)
            if first_batch:
                has_cached_batches = True
                log("Found cached batches")
        except:
            has_cached_batches = False
        
        # If we have cached batches, return them
        if has_cached_batches:
            log("Reading from cached batches")
            count = 0
            for batch in batch_handler.read_batches():
                if limit and count >= limit:
                    break
                    
                # If limit is within this batch, slice it
                if limit and count + len(batch) > limit:
                    yield batch[:limit - count]
                    count = limit
                else:
                    yield batch
                    count += len(batch)
                
                log(f"Yielded batch, total records so far: {count}")
                
            log(f"Finished reading {count} records from cache")
            return
        
        # If using local data, load from file
        if self.use_local_data:
            if not self.local_data_path or not Path(self.local_data_path).exists():
                log(f"Error: Local data path not found: {self.local_data_path}")
                yield []
                return
                
            try:
                log(f"Loading data from local file: {self.local_data_path}")
                # Load data from file
                with open(self.local_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # If it's a dictionary with a 'records' key, extract the records
                if isinstance(data, dict) and 'records' in data:
                    records = data['records']
                # If it's a list, use it directly
                elif isinstance(data, list):
                    records = data
                else:
                    log(f"Error: Unexpected data format in {self.local_data_path}")
                    yield []
                    return
                
                # Apply limit if needed
                if limit and limit < len(records):
                    records = records[:limit]
                
                # Process in batches
                total_loaded = 0
                for i in range(0, len(records), self.batch_size):
                    batch = records[i:min(i+self.batch_size, len(records))]
                    
                    # Process batch to ensure correct format
                    processed_batch = []
                    for item in batch:
                        # Get text and ground truth
                        text = item.get('text')
                        if not text and 'content' in item:
                            text = item.get('content')
                        
                        if not text:
                            continue
                            
                        # Look for ground truth
                        ground_truth = item.get('sentiment') or item.get('label')
                        
                        # Normalize ground truth
                        if ground_truth is not None:
                            ground_truth = self._normalize_sentiment_label(ground_truth)
                        
                        # Create record
                        record = {
                            'id': str(item.get('id', f"local_{i}")),
                            'text': text,
                            'ground_truth': ground_truth
                        }
                        
                        processed_batch.append(record)
                        
                        # Add to batch handler for caching
                        batch_handler.add_item(record)
                    
                    # Update loaded count
                    total_loaded += len(processed_batch)
                    log(f"Loaded {total_loaded} local records")
                    
                    # Yield this batch
                    yield processed_batch
                
                # Make sure last batch is flushed
                batch_handler.flush()
                log(f"Completed loading {total_loaded} local records")
                return
                
            except Exception as e:
                log(f"Error loading local data: {e}")
                log(f"Traceback: {traceback.format_exc()}")
                yield []
                return
                
        # If no cache or error, and not using local data, load from Qdrant
        if not self.client:
            log("No Qdrant client available")
            yield []
            return
            
        try:
            # Get collection info
            try:
                collection_info = self.client.get_collection(self.collection_name)
                total_count = self.client.count(self.collection_name).count
                log(f"Found {total_count} records in collection")
            except Exception as e:
                log(f"Error getting collection info: {e}")
                yield []
                return
            
            # Apply limit
            if limit and limit < total_count:
                target_count = limit
            else:
                target_count = total_count
            
            # Load records in batches
            offset = None
            loaded = 0
            
            log(f"Starting to load {target_count} records in batches of {self.batch_size}")
            
            # Batch loading loop
            while loaded < target_count:
                batch_limit = min(self.batch_size, target_count - loaded)
                
                # Check memory before loading more
                if not self._check_memory_usage():
                    log("Memory usage too high, forcing garbage collection")
                    gc.collect()
                
                # Scroll through records
                try:
                    result = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=batch_limit,
                        offset=offset
                    )
                    
                    if isinstance(result, tuple):
                        points, offset = result
                    else:
                        points = result.points
                        offset = result.next_page_offset
                except Exception as e:
                    log(f"Error scrolling collection: {e}")
                    break
                
                if not points:
                    break
                    
                # Process points
                batch_records = []
                for point in points:
                    # Extract data
                    if hasattr(point, 'payload'):
                        payload = point.payload
                        point_id = point.id
                    else:
                        payload = point.get('payload', {})
                        point_id = point.get('id')
                    
                    # Get text and ground truth
                    text = payload.get('text')
                    if not text:
                        continue
                        
                    # Look for ground truth in various locations
                    ground_truth = None
                    metadata = payload.get('metadata', {})
                    
                    # Direct sentiment field
                    if 'sentiment' in metadata:
                        ground_truth = metadata['sentiment']
                    
                    # Check raw payload
                    raw_payload = payload.get('raw_payload', {})
                    if not ground_truth and 'sentiment' in raw_payload:
                        ground_truth = raw_payload['sentiment']
                    
                    # Check other fields
                    for field in ['sentiment_score', 'rating', 'score', 'label']:
                        if not ground_truth and field in metadata:
                            ground_truth = metadata[field]
                        if not ground_truth and field in raw_payload:
                            ground_truth = raw_payload[field]
                    
                    # Normalize ground truth
                    if ground_truth is not None:
                        ground_truth = self._normalize_sentiment_label(ground_truth)
                    
                    # Create minimal record to save memory
                    record = {
                        'id': str(point_id),
                        'text': text,
                        'ground_truth': ground_truth
                    }
                    
                    batch_records.append(record)
                
                # Add to batch handler for caching
                for record in batch_records:
                    batch_handler.add_item(record)
                
                # Update loaded count
                batch_loaded = len(batch_records)
                loaded += batch_loaded
                log(f"Loaded {loaded}/{target_count} records")
                
                # Yield this batch
                yield batch_records
                
                # Check if we're done
                if offset is None or loaded >= target_count:
                    break
            
            # Make sure last batch is flushed
            batch_handler.flush()
            log(f"Completed loading {loaded} records")
            
        except Exception as e:
            log(f"Error loading data from Qdrant: {e}")
            log(f"Traceback: {traceback.format_exc()}")
            yield []
    
    def analyze_records_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of records."""
        results = []
        
        for record in records:
            # Check memory periodically
            if len(results) % 100 == 0:
                self._check_memory_usage()
            
            # Analyze text
            sentiment_result = self.analyze_text(record['text'])
            
            # Create result record (minimal version to save memory)
            result = {
                'id': record.get('id', ''),
                'predicted_sentiment': sentiment_result['label'],
                'sentiment_score': sentiment_result['score'],
            }
            
            # Add comparison with ground truth
            if record.get('ground_truth') is not None:
                result['ground_truth'] = record['ground_truth']
                result['correct'] = record['ground_truth'] == result['predicted_sentiment']
            
            results.append(result)
        
        return results
    
    def analyze_records(self, limit: Optional[int] = None) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Analyze sentiment for all records in batches.
        Returns a generator that yields batches of analyzed records.
        """
        log(f"Starting sentiment analysis")
        
        # Check for cached results
        results_base = self.cache_dir / f"{self.collection_name}_analysis"
        results_handler = BatchFileHandler(results_base, batch_size=self.batch_size)
        
        # Check if we already have results cached
        has_cached_results = False
        try:
            first_result_batch = next(results_handler.read_batches(), None)
            if first_result_batch:
                has_cached_results = True
                log("Found cached analysis results")
        except:
            has_cached_results = False
        
        # If we have cached results, return them
        if has_cached_results:
            log("Reading from cached analysis results")
            count = 0
            for batch in results_handler.read_batches():
                if limit and count >= limit:
                    break
                    
                # If limit is within this batch, slice it
                if limit and count + len(batch) > limit:
                    yield batch[:limit - count]
                    count = limit
                else:
                    yield batch
                    count += len(batch)
                
                log(f"Yielded result batch, total results so far: {count}")
                
            log(f"Finished reading {count} analysis results from cache")
            return
        
        # Process records in batches
        processed_count = 0
        record_batches = self.load_data(limit=limit)
        
        for i, record_batch in enumerate(record_batches):
            if not record_batch:
                continue
                
            log(f"Processing batch {i+1} with {len(record_batch)} records")
            log_memory()
            
            # Analyze this batch
            results_batch = self.analyze_records_batch(record_batch)
            
            # Save results to cache
            for result in results_batch:
                results_handler.add_item(result)
            
            processed_count += len(results_batch)
            log(f"Processed {processed_count} records so far")
            
            # Yield this batch of results
            yield results_batch
            
            # Force garbage collection
            del record_batch
            gc.collect()
        
        # Make sure last batch is flushed
        results_handler.flush()
        log(f"Completed analysis of {processed_count} records")
    
    def compute_metrics(self, result_batches: Union[Generator[List[Dict[str, Any]], None, None], List[List[Dict[str, Any]]]]) -> Dict[str, Any]:
        """Compute metrics from analysis results incrementally."""
        log("Computing metrics")
        
        # Initialize counters
        prediction_counts = Counter()
        ground_truth_counts = Counter()
        correct_count = 0
        total_count = 0
        confusion = defaultdict(lambda: defaultdict(int))
        
        # Initialize class metrics containers
        labels = set(["positive", "negative", "neutral"])  # Start with common labels
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)
        
        # Process batches
        for batch in result_batches:
            for result in batch:
                # Update total
                total_count += 1
                
                # Update prediction counts
                sentiment = result.get('predicted_sentiment')
                if sentiment:
                    prediction_counts[sentiment] += 1
                
                # Process ground truth if available
                ground_truth = result.get('ground_truth')
                if ground_truth is not None:
                    # Update ground truth counts
                    ground_truth_counts[ground_truth] += 1
                    
                    # Update labels set
                    labels.add(ground_truth)
                    if sentiment:
                        labels.add(sentiment)
                    
                    # Update confusion matrix
                    confusion[ground_truth][sentiment] += 1
                    
                    # Update class metrics
                    if result.get('correct', False):
                        correct_count += 1
                        true_positives[ground_truth] += 1
                    else:
                        false_positives[sentiment] += 1
                        false_negatives[ground_truth] += 1
            
            # Log progress
            if total_count % 1000 == 0:
                log(f"Processed {total_count} results for metrics")
        
        # Calculate percentages
        percentages = {k: (v / max(total_count, 1)) * 100 for k, v in prediction_counts.items()}
        
        # Evaluation metrics
        eval_metrics = {
            "has_ground_truth": False,
            "accuracy": None,
            "confusion_matrix": None,
            "class_metrics": None
        }
        
        # Calculate ground truth metrics if available
        ground_truth_count = sum(ground_truth_counts.values())
        if ground_truth_count > 0:
            eval_metrics["has_ground_truth"] = True
            eval_metrics["ground_truth_count"] = ground_truth_count
            
            # Calculate accuracy
            eval_metrics["accuracy"] = correct_count / ground_truth_count if ground_truth_count > 0 else 0
            
            # Format confusion matrix
            eval_metrics["confusion_matrix"] = {
                true_label: {pred_label: confusion[true_label][pred_label] for pred_label in labels}
                for true_label in labels
            }
            eval_metrics["labels"] = sorted(list(labels))
            
            # Calculate class metrics
            class_metrics = {}
            for label in labels:
                # Calculate precision, recall, F1
                tp = true_positives[label]
                fp = false_positives[label]
                fn = false_negatives[label]
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics[label] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": ground_truth_counts.get(label, 0)
                }
            
            eval_metrics["class_metrics"] = class_metrics
            
            # Calculate ground truth distribution
            eval_metrics["ground_truth_distribution"] = {
                "counts": dict(ground_truth_counts),
                "percentages": {k: (v / ground_truth_count) * 100 for k, v in ground_truth_counts.items()}
            }
        
        return {
            "total_records": total_count,
            "sentiment_counts": dict(prediction_counts),
            "sentiment_percentages": percentages,
            "evaluation": eval_metrics
        }
    
    def save_metrics(self, metrics: Dict[str, Any], output_file: str) -> None:
        """Save metrics to file."""
        log(f"Saving metrics to {output_file}")
        
        # Save metrics
        metrics_file = f"{output_file}_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate report
        report_file = f"{output_file}_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SENTIMENT ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total records: {metrics['total_records']}\n\n")
            
            f.write("Sentiment distribution:\n")
            for sentiment, count in metrics['sentiment_counts'].items():
                percentage = metrics['sentiment_percentages'][sentiment]
                f.write(f"  - {sentiment}: {count} ({percentage:.2f}%)\n")
            f.write("\n")
            
            if metrics['evaluation']['has_ground_truth']:
                f.write(f"Records with ground truth: {metrics['evaluation']['ground_truth_count']}\n")
                f.write(f"Overall accuracy: {metrics['evaluation']['accuracy']:.4f} ({metrics['evaluation']['accuracy']*100:.2f}%)\n\n")
                
                f.write("Class metrics:\n")
                for label, metrics_dict in metrics['evaluation']['class_metrics'].items():
                    f.write(f"  {label}:\n")
                    f.write(f"    Precision: {metrics_dict['precision']:.4f}\n")
                    f.write(f"    Recall: {metrics_dict['recall']:.4f}\n")
                    f.write(f"    F1-score: {metrics_dict['f1']:.4f}\n")
                    f.write(f"    Support: {metrics_dict['support']}\n")
                f.write("\n")
                
                f.write("Ground truth distribution:\n")
                for label, count in metrics['evaluation']['ground_truth_distribution']['counts'].items():
                    percentage = metrics['evaluation']['ground_truth_distribution']['percentages'][label]
                    f.write(f"  - {label}: {count} ({percentage:.2f}%)\n")
                f.write("\n")
                
                f.write("Confusion Matrix:\n")
                labels = metrics['evaluation']['labels']
                matrix = metrics['evaluation']['confusion_matrix']
                
                # Header
                header = "True \\ Pred |"
                for label in labels:
                    header += f" {label} |"
                f.write(header + "\n")
                
                # Separator
                f.write("-" * len(header) + "\n")
                
                # Data rows
                for true_label in labels:
                    row = f"{true_label} |"
                    for pred_label in labels:
                        row += f" {matrix[true_label][pred_label]} |"
                    f.write(row + "\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        log(f"Saved metrics to {metrics_file} and {report_file}")
    
    def run_analysis(self, limit: Optional[int] = None, output_file: str = "sentiment_results") -> Dict[str, Any]:
        """Run the complete sentiment analysis pipeline with batching."""
        start_time = time.time()
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_file).parent
        if output_dir != Path('.'):
            output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Analyze records in batches
            result_batches = list(self.analyze_records(limit=limit))
            
            # Check if any results were generated
            if not result_batches or all(len(batch) == 0 for batch in result_batches):
                log("No records analyzed. Check connection to Qdrant or local data path.")
                return {
                    "error": "No records analyzed",
                    "results_count": 0,
                    "elapsed_time": time.time() - start_time
                }
            
            # Compute metrics from batches
            metrics = self.compute_metrics(result_batches)
            
            # Save metrics
            self.save_metrics(metrics, output_file)
            
            # Return summary
            end_time = time.time()
            elapsed = end_time - start_time
            
            return {
                "results_count": metrics["total_records"],
                "metrics": metrics,
                "elapsed_time": elapsed,
                "output_files": [
                    f"{output_file}_metrics.json",
                    f"{output_file}_report.txt"
                ]
            }
        except Exception as e:
            log(f"Error in run_analysis: {e}")
            log(f"Traceback: {traceback.format_exc()}")
            
            # Return error information
            return {
                "error": str(e),
                "results_count": 0,
                "elapsed_time": time.time() - start_time
            }

    def prepare_data_for_training(self, limit: Optional[int] = None, test_size: float = 0.2, val_size: float = 0.1, 
                                 random_state: int = 42) -> Tuple[Dict[str, List[Any]], Dict[str, int]]:
        """
        Prepare data for ML model training by loading data and splitting into train/val/test sets.
        
        Returns:
            Tuple containing:
            - Dictionary with 'train', 'val', 'test' splits, each containing lists of text and labels
            - Dictionary with dataset statistics
        """
        log("Preparing data for training...")
        
        # Load all data
        all_records = []
        for batch in self.load_data(limit=limit):
            all_records.extend(batch)
            
        log(f"Loaded {len(all_records)} records for training preparation")
        
        # Filter records that have ground truth
        valid_records = [r for r in all_records if r.get('ground_truth') is not None]
        log(f"Found {len(valid_records)} records with ground truth labels")
        
        if len(valid_records) == 0:
            log("No records with ground truth found. Cannot prepare for training.")
            return None, {"error": "No labeled data available"}
            
        # Extract text and labels
        texts = [r['text'] for r in valid_records]
        labels = [r['ground_truth'] for r in valid_records]
        
        # Split into train, validation, and test sets
        # First split into train+val and test
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Then split train+val into train and val
        # Recalculate val_size as a proportion of train_val
        adjusted_val_size = val_size / (1 - test_size)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels, test_size=adjusted_val_size, 
            random_state=random_state, stratify=train_val_labels
        )
        
        # Count class distribution
        train_counts = Counter(train_labels)
        val_counts = Counter(val_labels)
        test_counts = Counter(test_labels)
        all_counts = Counter(labels)
        
        # Prepare result
        data_splits = {
            'train': {'texts': train_texts, 'labels': train_labels},
            'val': {'texts': val_texts, 'labels': val_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        }
        
        stats = {
            'total': len(valid_records),
            'train': len(train_texts),
            'val': len(val_texts),
            'test': len(test_texts),
            'class_distribution': {
                'all': dict(all_counts),
                'train': dict(train_counts),
                'val': dict(val_counts),
                'test': dict(test_counts)
            }
        }
        
        log(f"Data split complete: {stats['train']} train, {stats['val']} validation, {stats['test']} test samples")
        return data_splits, stats
    
    def train_model(self, data_splits: Dict[str, Dict[str, List[Any]]], output_file: str = "sentiment_model",
                   max_epochs: int = None, batch_size: int = None, learning_rate: float = None,
                   weight_decay: float = None, hidden_dims: List[int] = None, 
                   early_stopping_patience: int = None) -> Dict[str, Any]:
        """
        Train a machine learning model on the prepared data.
        
        Args:
            data_splits: Dictionary with train/val/test splits
            output_file: Base filename for saving model and metrics
            max_epochs: Maximum number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            hidden_dims: List of hidden layer dimensions
            early_stopping_patience: Number of epochs to wait before early stopping
            
        Returns:
            Dictionary with training results and metrics
        """
        if not self.use_ml or not TORCH_AVAILABLE:
            return {
                "error": "ML libraries not available or ML mode not enabled",
                "success": False
            }
        
        if data_splits is None:
            return {
                "error": "No data provided for training",
                "success": False
            }
        
        log("Starting model training...")
        
        # Extract training parameters (use params if not explicitly provided)
        max_epochs = max_epochs or self.params["epochs"]
        batch_size = batch_size or self.params["train_batch_size"]
        learning_rate = learning_rate or self.params["learning_rate"]
        weight_decay = weight_decay or self.params["weight_decay"]
        hidden_dims = hidden_dims or self.params["hidden_dims"]
        early_stopping_patience = early_stopping_patience or self.params["early_stopping_patience"]
        
        # Create output directory
        output_dir = Path(output_file).parent
        if output_dir != Path('.'):
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data from splits
        train_texts = data_splits['train']['texts']
        train_labels = data_splits['train']['labels']
        val_texts = data_splits['val']['texts']
        val_labels = data_splits['val']['labels']
        
        # Track training time
        start_time = time.time()
        
        try:
            # Prepare datasets
            log("Preparing datasets...")
            
            # Create datasets based on model type
            if self.ml_model_type == "bert" and TRANSFORMERS_AVAILABLE and self.tokenizer:
                log("Creating BERT-based datasets")
                
                # Create datasets with tokenizer
                train_dataset = ArabicSentimentDataset(
                    train_texts, train_labels, analyzer=self, tokenizer=self.tokenizer
                )
                val_dataset = ArabicSentimentDataset(
                    val_texts, val_labels, analyzer=self, tokenizer=self.tokenizer
                )
                
                # Create data loaders
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=self.params["eval_batch_size"], shuffle=False
                )
                
                # Create model
                self.model = BertSentimentClassifier(
                    bert_model_name=self.bert_model_name,
                    feature_dim=self.params["feature_dim"],
                    dropout_rate=self.params["dropout_rate"]
                ).to(self.device)
                
                # Define optimizer and scheduler
                optimizer = AdamW(
                    self.model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
                
                # Create learning rate scheduler
                total_steps = len(train_loader) * max_epochs
                warmup_steps = self.params["warmup_steps"]
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
                
            else:
                log(f"Creating {self.ml_model_type} datasets with rule-based features")
                
                # Generate features for all texts
                train_features = []
                train_numeric_labels = []
                for i, (text, label) in enumerate(zip(train_texts, train_labels)):
                    # Extract features
                    result = self.analyze_text(text)
                    features = [
                        result.get('pos_score', 0.0),
                        result.get('neg_score', 0.0),
                        result.get('total_score', 0.0),
                        result.get('emoji_sentiment', 0.0),
                        result.get('phrase_sentiment', 0.0),
                        float(len(result.get('pos_words', []))),
                        float(len(result.get('neg_words', []))),
                        float(len(result.get('emojis', []))),
                        float(len(result.get('phrases', []))),
                        result.get('score', 0.5),
                    ]
                    train_features.append(features)
                    
                    # Convert label to numeric
                    label_map = {"positive": 0, "negative": 1, "neutral": 2}
                    train_numeric_labels.append(label_map.get(label, 0))
                    
                    # Log progress
                    if (i + 1) % 100 == 0:
                        log(f"Processed {i + 1}/{len(train_texts)} training samples")
                
                # Do the same for validation set
                val_features = []
                val_numeric_labels = []
                for i, (text, label) in enumerate(zip(val_texts, val_labels)):
                    # Extract features
                    result = self.analyze_text(text)
                    features = [
                        result.get('pos_score', 0.0),
                        result.get('neg_score', 0.0),
                        result.get('total_score', 0.0),
                        result.get('emoji_sentiment', 0.0),
                        result.get('phrase_sentiment', 0.0),
                        float(len(result.get('pos_words', []))),
                        float(len(result.get('neg_words', []))),
                        float(len(result.get('emojis', []))),
                        float(len(result.get('phrases', []))),
                        result.get('score', 0.5),
                    ]
                    val_features.append(features)
                    
                    # Convert label to numeric
                    label_map = {"positive": 0, "negative": 1, "neutral": 2}
                    val_numeric_labels.append(label_map.get(label, 0))
                    
                    # Log progress
                    if (i + 1) % 100 == 0:
                        log(f"Processed {i + 1}/{len(val_texts)} validation samples")
                
                # Convert to tensors
                train_features_tensor = torch.tensor(train_features, dtype=torch.float)
                train_labels_tensor = torch.tensor(train_numeric_labels, dtype=torch.long)
                val_features_tensor = torch.tensor(val_features, dtype=torch.float)
                val_labels_tensor = torch.tensor(val_numeric_labels, dtype=torch.long)
                
                # Create TensorDatasets
                train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
                val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
                
                # Create data loaders
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=self.params["eval_batch_size"], shuffle=False
                )
                
                # Create model
                if self.ml_model_type == "hybrid":
                    self.model = HybridSentimentClassifier(
                        input_dim=self.params["feature_dim"],
                        hidden_dims=hidden_dims,
                        dropout_rate=self.params["dropout_rate"]
                    ).to(self.device)
                else:
                    self.model = SentimentClassifier(
                        input_dim=self.params["feature_dim"],
                        hidden_dims=hidden_dims,
                        dropout_rate=self.params["dropout_rate"]
                    ).to(self.device)
                
                # Define optimizer and scheduler
                optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=2
                )
            
            # Define loss function
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            log(f"Starting training for {max_epochs} epochs")
            
            # Initialize tracking variables
            best_val_accuracy = 0.0
            best_val_f1 = 0.0
            best_epoch = 0
            patience_counter = 0
            training_history = {
                'train_loss': [],
                'train_accuracy': [],
                'val_loss': [],
                'val_accuracy': [],
                'val_f1': [],
                'lr': []
            }
            
            # Training loop
            for epoch in range(max_epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                # Create progress bar for training
                train_iterator = tqdm_auto(
                    train_loader, 
                    desc=f"Epoch {epoch+1}/{max_epochs} [Train]",
                    leave=False
                ) if TQDM_AVAILABLE else train_loader
                
                for batch in train_iterator:
                    # Get batch data
                    if self.ml_model_type == "bert":
                        # Process BERT batch
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        # Forward pass
                        optimizer.zero_grad()
                        
                        # Check if we have additional features
                        if 'features' in batch:
                            features = batch['features'].to(self.device)
                            outputs = self.model(input_ids, attention_mask, features)
                        else:
                            outputs = self.model(input_ids, attention_mask)
                    else:
                        # Process feature-based batch
                        features, labels = batch
                        features = features.to(self.device)
                        labels = labels.to(self.device)
                        
                        # Forward pass
                        optimizer.zero_grad()
                        outputs = self.model(features)
                    
                    # Calculate loss and backpropagate
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    # Update training metrics
                    train_loss += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs, 1)
                    train_correct += (predicted == labels).sum().item()
                    train_total += labels.size(0)
                    
                    # Update learning rate for BERT
                    if self.ml_model_type == "bert" and isinstance(scheduler, type(get_linear_schedule_with_warmup(optimizer, 1, 10))):
                        scheduler.step()
                
                # Calculate training metrics
                train_loss = train_loss / train_total
                train_accuracy = train_correct / train_total
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                all_labels = []
                all_predictions = []
                
                # Create progress bar for validation
                val_iterator = tqdm_auto(
                    val_loader, 
                    desc=f"Epoch {epoch+1}/{max_epochs} [Val]",
                    leave=False
                ) if TQDM_AVAILABLE else val_loader
                
                with torch.no_grad():
                    for batch in val_iterator:
                        # Get batch data
                        if self.ml_model_type == "bert":
                            # Process BERT batch
                            input_ids = batch['input_ids'].to(self.device)
                            attention_mask = batch['attention_mask'].to(self.device)
                            labels = batch['label'].to(self.device)
                            
                            # Forward pass
                            if 'features' in batch:
                                features = batch['features'].to(self.device)
                                outputs = self.model(input_ids, attention_mask, features)
                            else:
                                outputs = self.model(input_ids, attention_mask)
                        else:
                            # Process feature-based batch
                            features, labels = batch
                            features = features.to(self.device)
                            labels = labels.to(self.device)
                            
                            # Forward pass
                            outputs = self.model(features)
                        
                        # Calculate loss
                        loss = criterion(outputs, labels)
                        
                        # Update validation metrics
                        val_loss += loss.item() * labels.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)
                        
                        # Store for F1 calculation
                        all_labels.extend(labels.cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())
                
                # Calculate validation metrics
                val_loss = val_loss / val_total
                val_accuracy = val_correct / val_total
                
                # Calculate F1 score (macro average)
                val_f1 = f1_score(all_labels, all_predictions, average='macro')
                
                # Update learning rate scheduler for non-BERT models
                if self.ml_model_type != "bert" or not isinstance(scheduler, type(get_linear_schedule_with_warmup(optimizer, 1, 10))):
                    scheduler.step(val_f1)  # Use F1 for scheduling
                
                # Update training history
                training_history['train_loss'].append(train_loss)
                training_history['train_accuracy'].append(train_accuracy)
                training_history['val_loss'].append(val_loss)
                training_history['val_accuracy'].append(val_accuracy)
                training_history['val_f1'].append(val_f1)
                training_history['lr'].append(optimizer.param_groups[0]['lr'])
                
                # Print epoch results
                log(f"Epoch {epoch+1}/{max_epochs}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
                
                # Check for best model
                improved = False
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_val_accuracy = val_accuracy
                    best_epoch = epoch + 1
                    
                    # Save best model
                    model_path = Path(output_file + ".pt")
                    torch.save(self.model.state_dict(), model_path)
                    log(f"New best model saved to {model_path} (F1: {best_val_f1:.4f})")
                    
                    # Reset patience counter
                    patience_counter = 0
                    improved = True
                else:
                    # Increment patience counter
                    patience_counter += 1
                
                # Early stopping check
                if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                    log(f"Early stopping triggered after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)")
                    break
                
                # Log memory usage
                log_memory()
            
            # Training complete
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Load best model
            best_model_path = Path(output_file + ".pt")
            if best_model_path.exists():
                self.model.load_state_dict(torch.load(best_model_path))
                log(f"Loaded best model from epoch {best_epoch}")
            
            # Save training history
            history_path = Path(output_file + "_history.json")
            with open(history_path, 'w') as f:
                # Convert tensors to Python types for JSON serialization
                serializable_history = {}
                for key, values in training_history.items():
                    serializable_history[key] = [float(v) for v in values]
                json.dump(serializable_history, f, indent=2)
            
            # Generate training plots
            self._plot_training_history(training_history, output_file + "_plots.png")
            
            # Return training results
            return {
                "success": True,
                "epochs_completed": epoch + 1,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
                "best_val_f1": best_val_f1,
                "training_time": elapsed,
                "model_path": str(best_model_path),
                "history_path": str(history_path)
            }
            
        except Exception as e:
            log(f"Error during model training: {e}")
            log(f"Traceback: {traceback.format_exc()}")
            
            return {
                "error": str(e),
                "success": False
            }
    
    def _plot_training_history(self, history: Dict[str, List[float]], output_file: str) -> None:
        """
        Plot training metrics history and save to file.
        
        Args:
            history: Dictionary with training metrics history
            output_file: Path to save plot image
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Training Metrics', fontsize=16)
            
            # Plot loss
            axes[0, 0].plot(history['train_loss'], label='Train')
            axes[0, 0].plot(history['val_loss'], label='Validation')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Plot accuracy
            axes[0, 1].plot(history['train_accuracy'], label='Train')
            axes[0, 1].plot(history['val_accuracy'], label='Validation')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Plot F1 score
            axes[1, 0].plot(history['val_f1'], label='Validation F1')
            axes[1, 0].set_title('F1 Score (Macro)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Plot learning rate
            axes[1, 1].plot(history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save figure
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            log(f"Training plots saved to {output_file}")
        except Exception as e:
            log(f"Error generating training plots: {e}")
    
    def evaluate_model(self, data_splits: Dict[str, Dict[str, List[Any]]], output_file: str = "model_evaluation") -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            data_splits: Dictionary with train/val/test splits
            output_file: Base filename for saving evaluation results
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.use_ml or not TORCH_AVAILABLE or self.model is None:
            return {
                "error": "No trained model available for evaluation",
                "success": False
            }
        
        if data_splits is None or 'test' not in data_splits:
            return {
                "error": "No test data provided for evaluation",
                "success": False
            }
        
        log("Evaluating model on test data...")
        
        # Extract test data
        test_texts = data_splits['test']['texts']
        test_labels = data_splits['test']['labels']
        
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            # Prepare test dataset
            if self.ml_model_type == "bert" and TRANSFORMERS_AVAILABLE and self.tokenizer:
                # Create BERT dataset
                test_dataset = ArabicSentimentDataset(
                    test_texts, test_labels, analyzer=self, tokenizer=self.tokenizer
                )
            else:
                # Generate features for test data
                test_features = []
                test_numeric_labels = []
                for text, label in zip(test_texts, test_labels):
                    # Extract features
                    result = self.analyze_text(text)
                    features = [
                        result.get('pos_score', 0.0),
                        result.get('neg_score', 0.0),
                        result.get('total_score', 0.0),
                        result.get('emoji_sentiment', 0.0),
                        result.get('phrase_sentiment', 0.0),
                        float(len(result.get('pos_words', []))),
                        float(len(result.get('neg_words', []))),
                        float(len(result.get('emojis', []))),
                        float(len(result.get('phrases', []))),
                        result.get('score', 0.5),
                    ]
                    test_features.append(features)
                    
                    # Convert label to numeric
                    label_map = {"positive": 0, "negative": 1, "neutral": 2}
                    test_numeric_labels.append(label_map.get(label, 0))
                
                # Convert to tensors
                test_features_tensor = torch.tensor(test_features, dtype=torch.float)
                test_labels_tensor = torch.tensor(test_numeric_labels, dtype=torch.long)
                
                # Create dataset
                test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
            
            # Create data loader
            test_loader = DataLoader(
                test_dataset, batch_size=self.params["eval_batch_size"], shuffle=False
            )
            
            # Evaluation loop
            all_labels = []
            all_predictions = []
            
            with torch.no_grad():
                for batch in tqdm_auto(test_loader, desc="Evaluating") if TQDM_AVAILABLE else test_loader:
                    # Get batch data
                    if self.ml_model_type == "bert":
                        # Process BERT batch
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        # Forward pass
                        if 'features' in batch:
                            features = batch['features'].to(self.device)
                            outputs = self.model(input_ids, attention_mask, features)
                        else:
                            outputs = self.model(input_ids, attention_mask)
                    else:
                        # Process feature-based batch
                        features, labels = batch
                        features = features.to(self.device)
                        labels = labels.to(self.device)
                        
                        # Forward pass
                        outputs = self.model(features)
                    
                    # Get predictions
                    _, predicted = torch.max(outputs, 1)
                    
                    # Store for metrics calculation
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
            
            # Convert numeric labels back to text
            label_map_reverse = {0: "positive", 1: "negative", 2: "neutral"}
            text_labels = [label_map_reverse[l] for l in all_labels]
            text_predictions = [label_map_reverse[p] for p in all_predictions]
            
            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_predictions)
            f1_macro = f1_score(all_labels, all_predictions, average='macro')
            f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
            class_report = classification_report(all_labels, all_predictions, target_names=["positive", "negative", "neutral"], output_dict=True)
            conf_matrix = confusion_matrix(all_labels, all_predictions)
            
            # Save evaluation results
            results = {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "classification_report": class_report,
                "confusion_matrix": conf_matrix.tolist(),
                "model_type": self.ml_model_type,
                "test_samples": len(test_texts)
            }
            
            # Save results to file
            results_file = f"{output_file}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Generate evaluation report
            report_file = f"{output_file}_report.txt"
            with open(report_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("MODEL EVALUATION REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Model type: {self.ml_model_type}\n")
                f.write(f"Test samples: {len(test_texts)}\n\n")
                
                f.write("Overall metrics:\n")
                f.write(f"  Accuracy: {accuracy:.4f}\n")
                f.write(f"  F1 (macro): {f1_macro:.4f}\n")
                f.write(f"  F1 (weighted): {f1_weighted:.4f}\n\n")
                
                f.write("Classification Report:\n")
                f.write(classification_report(all_labels, all_predictions, target_names=["positive", "negative", "neutral"]))
                f.write("\n")
                
                f.write("Confusion Matrix:\n")
                f.write(f"{conf_matrix}\n\n")
                
                # Sample predictions
                f.write("Sample Predictions (first 10):\n")
                for i in range(min(10, len(text_labels))):
                    f.write(f"  Text: {test_texts[i][:50]}{'...' if len(test_texts[i]) > 50 else ''}\n")
                    f.write(f"  True: {text_labels[i]}, Predicted: {text_predictions[i]}\n\n")
                
                f.write("=" * 80 + "\n")
            
            # Plot confusion matrix
            self._plot_confusion_matrix(conf_matrix, ["positive", "negative", "neutral"], f"{output_file}_confusion.png")
            
            # Return evaluation results
            return {
                "success": True,
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "results_file": results_file,
                "report_file": report_file
            }
            
        except Exception as e:
            log(f"Error during model evaluation: {e}")
            log(f"Traceback: {traceback.format_exc()}")
            
            return {
                "error": str(e),
                "success": False
            }
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], output_file: str) -> None:
        """
        Plot confusion matrix and save to file.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            output_file: Path to save plot image
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Normalize confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create figure
            plt.figure(figsize=(10, 8))
            plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Normalized Confusion Matrix', fontsize=16)
            plt.colorbar()
            
            # Add class labels
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
            plt.yticks(tick_marks, class_names, fontsize=12)
            
            # Add values to cells
            thresh = cm_norm.max() / 2.0
            for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]:.2f})",
                        horizontalalignment="center",
                        color="white" if cm_norm[i, j] > thresh else "black",
                        fontsize=10)
            
            plt.tight_layout()
            plt.ylabel('True label', fontsize=14)
            plt.xlabel('Predicted label', fontsize=14)
            
            # Save figure
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            log(f"Confusion matrix plot saved to {output_file}")
        except Exception as e:
            log(f"Error generating confusion matrix plot: {e}")
    
    def tune_hyperparameters(self, data_splits: Dict[str, Dict[str, List[Any]]], output_file: str = "hyperparameter_tuning",
                           num_trials: int = 10, search_space: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning to find the best model configuration.
        
        Args:
            data_splits: Dictionary with train/val/test splits
            output_file: Base filename for saving tuning results
            num_trials: Number of hyperparameter configurations to try
            search_space: Dictionary with hyperparameter search ranges (overrides defaults)
            
        Returns:
            Dictionary with tuning results and best parameters
        """
        if not self.use_ml or not TORCH_AVAILABLE:
            return {
                "error": "ML libraries not available or ML mode not enabled",
                "success": False
            }
        
        if data_splits is None:
            return {
                "error": "No data provided for tuning",
                "success": False
            }
        
        log(f"Starting hyperparameter tuning with {num_trials} trials...")
        
        # Define default search space
        default_search_space = {
            "learning_rate": (1e-5, 5e-4),
            "batch_size": [16, 32, 64],
            "hidden_dims": [
                [32],
                [64],
                [128],
                [64, 32],
                [128, 64],
                [128, 64, 32]
            ],
            "dropout_rate": (0.1, 0.5),
            "weight_decay": (0.001, 0.1)
        }
        
        # Override with provided search space if any
        if search_space:
            for key, value in search_space.items():
                default_search_space[key] = value
        
        # Create output directory
        output_dir = Path(output_file).parent
        if output_dir != Path('.'):
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking variables
        best_val_f1 = 0.0
        best_params = None
        best_metrics = None
        tuning_history = []
        
        # Run trials
        for trial in range(num_trials):
            log(f"\nTrial {trial+1}/{num_trials}")
            
            # Sample hyperparameters
            params = {}
            for name, space in default_search_space.items():
                if isinstance(space, tuple) and len(space) == 2:
                    # Continuous parameter
                    min_val, max_val = space
                    if name == "learning_rate" or name == "weight_decay":
                        # Sample on log scale
                        log_min = math.log10(min_val)
                        log_max = math.log10(max_val)
                        value = 10 ** (log_min + (log_max - log_min) * random.random())
                    else:
                        # Sample on linear scale
                        value = min_val + (max_val - min_val) * random.random()
                    params[name] = value
                elif isinstance(space, list):
                    # Categorical parameter
                    params[name] = random.choice(space)
                else:
                    # Fixed parameter
                    params[name] = space
            
            log(f"Trial parameters: {json.dumps(params, default=lambda x: str(x), indent=2)}")
            
            # Update model parameters
            trial_params = self.params.copy()
            trial_params.update(params)
            
            # Train model with these parameters
            trial_output_file = f"{output_file}_trial_{trial+1}"
            train_result = self.train_model(
                data_splits=data_splits,
                output_file=trial_output_file,
                max_epochs=5,  # Use fewer epochs for tuning
                batch_size=params.get("batch_size", self.params["train_batch_size"]),
                learning_rate=params.get("learning_rate", self.params["learning_rate"]),
                weight_decay=params.get("weight_decay", self.params["weight_decay"]),
                hidden_dims=params.get("hidden_dims", self.params["hidden_dims"]),
                early_stopping_patience=2  # Use shorter patience for tuning
            )
            
            # Skip failed trials
            if not train_result.get("success", False):
                log(f"Trial {trial+1} failed: {train_result.get('error', 'Unknown error')}")
                continue
            
            # Evaluate on validation data
            val_f1 = train_result.get("best_val_f1", 0.0)
            val_accuracy = train_result.get("best_val_accuracy", 0.0)
            
            # Store trial results
            trial_result = {
                "trial": trial + 1,
                "params": params,
                "val_f1": val_f1,
                "val_accuracy": val_accuracy,
                "epochs_completed": train_result.get("epochs_completed", 0),
                "best_epoch": train_result.get("best_epoch", 0)
            }
            tuning_history.append(trial_result)
            
            # Update best if improved
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_params = params.copy()
                best_metrics = {
                    "val_f1": val_f1,
                    "val_accuracy": val_accuracy,
                    "epochs_completed": train_result.get("epochs_completed", 0),
                    "best_epoch": train_result.get("best_epoch", 0),
                    "trial": trial + 1
                }
                log(f"New best F1: {best_val_f1:.4f} (Trial {trial+1})")
                
                # Copy best model
                best_model_path = Path(train_result.get("model_path", ""))
                if best_model_path.exists():
                    target_path = Path(f"{output_file}_best_model.pt")
                    import shutil
                    shutil.copy(best_model_path, target_path)
                    log(f"Copied best model to {target_path}")
        
        # Save tuning results
        tuning_results = {
            "best_params": best_params,
            "best_metrics": best_metrics,
            "history": tuning_history,
            "model_type": self.ml_model_type,
            "num_trials": num_trials,
            "search_space": default_search_space
        }
        
        results_file = f"{output_file}_results.json"
        with open(results_file, 'w') as f:
            json.dump(tuning_results, f, default=lambda x: str(x), indent=2)
        
        # Generate tuning report
        report_file = f"{output_file}_report.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HYPERPARAMETER TUNING REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model type: {self.ml_model_type}\n")
            f.write(f"Number of trials: {num_trials}\n")
            f.write(f"Best validation F1: {best_val_f1:.4f}\n")
            f.write(f"Best validation accuracy: {best_metrics.get('val_accuracy', 0.0):.4f}\n")
            f.write(f"Best trial: {best_metrics.get('trial', 0)}\n\n")
            
            f.write("Best parameters:\n")
            for name, value in best_params.items():
                f.write(f"  {name}: {value}\n")
            f.write("\n")
            
            f.write("All trials:\n")
            for trial in tuning_history:
                f.write(f"  Trial {trial['trial']}: F1={trial['val_f1']:.4f}, Accuracy={trial['val_accuracy']:.4f}\n")
                f.write(f"    Parameters: {json.dumps(trial['params'], default=lambda x: str(x))}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        # Generate tuning plots
        self._plot_tuning_results(tuning_history, f"{output_file}_plots.png")
        
        # Return tuning results
        return {
            "success": True,
            "best_params": best_params,
            "best_val_f1": best_val_f1,
            "best_val_accuracy": best_metrics.get("val_accuracy", 0.0),
            "results_file": results_file,
            "report_file": report_file,
            "best_model_path": f"{output_file}_best_model.pt"
        }
    
    def _plot_tuning_results(self, history: List[Dict[str, Any]], output_file: str) -> None:
        """
        Plot hyperparameter tuning results and save to file.
        
        Args:
            history: List of trial results
            output_file: Path to save plot image
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Extract data
            trials = [t['trial'] for t in history]
            f1_scores = [t['val_f1'] for t in history]
            accuracy_scores = [t['val_accuracy'] for t in history]
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 1, figsize=(10, 12))
            
            # Plot F1 scores
            axes[0].plot(trials, f1_scores, 'o-', color='blue')
            axes[0].set_title('Validation F1 Score by Trial', fontsize=14)
            axes[0].set_xlabel('Trial', fontsize=12)
            axes[0].set_ylabel('F1 Score', fontsize=12)
            axes[0].grid(True)
            
            # Add point labels
            for i, (x, y) in enumerate(zip(trials, f1_scores)):
                axes[0].annotate(f"{y:.4f}", (x, y), textcoords="offset points", 
                               xytext=(0, 10), ha='center')
            
            # Plot accuracy scores
            axes[1].plot(trials, accuracy_scores, 'o-', color='green')
            axes[1].set_title('Validation Accuracy by Trial', fontsize=14)
            axes[1].set_xlabel('Trial', fontsize=12)
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].grid(True)
            
            # Add point labels
            for i, (x, y) in enumerate(zip(trials, accuracy_scores)):
                axes[1].annotate(f"{y:.4f}", (x, y), textcoords="offset points", 
                               xytext=(0, 10), ha='center')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            log(f"Tuning plots saved to {output_file}")
        except Exception as e:
            log(f"Error generating tuning plots: {e}")
    
    def finalize_model(self, best_params: Dict[str, Any], data_splits: Dict[str, Dict[str, List[Any]]], 
                     output_file: str = "final_model") -> Dict[str, Any]:
        """
        Train a final model with the best hyperparameters and evaluate it.
        
        Args:
            best_params: Dictionary with best hyperparameters
            data_splits: Dictionary with train/val/test splits
            output_file: Base filename for saving final model and results
            
        Returns:
            Dictionary with final model results
        """
        if not self.use_ml or not TORCH_AVAILABLE:
            return {
                "error": "ML libraries not available or ML mode not enabled",
                "success": False
            }
        
        if data_splits is None:
            return {
                "error": "No data provided for training",
                "success": False
            }
        
        log("Training final model with best hyperparameters...")
        
        # Update model parameters
        final_params = self.params.copy()
        final_params.update(best_params)
        
        # Save original parameters to restore later
        original_params = self.params.copy()
        self.params = final_params
        
        try:
            # Train model with best parameters
            train_result = self.train_model(
                data_splits=data_splits,
                output_file=output_file,
                max_epochs=self.params["epochs"],
                batch_size=self.params.get("batch_size", self.params["train_batch_size"]),
                learning_rate=self.params.get("learning_rate", self.params["learning_rate"]),
                weight_decay=self.params.get("weight_decay", self.params["weight_decay"]),
                hidden_dims=self.params.get("hidden_dims", self.params["hidden_dims"]),
                early_stopping_patience=self.params["early_stopping_patience"]
            )
            
            # Skip if training failed
            if not train_result.get("success", False):
                log(f"Final model training failed: {train_result.get('error', 'Unknown error')}")
                self.params = original_params  # Restore original params
                return train_result
            
            # Evaluate final model
            eval_result = self.evaluate_model(
                data_splits=data_splits,
                output_file=f"{output_file}_evaluation"
            )
            
            # Combine results
            final_results = {
                "success": True,
                "training_results": train_result,
                "evaluation_results": eval_result,
                "best_parameters": best_params,
                "model_path": train_result.get("model_path"),
                "final_test_accuracy": eval_result.get("accuracy"),
                "final_test_f1": eval_result.get("f1_macro")
            }
            
            # Save final results
            results_file = f"{output_file}_results.json"
            with open(results_file, 'w') as f:
                json.dump(final_results, f, default=lambda x: str(x), indent=2)
            
            # Generate summary report
            report_file = f"{output_file}_summary.txt"
            with open(report_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("FINAL MODEL SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(f"Model type: {self.ml_model_type}\n\n")
                
                f.write("Best parameters:\n")
                for name, value in best_params.items():
                    f.write(f"  {name}: {value}\n")
                f.write("\n")
                
                f.write("Training metrics:\n")
                f.write(f"  Epochs completed: {train_result.get('epochs_completed', 0)}\n")
                f.write(f"  Best epoch: {train_result.get('best_epoch', 0)}\n")
                f.write(f"  Best validation F1: {train_result.get('best_val_f1', 0.0):.4f}\n")
                f.write(f"  Best validation accuracy: {train_result.get('best_val_accuracy', 0.0):.4f}\n")
                f.write(f"  Training time: {train_result.get('training_time', 0.0):.2f} seconds\n\n")
                
                f.write("Test metrics:\n")
                f.write(f"  Test accuracy: {eval_result.get('accuracy', 0.0):.4f}\n")
                f.write(f"  Test F1 (macro): {eval_result.get('f1_macro', 0.0):.4f}\n")
                f.write(f"  Test F1 (weighted): {eval_result.get('f1_weighted', 0.0):.4f}\n\n")
                
                f.write("Files:\n")
                f.write(f"  Model: {train_result.get('model_path', '')}\n")
                f.write(f"  Training history: {train_result.get('history_path', '')}\n")
                f.write(f"  Evaluation report: {eval_result.get('report_file', '')}\n")
                f.write(f"  Results summary: {results_file}\n")
                
                f.write("\n" + "=" * 80 + "\n")
            
            log(f"Final model summary saved to {report_file}")
            
            # Restore original parameters
            self.params = original_params
            
            return final_results
            
        except Exception as e:
            log(f"Error during final model training: {e}")
            log(f"Traceback: {traceback.format_exc()}")
            
            # Restore original parameters
            self.params = original_params
            
            return {
                "error": str(e),
                "success": False
            }

# Main execution
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Arabic sentiment analysis on Qdrant collection')
    parser.add_argument('--collection', type=str, default='sentiment_analysis_dataset',
                      help='Name of Qdrant collection')
    parser.add_argument('--host', type=str, default='localhost',
                      help='Qdrant server host')
    parser.add_argument('--port', type=int, default=6333,
                      help='Qdrant server port')
    parser.add_argument('--limit', type=int, default=None,
                      help='Limit number of records to analyze')
    parser.add_argument('--output', type=str, default='sentiment_results',
                      help='Output filename prefix')
    parser.add_argument('--cache-dir', type=str, default='./sentiment_cache',
                      help='Cache directory')
    parser.add_argument('--model-dir', type=str, default='./sentiment_models',
                      help='Model directory')
    parser.add_argument('--batch-size', type=int, default=1000,
                      help='Batch size for processing')
    parser.add_argument('--memory-limit', type=int, default=1024,
                      help='Memory limit in MB')
    parser.add_argument('--no-cache', action='store_true',
                      help='Force new analysis, ignoring cache')
    parser.add_argument('--use-local-data', action='store_true',
                      help='Use local data file instead of Qdrant')
    parser.add_argument('--local-data-path', type=str, default=None,
                      help='Path to local data file (JSON format)')
    parser.add_argument('--use-ml', action='store_true',
                      help='Use machine learning model for predictions')
    parser.add_argument('--ml-model-type', type=str, default='hybrid',
                      choices=['hybrid', 'bert', 'neural'],
                      help='Type of ML model to use')
    parser.add_argument('--bert-model', type=str, default='asafaya/bert-base-arabic',
                      help='BERT model to use')
    parser.add_argument('--train', action='store_true',
                      help='Train a new ML model')
    parser.add_argument('--evaluate', action='store_true',
                      help='Evaluate the trained model')
    parser.add_argument('--tune', action='store_true',
                      help='Perform hyperparameter tuning')
    parser.add_argument('--num-trials', type=int, default=10,
                      help='Number of trials for hyperparameter tuning')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use for PyTorch (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = OptimizedArabicSentimentAnalyzer(
        collection_name=args.collection,
        qdrant_host=args.host,
        qdrant_port=args.port,
        force_new_analysis=args.no_cache,
        cache_dir=args.cache_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        memory_limit_mb=args.memory_limit,
        use_local_data=args.use_local_data,
        local_data_path=args.local_data_path,
        use_ml=args.use_ml,
        ml_model_type=args.ml_model_type,
        bert_model_name=args.bert_model,
        device=args.device
    )
    
    # Run different modes based on arguments
    if args.train:
        # Prepare data for training
        data_splits, stats = analyzer.prepare_data_for_training(limit=args.limit)
        
        if data_splits is None:
            log("Failed to prepare data for training")
            sys.exit(1)
        
        log(f"Data statistics: {json.dumps(stats, indent=2)}")
        
        if args.tune:
            # Perform hyperparameter tuning
            tuning_results = analyzer.tune_hyperparameters(
                data_splits=data_splits,
                output_file=f"{args.output}_tuning",
                num_trials=args.num_trials
            )
            
            if tuning_results.get("success"):
                log(f"Hyperparameter tuning complete. Best F1: {tuning_results['best_val_f1']:.4f}")
                
                # Train final model with best parameters
                final_results = analyzer.finalize_model(
                    best_params=tuning_results["best_params"],
                    data_splits=data_splits,
                    output_file=f"{args.output}_final"
                )
                
                if final_results.get("success"):
                    log(f"Final model training complete. Test F1: {final_results['final_test_f1']:.4f}")
            else:
                log(f"Hyperparameter tuning failed: {tuning_results.get('error')}")
        else:
            # Train model directly
            train_results = analyzer.train_model(
                data_splits=data_splits,
                output_file=f"{args.output}_model"
            )
            
            if train_results.get("success"):
                log(f"Model training complete. Best validation F1: {train_results['best_val_f1']:.4f}")
                
                # Evaluate model
                eval_results = analyzer.evaluate_model(
                    data_splits=data_splits,
                    output_file=f"{args.output}_evaluation"
                )
                
                if eval_results.get("success"):
                    log(f"Model evaluation complete. Test accuracy: {eval_results['accuracy']:.4f}")
            else:
                log(f"Model training failed: {train_results.get('error')}")
    
    elif args.evaluate:
        # Just evaluate existing model
        data_splits, stats = analyzer.prepare_data_for_training(limit=args.limit)
        
        if data_splits is None:
            log("Failed to prepare data for evaluation")
            sys.exit(1)
        
        eval_results = analyzer.evaluate_model(
            data_splits=data_splits,
            output_file=f"{args.output}_evaluation"
        )
        
        if eval_results.get("success"):
            log(f"Model evaluation complete. Test accuracy: {eval_results['accuracy']:.4f}")
        else:
            log(f"Model evaluation failed: {eval_results.get('error')}")
    
    else:
        # Run standard sentiment analysis
        results = analyzer.run_analysis(limit=args.limit, output_file=args.output)
        
        # Print summary
        if "error" in results:
            log(f"Analysis failed: {results['error']}")
        else:
            log(f"\nAnalysis completed successfully!")
            log(f"Total records analyzed: {results['results_count']}")
            log(f"Time elapsed: {results['elapsed_time']:.2f} seconds")
            log(f"Output files: {', '.join(results['output_files'])}")
            
            # Print sentiment distribution
            if 'metrics' in results:
                metrics = results['metrics']
                log("\nSentiment distribution:")
                for sentiment, count in metrics['sentiment_counts'].items():
                    percentage = metrics['sentiment_percentages'][sentiment]
                    log(f"  {sentiment}: {count} ({percentage:.2f}%)")
                
                # Print accuracy if available
                if metrics['evaluation']['has_ground_truth']:
                    log(f"\nAccuracy: {metrics['evaluation']['accuracy']:.4f}")
                    log(f"F1 scores:")
                    for label, class_metrics in metrics['evaluation']['class_metrics'].items():
                        log(f"  {label}: {class_metrics['f1']:.4f}")
                    
                    # Print overall metrics
                    log(f"\nOverall performance:")
                    log(f"  Total with ground truth: {metrics['evaluation']['ground_truth_count']}")
                    
                    # Calculate macro average of metrics
                    precisions = [m['precision'] for m in metrics['evaluation']['class_metrics'].values()]
                    recalls = [m['recall'] for m in metrics['evaluation']['class_metrics'].values()]
                    f1s = [m['f1'] for m in metrics['evaluation']['class_metrics'].values()]
                    
                    log(f"  Macro-averaged precision: {sum(precisions)/len(precisions):.4f}")
                    log(f"  Macro-averaged recall: {sum(recalls)/len(recalls):.4f}")
                    log(f"  Macro-averaged F1: {sum(f1s)/len(f1s):.4f}") 
                    log("\nAnalysis complete!")