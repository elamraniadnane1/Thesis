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
from typing import Dict, List, Any, Optional, Tuple, Iterator, Generator, Union
from collections import Counter, defaultdict
from pathlib import Path
from itertools import product

import tqdm
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("Warning: Qdrant client not available. Only local data mode will work.")

# Set to True to enable detailed logging
DEBUG = True

def log(message: str) -> None:
    """Print debug message if DEBUG is enabled."""
    if DEBUG:
        print(f"[DEBUG] {message}")

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

class OptimizedArabicSentimentAnalyzer:
    """
    Memory-optimized Arabic sentiment analyzer with batch processing capabilities
    designed specifically to achieve 85%+ accuracy.
    """
    
    def __init__(
        self,
        collection_name: str = "sentiment_analysis_dataset",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        force_new_analysis: bool = True,
        cache_dir: str = "./sentiment_cache",
        batch_size: int = 1000,
        params: Optional[Dict[str, Any]] = None,
        memory_limit_mb: int = 1024,  # 1GB default memory limit
        use_local_data: bool = False,
        local_data_path: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        self.memory_limit_mb = memory_limit_mb
        self.use_local_data = use_local_data
        self.local_data_path = local_data_path
        
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

def find_best_parameters(
    limit: int = 1000, 
    tuning_iterations: int = 5,
    output_file: str = "hyperparameter_tuning_results",
    batch_size: int = 100,
    use_local_data: bool = False,
    local_data_path: str = None
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning to find the best configuration.
    Uses random search with progressively more focused parameter ranges.
    Memory-optimized version with batch processing.
    """
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING FOR SENTIMENT ANALYSIS")
    print("=" * 80)
    
    # Initialize tracking variables
    best_accuracy = 0.0
    best_params = None
    best_metrics = None
    results_history = []
    
    # Set parameter search space
    param_ranges = {
        "positive_bias": (2.0, 10.0),
        "negative_bias": (2.0, 8.0),
        "neutral_threshold": (-0.5, 0.1),
        "min_pos_threshold": (0.01, 0.3),
        "min_neg_threshold": (0.01, 0.3),
        "positive_weight": (1.0, 3.0),
        "negative_weight": (1.5, 4.0),
        "neutral_weight": (0.3, 1.0),
    }
    
    # Boolean parameters to try
    boolean_params = {
        "use_contextual_valence_shifters": [True, False],
        "apply_dynamic_weighting": [True, False],
        "prioritize_domain_matches": [True, False],
        "aggressive_classification": [True, False],
    }
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_file).parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # For each iteration, narrow the search space
    for iteration in range(tuning_iterations):
        print(f"\nIteration {iteration+1}/{tuning_iterations}")
        
        # Generate random parameters from current ranges
        params = {}
        for param, (min_val, max_val) in param_ranges.items():
            params[param] = min_val + (max_val - min_val) * random.random()
        
        # Set boolean parameters randomly
        for param, options in boolean_params.items():
            params[param] = random.choice(options)
        
        print(f"Testing parameters: {json.dumps(params, indent=2)}")
        
        # Initialize analyzer with these parameters
        analyzer = OptimizedArabicSentimentAnalyzer(
            force_new_analysis=True,
            params=params,
            batch_size=batch_size,
            use_local_data=use_local_data,
            local_data_path=local_data_path
        )
        
        try:
            # Run analysis
            results = analyzer.run_analysis(limit=limit, output_file=f"tuning_{iteration}")
            
            # Check if analysis was successful
            if results.get("error"):
                print(f"Error during analysis: {results.get('error')}")
                print("Skipping to next iteration...")
                continue
                
            # Extract accuracy
            metrics = results.get("metrics", {})
            eval_metrics = metrics.get("evaluation", {})
            accuracy = eval_metrics.get("accuracy")
            
            # Skip iteration if accuracy is None
            if accuracy is None:
                print("No accuracy data available for this iteration.")
                print("Skipping to next iteration...")
                continue
            
            # Print results
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Class-specific metrics
            class_metrics = eval_metrics.get("class_metrics", {})
            if class_metrics:
                print("Class metrics:")
                for label, metrics_dict in class_metrics.items():
                    print(f"  {label}:")
                    print(f"    Precision: {metrics_dict.get('precision', 0):.4f}")
                    print(f"    Recall: {metrics_dict.get('recall', 0):.4f}")
                    print(f"    F1-score: {metrics_dict.get('f1', 0):.4f}")
            
            # Save history (minimal version to save memory)
            results_history.append({
                "iteration": iteration,
                "params": params,
                "accuracy": accuracy,
                "class_metrics": class_metrics,
                "distribution": metrics.get("sentiment_counts", {})
            })
            
            # Update best if improvement found
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params.copy()
                best_metrics = {
                    "accuracy": accuracy,
                    "class_metrics": class_metrics,
                    "distribution": metrics.get("sentiment_counts", {})
                }
                print(f"New best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
                
                # Check if we've met our target
                if best_accuracy >= 0.85:
                    print("\n🎉 Target accuracy of 85% achieved! 🎉")
                    break
        except Exception as e:
            print(f"Error during iteration {iteration+1}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            print("Continuing to next iteration...")
        finally:
            # Force garbage collection
            gc.collect()
            log_memory()
        
        # Narrow search space around best parameters
        if best_params and iteration < tuning_iterations - 1:
            for param, (min_val, max_val) in param_ranges.items():
                # Calculate new bounds centered around best value
                best_val = best_params[param]
                range_width = (max_val - min_val) * 0.5  # Narrow by 50% each time
                new_min = max(min_val, best_val - range_width/2)
                new_max = min(max_val, best_val + range_width/2)
                param_ranges[param] = (new_min, new_max)
    
    # Save tuning results
    try:
        with open(f"{output_file}.json", "w") as f:
            json.dump({
                "best_params": best_params,
                "best_accuracy": best_accuracy,
                "history": results_history
            }, f, indent=2)
        
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING COMPLETE")
        print("=" * 80)
        
        if best_params:
            print(f"Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            print(f"Best parameters: {json.dumps(best_params, indent=2)}")
            print(f"Results saved to {output_file}.json")
            
            # Run one final analysis with the best parameters
            print("\nRunning final analysis with best parameters...")
            analyzer = OptimizedArabicSentimentAnalyzer(
                force_new_analysis=True,
                params=best_params,
                batch_size=batch_size,
                use_local_data=use_local_data,
                local_data_path=local_data_path
            )
            final_results = analyzer.run_analysis(output_file="optimized_results")
        else:
            print("No successful parameters found during tuning.")
            print("Please check your connection to Qdrant or try with local data.")
    except Exception as e:
        print(f"Error saving tuning results: {str(e)}")
        
    return {
        "best_params": best_params,
        "best_accuracy": best_accuracy,
        "best_metrics": best_metrics,
        "history": results_history
    }

def generate_sample_data(output_file: str, num_samples: int = 100):
    """
    Generate sample data for testing when Qdrant is not available.
    Creates a JSON file with Arabic text samples and sentiment labels.
    """
    print(f"Generating {num_samples} sample records for testing...")
    
    # Arabic text samples (short phrases with sentiment)
    positive_samples = [
        "أنا سعيد جداً بهذه الخدمة",
        "الخدمة ممتازة وسريعة",
        "شكراً على الخدمة الرائعة",
        "المنتج جيد جداً وبجودة عالية",
        "أنا راضٍ تماماً عن الخدمة",
        "موظفو الخدمة محترفون جداً",
        "خدمة عملاء ممتازة",
        "منتج رائع وبسعر معقول",
        "تجربة إيجابية جداً",
        "أفضل خدمة حصلت عليها",
        "المشروع ناجح جداً",
        "عمل رائع، شكراً لكم",
        "تحسن كبير في الخدمات",
        "مبادرة ممتازة وفعالة",
        "الخدمة تستحق التقدير",
        "فريق عمل متعاون ومحترف",
        "خدمة سريعة وفعالة",
        "تطور ملحوظ في المشروع",
        "نشكركم على جهودكم الرائعة",
        "التغييرات إيجابية جداً"
    ]
    
    negative_samples = [
        "الخدمة سيئة جداً",
        "غير راضٍ عن المنتج",
        "خدمة بطيئة ومحبطة",
        "لم أحصل على ما وعدت به",
        "تجربة سلبية ومخيبة للآمال",
        "المنتج رديء الجودة",
        "موظفو الخدمة غير متعاونين",
        "سعر مرتفع مقابل خدمة سيئة",
        "مشاكل كثيرة في الخدمة",
        "الانتظار طويل جداً",
        "الموقع غير منظم وصعب الاستخدام",
        "المشروع متأخر جداً",
        "الخدمة أسوأ من السابق",
        "لا أنصح بهذه الخدمة",
        "مشاكل فنية كثيرة",
        "عدم الاستجابة للشكاوى",
        "خدمة غير مهنية بالمرة",
        "تجربة محبطة جداً",
        "الإدارة سيئة وغير منظمة",
        "أضعف خدمة في المنطقة"
    ]
    
    neutral_samples = [
        "الخدمة عادية",
        "المنتج متوسط",
        "الخدمة مقبولة",
        "لا جيدة ولا سيئة",
        "تجربة عادية",
        "يمكن تحسين الخدمة",
        "أتوقع أن تتحسن الخدمة",
        "بعض النقاط إيجابية وبعضها سلبية",
        "الخدمة تؤدي الغرض",
        "المنتج يعمل بشكل طبيعي",
        "الموظفون عاديون",
        "تجربة متوسطة",
        "الأسعار معقولة نوعاً ما",
        "ليس لدي رأي محدد",
        "سأنتظر لتقييم الخدمة بشكل أفضل",
        "الوقت سيثبت جودة الخدمة",
        "لم أستخدم الخدمة بشكل كافٍ للحكم",
        "الخدمة تلبي الحد الأدنى من المتطلبات",
        "محايد تجاه هذه التجربة",
        "لا أستطيع الحكم بعد"
    ]
    
    # Generate random samples
    records = []
    for i in range(num_samples):
        # Decide sentiment class (40% positive, 40% negative, 20% neutral)
        rand = random.random()
        if rand < 0.4:
            text = random.choice(positive_samples)
            sentiment = "positive"
        elif rand < 0.8:
            text = random.choice(negative_samples)
            sentiment = "negative"
        else:
            text = random.choice(neutral_samples)
            sentiment = "neutral"
        
        # Create record
        record = {
            "id": f"sample_{i}",
            "text": text,
            "sentiment": sentiment
        }
        records.append(record)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"records": records}, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {num_samples} sample records and saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Memory-Optimized Arabic sentiment analysis')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--iterations', type=int, default=5, help='Number of tuning iterations')
    parser.add_argument('--tune_limit', type=int, default=1000, help='Record limit for tuning')
    parser.add_argument('--host', default='localhost', help='Qdrant host')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port')
    parser.add_argument('--collection', default='sentiment_analysis_dataset', help='Collection name')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of records to process')
    parser.add_argument('--output', default='sentiment_results', help='Output file prefix')
    parser.add_argument('--cache_dir', default='./sentiment_cache', help='Directory for caching results')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for processing')
    parser.add_argument('--memory_limit', type=int, default=1024, 
                      help='Memory limit in MB (will try to stay under this limit)')
    parser.add_argument('--local', action='store_true', help='Use local data instead of Qdrant')
    parser.add_argument('--local_data', default=None, help='Path to local data file (JSON)')
    parser.add_argument('--generate_samples', action='store_true', help='Generate sample data for testing')
    parser.add_argument('--sample_count', type=int, default=100, help='Number of sample records to generate')
    parser.add_argument('--sample_output', default='sample_data.json', help='Output file for sample data')
    
    # Preset parameter combinations
    parser.add_argument('--preset', choices=['balanced', 'aggressive', 'high_recall', 'high_precision'], 
                      help='Use a preset parameter configuration')
    
    args = parser.parse_args()
    
    # Generate sample data if requested
    if args.generate_samples:
        sample_file = generate_sample_data(args.sample_output, args.sample_count)
        print(f"You can now use this file with: --local --local_data {sample_file}")
        return
    
    # Define preset parameters
    presets = {
        "balanced": {
            "positive_bias": 3.0,
            "negative_bias": 2.5,
            "neutral_threshold": -0.05,
            "min_pos_threshold": 0.1,
            "min_neg_threshold": 0.1,
            "negation_factor": 1.3,
            "negation_window": 4,
            "context_weight": 1.2,
            "emoji_weight": 1.5,
            "domain_boost": 1.3,
            "arabic_boost": 1.1,
            "use_contextual_valence_shifters": True,
            "apply_dynamic_weighting": True,
            "prioritize_domain_matches": True,
            "aggressive_classification": True,
            "positive_weight": 1.5,
            "negative_weight": 2.0,
            "neutral_weight": 0.7,
        },
        "aggressive": {
            "positive_bias": 5.0,
            "negative_bias": 4.0,
            "neutral_threshold": -0.2,
            "min_pos_threshold": 0.05,
            "min_neg_threshold": 0.05,
            "negation_factor": 1.5,
            "negation_window": 5,
            "context_weight": 1.3,
            "emoji_weight": 2.0,
            "domain_boost": 1.5,
            "arabic_boost": 1.2,
            "use_contextual_valence_shifters": True,
            "apply_dynamic_weighting": True,
            "prioritize_domain_matches": True,
            "aggressive_classification": True,
            "positive_weight": 2.0,
            "negative_weight": 3.0,
            "neutral_weight": 0.5,
        },
        "high_recall": {
            "positive_bias": 7.0,
            "negative_bias": 6.0,
            "neutral_threshold": -0.3,
            "min_pos_threshold": 0.01,
            "min_neg_threshold": 0.01,
            "negation_factor": 1.2,
            "negation_window": 3,
            "context_weight": 1.1,
            "emoji_weight": 1.8,
            "domain_boost": 1.4,
            "arabic_boost": 1.1,
            "use_contextual_valence_shifters": True,
            "apply_dynamic_weighting": False,
            "prioritize_domain_matches": True,
            "aggressive_classification": True,
            "positive_weight": 2.5,
            "negative_weight": 3.5,
            "neutral_weight": 0.3,
        },
        "high_precision": {
            "positive_bias": 2.5,
            "negative_bias": 2.0,
            "neutral_threshold": 0.0,
            "min_pos_threshold": 0.15,
            "min_neg_threshold": 0.15,
            "negation_factor": 1.4,
            "negation_window": 4,
            "context_weight": 1.3,
            "emoji_weight": 1.3,
            "domain_boost": 1.2,
            "arabic_boost": 1.1,
            "use_contextual_valence_shifters": True,
            "apply_dynamic_weighting": True,
            "prioritize_domain_matches": True,
            "aggressive_classification": False,
            "positive_weight": 1.2,
            "negative_weight": 1.5,
            "neutral_weight": 0.8,
        }
    }
    
    # Create cache directory
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Log memory usage at start
    log_memory()
    
    # Perform hyperparameter tuning if requested
    if args.tune:
        find_best_parameters(
            limit=args.tune_limit,
            tuning_iterations=args.iterations,
            output_file=args.output + "_tuning",
            batch_size=args.batch_size,
            use_local_data=args.local,
            local_data_path=args.local_data
        )
        return
        
    print("\n" + "=" * 80)
    print("MEMORY-OPTIMIZED ARABIC SENTIMENT ANALYSIS")
    print("=" * 80)
    
    # Set up parameters
    params = None
    if args.preset:
        params = presets[args.preset]
        print(f"Using {args.preset} preset parameters")
    
    # Initialize analyzer
    analyzer = OptimizedArabicSentimentAnalyzer(
        collection_name=args.collection,
        qdrant_host=args.host,
        qdrant_port=args.port,
        force_new_analysis=True,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        params=params,
        memory_limit_mb=args.memory_limit,
        use_local_data=args.local,
        local_data_path=args.local_data
    )
    
    # Run analysis
    results = analyzer.run_analysis(
        limit=args.limit,
        output_file=args.output
    )
    
    # Check for errors
    if results.get("error"):
        print(f"\nERROR: {results['error']}")
        print("Make sure Qdrant is running or use --local with --local_data to use local data.")
        print("=" * 80)
        return
    
    # Print summary
    print("\n" + "=" * 80)
    print("SENTIMENT ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Total records processed: {results.get('results_count', 0)}")
    
    metrics = results.get('metrics', {})
    if metrics:
        print("\nSentiment distribution:")
        for sentiment, count in metrics.get('sentiment_counts', {}).items():
            percentage = metrics.get('sentiment_percentages', {}).get(sentiment, 0)
            print(f"  - {sentiment}: {count} ({percentage:.2f}%)")
        
        # Print evaluation metrics
        evaluation = metrics.get('evaluation', {})
        if evaluation.get('has_ground_truth', False):
            print("\nEvaluation metrics:")
            print(f"  - Records with ground truth: {evaluation.get('ground_truth_count', 0)}")
            print(f"  - Overall accuracy: {evaluation.get('accuracy', 0):.4f} ({evaluation.get('accuracy', 0)*100:.2f}%)")
            
            print("\nClass metrics:")
            for label, metrics_dict in evaluation.get('class_metrics', {}).items():
                print(f"  {label}:")
                print(f"    Precision: {metrics_dict.get('precision', 0):.4f}")
                print(f"    Recall: {metrics_dict.get('recall', 0):.4f}")
                print(f"    F1-score: {metrics_dict.get('f1', 0):.4f}")
    
    print("\nOutput files:")
    for file_pattern in results.get('output_files', []):
        print(f"  - {file_pattern}")
    
    print(f"\nTotal time: {results.get('elapsed_time', 0):.2f} seconds")
    print("=" * 80)
    
    # Final memory usage
    log_memory()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("If using Qdrant, make sure it's running or try using --generate_samples for local testing.")