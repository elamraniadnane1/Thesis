import os
import json
import argparse
import sys
import re
import math
import time
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
from pathlib import Path

import tqdm
from qdrant_client import QdrantClient

# Set to True to enable detailed logging
DEBUG = True

def log(message: str) -> None:
    """Print debug message if DEBUG is enabled."""
    if DEBUG:
        print(f"[DEBUG] {message}")

class ForcedSentimentAnalysis:
    """
    Extremely aggressive sentiment analysis that prioritizes
    non-neutral classifications to ensure better distribution.
    """
    
    def __init__(
        self,
        collection_name: str = "sentiment_analysis_dataset",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        force_new_analysis: bool = True,
        positive_bias: float = 2.0,
        negative_bias: float = 1.5,
        cache_dir: str = "./sentiment_cache"
    ):
        self.collection_name = collection_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Force new analysis by deleting existing results
        if force_new_analysis:
            results_file = self.cache_dir / f"{self.collection_name}_results.pkl"
            if results_file.exists():
                log(f"Deleting old results file: {results_file}")
                os.remove(results_file)
                
            chunks_pattern = f"{self.collection_name}_chunk_*.json"
            for chunk_file in self.cache_dir.glob(chunks_pattern):
                log(f"Deleting old chunk file: {chunk_file}")
                os.remove(chunk_file)
        
        # Set extremely aggressive biases to counter neutral dominance
        self.positive_bias = positive_bias
        self.negative_bias = negative_bias
        
        # Connect to Qdrant
        try:
            self.client = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                prefer_grpc=False,
                timeout=120.0
            )
        except Exception as e:
            log(f"Error connecting to Qdrant: {e}")
            self.client = None
            
        # Load all the necessary lexicons and resources
        self.positive_words = self.load_positive_words()
        self.negative_words = self.load_negative_words()
        self.intensifiers = self.load_intensifiers()
        self.negations = self.load_negations()
        self.sentiment_mapping = self.load_sentiment_mapping()
        
    def load_positive_words(self) -> Dict[str, float]:
        """Load positive words with very high scores."""
        return {
            # Arabic positive words - all with high scores
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
            "مزيان": 1.5,
            "زوين": 1.5,
            "بخير": 1.3,
            "عاجبني": 1.5,
            "مبروك": 1.4,
            "واعر": 1.8,
            "زوينة": 1.5,
            "بنين": 1.4,
            "مضبوط": 1.3,
            "تطوير": 1.2,
            "تنمية": 1.2,
            "تحسين": 1.2,
            "إصلاح": 1.2,
            "ترميم": 1.1,
            "تجديد": 1.1,
            "فعال": 1.3,
            "ناجح": 1.3,
            "شفافية": 1.3,
            "مساءلة": 1.2,
            "مشاركة": 1.2,
            "عدالة": 1.3,
            "مساواة": 1.3,
            "نظافة": 1.3,
            "نظيفة": 1.3,
            "آمنة": 1.3,
            "آمن": 1.3,
            "منظم": 1.2,
            "سريع": 1.2,
            "مجاني": 1.3,
            "رخيص": 1.2,
            "متاح": 1.2,
            "سهل": 1.2,
            "مريح": 1.2,
            "مفيد": 1.3,
            "متطور": 1.3,
            "حديث": 1.2,
            "مستدام": 1.3,
            
            # English positive words
            "good": 1.5,
            "great": 1.8,
            "excellent": 2.0,
            "amazing": 1.9,
            "perfect": 2.0,
            "wonderful": 1.9,
            "best": 1.8,
            "fantastic": 1.9,
            "awesome": 1.9,
            "brilliant": 1.8,
            "outstanding": 1.8,
            "superb": 1.9,
            "fabulous": 1.8,
            "terrific": 1.8,
            "nice": 1.3,
            "like": 1.2,
            "love": 1.7,
            "happy": 1.5,
            "glad": 1.4,
            "pleased": 1.4,
            "satisfied": 1.4,
            "helpful": 1.3,
            "useful": 1.3,
            "beneficial": 1.3,
            "positive": 1.3,
            "recommend": 1.5,
            "recommended": 1.5,
            "success": 1.5,
            "successful": 1.5,
            "effective": 1.3,
            "efficient": 1.3,
            "impressive": 1.4,
            "innovative": 1.3,
            "creative": 1.3,
            "beautiful": 1.4,
            "pleasant": 1.3,
            "enjoyable": 1.4,
            "convenient": 1.3,
            "easy": 1.3,
            "affordable": 1.3,
            "reliable": 1.3,
            "quality": 1.3,
            "valuable": 1.3,
            "worth": 1.3,
            "worthwhile": 1.3,
            "appealing": 1.3,
            "attractive": 1.3,
            "delightful": 1.4,
            "thrilled": 1.5,
            "excited": 1.4,
            "grateful": 1.4,
            "thankful": 1.4,
            "appreciate": 1.3,
            "appreciated": 1.3,
            "satisfying": 1.3,
            "favorable": 1.3,
            "promising": 1.3,
            "ideal": 1.4,
            "superior": 1.4,
            "exceptional": 1.5,
            "extraordinary": 1.5,
            "remarkable": 1.4,
            "splendid": 1.4,
            "magnificent": 1.5,
            "marvelous": 1.4,
            "top": 1.3,
            "premium": 1.3,
            "generous": 1.3,
            "smooth": 1.2,
            "comfortable": 1.3,
            "clean": 1.2,
            "fresh": 1.2,
            "safe": 1.3,
            "secure": 1.3,
            "trusted": 1.3,
            "authentic": 1.3,
            "genuine": 1.3,
            "proper": 1.2,
            "correct": 1.2,
            "suitable": 1.2,
            "appropriate": 1.2,
            "timely": 1.2,
            "prompt": 1.2,
            "fast": 1.2,
            "quick": 1.2,
            "speedy": 1.2,
            "rapid": 1.2,
            "instant": 1.2,
            "immediate": 1.2,
            "decent": 1.1,
            "better": 1.2,
            "improved": 1.2,
            "enhanced": 1.2,
            "upgraded": 1.2,
            "advanced": 1.2,
            "sophisticated": 1.2,
            "modern": 1.2,
            "cutting-edge": 1.3,
            "state-of-the-art": 1.3,
            "innovative": 1.3,
            "novel": 1.2,
            "unique": 1.2,
            "special": 1.2,
            "diverse": 1.1,
            "inclusive": 1.2,
            "fair": 1.2,
            "honest": 1.3,
            "transparent": 1.3,
            "ethical": 1.3,
            "moral": 1.2,
            "respectful": 1.3,
            "considerate": 1.2,
            "thoughtful": 1.2,
            "caring": 1.3,
            "supportive": 1.3,
            "encouraging": 1.3,
            "motivating": 1.3,
            "inspiring": 1.3,
            "uplifting": 1.3,
            "positive": 1.3,
            "optimistic": 1.3,
            "hopeful": 1.2,
            "forward-looking": 1.2,
            "progressive": 1.2,
            "sustainable": 1.3,
            "eco-friendly": 1.3,
            "green": 1.2,
            "natural": 1.1,
            "organic": 1.1,
            "healthy": 1.2,
            "fit": 1.1,
            "smart": 1.2,
            "intelligent": 1.2,
            "wise": 1.2,
            "clever": 1.2,
            "bright": 1.2,
            "talented": 1.2,
            "skilled": 1.2,
            "competent": 1.2,
            "capable": 1.2,
            "qualified": 1.2,
            "experienced": 1.2,
            "professional": 1.2,
            "expert": 1.3,
            "masterful": 1.3,
            "proficient": 1.2,
            "adept": 1.2,
            "accomplished": 1.3,
            "achieved": 1.2,
            "completed": 1.2,
            "finished": 1.1,
            "done": 1.0,
            "resolved": 1.2,
            "solved": 1.2,
            "fixed": 1.2,
            "repaired": 1.2,
            "improved": 1.2,
            "positive": 1.2,
            "plus": 1.0,
            "advantage": 1.2,
            "benefit": 1.2,
            "gain": 1.1,
            "profit": 1.1,
            "reward": 1.2,
            "bonus": 1.2,
            "perk": 1.1,
            "gift": 1.2,
            "treat": 1.1,
            "prize": 1.2,
            "win": 1.3,
            "victory": 1.3,
            "triumph": 1.3,
            "success": 1.3,
            "accomplishment": 1.3,
            "achievement": 1.3,
            
            # French positive words (sometimes used in Morocco)
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
    
    def load_negative_words(self) -> Dict[str, float]:
        """Load negative words with very high negative scores."""
        return {
            # Arabic negative words
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
            "خايب": -1.5,
            "ماشي مزيان": -1.4,
            "مشي زوين": -1.4,
            "غالي": -1.2,
            "واعر": -1.6,
            "بطيء": -1.2,
            "ماكاينش": -1.1,
            "ماكاين والو": -1.3,
            "مخاصرة": -1.3,
            "الرشوة": -1.6,
            "مكرفص": -1.4,
            "متلف": -1.4,
            "مقود": -1.5,
            "مسكين": -1.2,
            "معفن": -1.5,
            "خراب": -1.5,
            "مشوم": -1.4,
            "ما صالحش": -1.4,
            "فساد": -1.8,
            "رشوة": -1.8,
            "تأخير": -1.3,
            "تأخر": -1.3,
            "بطء": -1.3,
            "بطيء": -1.3,
            "بطيئة": -1.3,
            "انقطاع": -1.4,
            "تعطل": -1.4,
            "عطل": -1.4,
            "مشكلة": -1.3,
            "مشاكل": -1.3,
            "سوء": -1.4,
            "سيء": -1.4,
            "سيئة": -1.4,
            "رديء": -1.4,
            "رديئة": -1.4,
            "تلوث": -1.4,
            "ملوث": -1.4,
            "ملوثة": -1.4,
            "قذارة": -1.4,
            "قذر": -1.4,
            "قذرة": -1.4,
            "وسخ": -1.4,
            "وسخة": -1.4,
            "خطر": -1.4,
            "خطير": -1.4,
            "خطيرة": -1.4,
            "غير آمن": -1.4,
            "غير آمنة": -1.4,
            "مكلف": -1.2,
            "مكلفة": -1.2,
            "غالي": -1.2,
            "غالية": -1.2,
            "صعب": -1.2,
            "صعبة": -1.2,
            "معقد": -1.2,
            "معقدة": -1.2,
            "غير متوفر": -1.3,
            "غير متوفرة": -1.3,
            "نقص": -1.3,
            "ناقص": -1.3,
            "ناقصة": -1.3,
            "محدود": -1.1,
            "محدودة": -1.1,
            "ازدحام": -1.3,
            "مزدحم": -1.3,
            "مزدحمة": -1.3,
            "إهمال": -1.4,
            "مهمل": -1.4,
            "مهملة": -1.4,
            "ضعيف": -1.3,
            "ضعيفة": -1.3,
            "بيروقراطية": -1.3,
            "متأخر": -1.3,
            "غير مكتمل": -1.3,
            "متوقف": -1.4,
            "فاشل": -1.5,
            "معطل": -1.4,
            "مكلف": -1.2,
            "مبالغ فيه": -1.3,
            "تجاوز الميزانية": -1.3,
            "متعثر": -1.3,
            "متدهور": -1.4,
            "غير مستدام": -1.3,
            "غير فعال": -1.3,
            "غير مجدي": -1.3,
            "غير ضروري": -1.2,
            
            # English negative words
            "bad": -1.5,
            "poor": -1.4,
            "terrible": -1.8,
            "awful": -1.7,
            "horrible": -1.8,
            "worst": -1.9,
            "disappointing": -1.4,
            "disappointed": -1.4,
            "dissatisfied": -1.4,
            "unsatisfied": -1.4,
            "unhappy": -1.4,
            "sad": -1.3,
            "angry": -1.4,
            "upset": -1.3,
            "annoyed": -1.3,
            "irritated": -1.3,
            "frustrated": -1.4,
            "annoying": -1.3,
            "irritating": -1.3,
            "frustrating": -1.4,
            "useless": -1.5,
            "worthless": -1.5,
            "waste": -1.4,
            "wasted": -1.4,
            "rubbish": -1.6,
            "garbage": -1.6,
            "trash": -1.6,
            "junk": -1.5,
            "crap": -1.6,
            "shit": -1.7,
            "hell": -1.4,
            "damn": -1.3,
            "fucking": -1.6,
            "fuck": -1.6,
            "fail": -1.6,
            "failed": -1.6,
            "failure": -1.6,
            "problem": -1.3,
            "problematic": -1.3,
            "issue": -1.2,
            "issues": -1.2,
            "error": -1.3,
            "errors": -1.3,
            "mistake": -1.3,
            "mistakes": -1.3,
            "flaw": -1.3,
            "flaws": -1.3,
            "defect": -1.3,
            "defects": -1.3,
            "bug": -1.3,
            "bugs": -1.3,
            "glitch": -1.3,
            "glitches": -1.3,
            "broken": -1.4,
            "damaged": -1.4,
            "corrupt": -1.5,
            "corrupted": -1.5,
            "corruption": -1.6,
            "bribe": -1.6,
            "bribery": -1.6,
            "cheat": -1.5,
            "cheating": -1.5,
            "fraud": -1.6,
            "fraudulent": -1.6,
            "scam": -1.6,
            "fake": -1.4,
            "false": -1.3,
            "lie": -1.4,
            "lying": -1.4,
            "liar": -1.4,
            "dishonest": -1.4,
            "unethical": -1.4,
            "immoral": -1.4,
            "unfair": -1.3,
            "unjust": -1.4,
            "biased": -1.3,
            "discriminatory": -1.4,
            "racist": -1.5,
            "sexist": -1.5,
            "prejudiced": -1.4,
            "dangerous": -1.4,
            "hazardous": -1.4,
            "unsafe": -1.4,
            "risky": -1.3,
            "threatening": -1.4,
            "harmful": -1.4,
            "damaging": -1.4,
            "destructive": -1.4,
            "toxic": -1.5,
            "poisonous": -1.5,
            "deadly": -1.5,
            "fatal": -1.5,
            "lethal": -1.5,
            "dirty": -1.3,
            "filthy": -1.4,
            "disgusting": -1.5,
            "gross": -1.4,
            "nasty": -1.4,
            "unclean": -1.3,
            "contaminated": -1.4,
            "polluted": -1.4,
            "pollution": -1.4,
            "waste": -1.3,
            "wasteful": -1.3,
            "inefficient": -1.3,
            "ineffective": -1.3,
            "ineffectual": -1.3,
            "incompetent": -1.4,
            "incapable": -1.3,
            "inept": -1.4,
            "unprofessional": -1.3,
            "amateur": -1.2,
            "amateurish": -1.3,
            "sloppy": -1.3,
            "messy": -1.2,
            "disorganized": -1.2,
            "chaotic": -1.3,
            "confusing": -1.2,
            "confused": -1.2,
            "perplexing": -1.2,
            "complicated": -1.1,
            "complex": -1.0,
            "difficult": -1.1,
            "hard": -1.0,
            "challenging": -1.0,
            "tough": -1.1,
            "stressful": -1.3,
            "stressed": -1.2,
            "worried": -1.2,
            "worrying": -1.2,
            "concerning": -1.1,
            "alarming": -1.3,
            "disturbing": -1.3,
            "distressing": -1.3,
            "upsetting": -1.3,
            "shocking": -1.3,
            "appalling": -1.4,
            "outrageous": -1.4,
            "scandalous": -1.4,
            "offensive": -1.3,
            "insulting": -1.3,
            "rude": -1.3,
            "disrespectful": -1.3,
            "impolite": -1.2,
            "inconsiderate": -1.2,
            "thoughtless": -1.2,
            "selfish": -1.3,
            "greedy": -1.3,
            "stingy": -1.3,
            "cheap": -1.2,
            "expensive": -1.2,
            "overpriced": -1.3,
            "costly": -1.2,
            "pricey": -1.2,
            "unreasonable": -1.3,
            "absurd": -1.3,
            "ridiculous": -1.3,
            "ludicrous": -1.3,
            "nonsensical": -1.3,
            "stupid": -1.4,
            "idiotic": -1.4,
            "dumb": -1.4,
            "foolish": -1.3,
            "silly": -1.2,
            "pathetic": -1.4,
            "pitiful": -1.3,
            "lame": -1.3,
            "mediocre": -1.2,
            "subpar": -1.3,
            "substandard": -1.3,
            "inferior": -1.3,
            "inadequate": -1.3,
            "insufficient": -1.3,
            "lacking": -1.2,
            "limited": -1.1,
            "restricted": -1.1,
            "constrained": -1.1,
            "unavailable": -1.2,
            "inaccessible": -1.2,
            "unreachable": -1.2,
            "unattainable": -1.2,
            "impossible": -1.2,
            "impractical": -1.2,
            "unfeasible": -1.2,
            "unworkable": -1.2,
            "unrealistic": -1.2,
            "improbable": -1.1,
            "unlikely": -1.1,
            "doubtful": -1.1,
            "questionable": -1.1,
            "dubious": -1.1,
            "suspicious": -1.2,
            "sketchy": -1.2,
            "shady": -1.3,
            "dodgy": -1.3,
            "fishy": -1.2,
            "criminal": -1.5,
            "illegal": -1.5,
            "unlawful": -1.5,
            "illicit": -1.4,
            "forbidden": -1.3,
            "prohibited": -1.3,
            "banned": -1.3,
            "unacceptable": -1.3,
            "intolerable": -1.3,
            "unbearable": -1.3,
            "insufferable": -1.3,
            "miserable": -1.4,
            "wretched": -1.4,
            "desperate": -1.3,
            "hopeless": -1.4,
            "helpless": -1.3,
            "lost": -1.2,
            "confused": -1.2,
            "bewildered": -1.2,
            "disoriented": -1.2,
            "overwhelmed": -1.3,
            "overburdened": -1.3,
            "overworked": -1.3,
            "exhausted": -1.3,
            "tired": -1.2,
            "fatigued": -1.2,
            "weary": -1.2,
            "drained": -1.2,
            "weak": -1.2,
            "feeble": -1.2,
            "frail": -1.2,
            "sick": -1.3,
            "ill": -1.3,
            "diseased": -1.4,
            "infected": -1.4,
            "contagious": -1.3,
            "dying": -1.5,
            "dead": -1.5,
            "killed": -1.5,
            "murdered": -1.6,
            "slaughtered": -1.6,
            "destroyed": -1.5,
            "ruined": -1.5,
            "devastated": -1.5,
            "demolished": -1.5,
            "shattered": -1.4,
            "crushed": -1.4,
            "broken": -1.3,
            "cracked": -1.2,
            "split": -1.1,
            "torn": -1.2,
            "ripped": -1.2,
            "damaged": -1.3,
            "harmed": -1.3,
            "hurt": -1.3,
            "injured": -1.3,
            "wounded": -1.3,
            "pained": -1.3,
            "suffering": -1.4,
            "agony": -1.4,
            "torture": -1.5,
            "torment": -1.4,
            "anguish": -1.4,
            "distress": -1.3,
            "trouble": -1.2,
            "difficulty": -1.2,
            "hardship": -1.3,
            "struggle": -1.2,
            "suffering": -1.4,
            "negative": -1.2,
            "minus": -1.0,
            "disadvantage": -1.2,
            "drawback": -1.2,
            "downside": -1.2,
            "con": -1.1,
            "weakness": -1.2,
            "liability": -1.2,
            "danger": -1.3,
            "threat": -1.3,
            "risk": -1.2,
            "hazard": -1.3,
            "peril": -1.3,
            "jeopardy": -1.3,
            "vulnerability": -1.2,
            "exposed": -1.1,
            "unprotected": -1.2,
            "defenseless": -1.3,
            "powerless": -1.3,
            "impotent": -1.3,
            "useless": -1.4,
            "pointless": -1.3,
            "meaningless": -1.3,
            "senseless": -1.3,
            "needless": -1.2,
            "unnecessary": -1.2,
            "irrelevant": -1.2,
            "insignificant": -1.2,
            "trivial": -1.1,
            "minor": -1.0,
            "petty": -1.1,
            "small": -0.9,
            "tiny": -0.9,
            "little": -0.9,
            "diminutive": -0.9,
            "minimal": -0.9,
            "negligible": -1.0,
            "slight": -0.9,
            "subtle": -0.8,
            "vague": -1.0,
            "ambiguous": -1.0,
            "unclear": -1.1,
            "uncertain": -1.1,
            "unsure": -1.0,
            "hesitant": -1.0,
            "reluctant": -1.0,
            "unwilling": -1.1,
            "opposed": -1.2,
            "against": -1.1,
            "contrary": -1.1,
            "conflicting": -1.1,
            "contradictory": -1.1,
            "inconsistent": -1.1,
            "incompatible": -1.1,
            "mismatched": -1.0,
            "unsuitable": -1.1,
            "inappropriate": -1.2,
            "improper": -1.2,
            "indecent": -1.3,
            "offensive": -1.3,
            "vulgar": -1.3,
            "obscene": -1.4,
            "profane": -1.3,
            "blasphemous": -1.3,
            "sacrilegious": -1.3,
            "unholy": -1.3,
            "evil": -1.5,
            "wicked": -1.4,
            "sinful": -1.3,
            "immoral": -1.3,
            "depraved": -1.4,
            "corrupt": -1.4,
            "crooked": -1.3,
            "dishonest": -1.3,
            "deceitful": -1.3,
            "deceptive": -1.3,
            "misleading": -1.3,
            "false": -1.3,
            "fake": -1.3,
            "phony": -1.3,
            "bogus": -1.3,
            "sham": -1.3,
            "counterfeit": -1.3,
            "imitation": -1.2,
            "copy": -1.1,
            "replica": -1.0,
            "duplicate": -1.0,
            "clone": -1.0,
            "repetitive": -1.1,
            "monotonous": -1.2,
            "boring": -1.3,
            "dull": -1.2,
            "tedious": -1.2,
            "dreary": -1.2,
            "mundane": -1.1,
            "uninspiring": -1.2,
            "uninteresting": -1.2,
            "bland": -1.2,
            "insipid": -1.2,
            "tasteless": -1.2,
            "flavorless": -1.1,
            "plain": -1.0,
            "ordinary": -1.0,
            "common": -0.9,
            "conventional": -0.9,
            "traditional": -0.8,
            "old": -1.0,
            "ancient": -1.0,
            "outdated": -1.2,
            "obsolete": -1.3,
            "antiquated": -1.2,
            "primitive": -1.2,
            "backward": -1.2,
            "regressive": -1.3,
            "retrograde": -1.2,
            "reactionary": -1.2,
            "conservative": -1.0,
            "rigid": -1.1,
            "inflexible": -1.2,
            "stubborn": -1.2,
            "obstinate": -1.2,
            "unyielding": -1.1,
            "intransigent": -1.1,
            "uncompromising": -1.1,
            "demanding": -1.1,
            "exacting": -1.1,
            "strict": -1.1,
            "harsh": -1.2,
            "severe": -1.2,
            "brutal": -1.4,
            "cruel": -1.4,
            "vicious": -1.4,
            "savage": -1.4,
            "barbaric": -1.4,
            "inhuman": -1.4,
            "inhumane": -1.4,
            "sadistic": -1.5,
            "heartless": -1.4,
            "merciless": -1.4,
            "ruthless": -1.4,
            "remorseless": -1.4,
            "pitiless": -1.4,
            "soulless": -1.4,
            "cold": -1.2,
            "frigid": -1.2,
            "icy": -1.2,
            "frosty": -1.1,
            "frozen": -1.1,
            "bitter": -1.3,
            "acrid": -1.2,
            "acidic": -1.2,
            "caustic": -1.3,
            "corrosive": -1.3,
            "abrasive": -1.2,
            "rough": -1.1,
            "rugged": -1.0,
            "bumpy": -1.0,
            "uneven": -1.0,
            "irregular": -1.0,
            "asymmetrical": -1.0,
            "unbalanced": -1.1,
            "unstable": -1.2,
            "precarious": -1.2,
            "wobbly": -1.1,
            "shaky": -1.1,
            "trembling": -1.1,
            "vibrating": -1.0,
            "noisy": -1.1,
            "loud": -1.1,
            "deafening": -1.2,
            "thunderous": -1.1,
            "booming": -1.1,
            "blaring": -1.2,
            "shrieking": -1.3,
            "screeching": -1.3,
            "screaming": -1.3,
            "yelling": -1.2,
            "shouting": -1.1,
            "angry": -1.3,
            "furious": -1.4,
            "enraged": -1.4,
            "infuriated": -1.4,
            "irate": -1.3,
            "livid": -1.3,
            "seething": -1.3,
            "boiling": -1.2,
            "heated": -1.1,
            "hot": -1.0,
            "fiery": -1.1,
            "burning": -1.1,
            "scalding": -1.2,
            "searing": -1.2,
            "scorching": -1.2,
            "blistering": -1.2,
            "painful": -1.3,
            "aching": -1.2,
            "sore": -1.2,
            "tender": -1.1,
            "sensitive": -1.0,
            "raw": -1.1,
            "exposed": -1.1,
            "vulnerable": -1.1,
            "fragile": -1.1,
            "delicate": -1.0,
            "brittle": -1.1,
            "breakable": -1.1,
            "crumbly": -1.1,
            "disintegrating": -1.2,
            "decomposing": -1.3,
            "rotting": -1.3,
            "decaying": -1.3,
            "putrid": -1.4,
            "rancid": -1.3,
            "moldy": -1.3,
            "stale": -1.2,
            "musty": -1.2,
            "fusty": -1.2,
            "stinky": -1.3,
            "smelly": -1.3,
            "malodorous": -1.3,
            "noxious": -1.3,
            "foul": -1.3,
            "fetid": -1.3,
            "offensive": -1.3,
            "repulsive": -1.4,
            "repugnant": -1.4,
            "revolting": -1.4,
            "disgusting": -1.4,
            "sickening": -1.3,
            "nauseating": -1.3,
            "gut-wrenching": -1.3,
            "stomach-turning": -1.3,
            "vomit-inducing": -1.4,
            
            # French negative words (sometimes used in Morocco)
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
            "coûteux": -1.2,
            "sale": -1.3,
            "malpropre": -1.3,
            "dangereux": -1.3,
            "risqué": -1.2,
            "pauvre": -1.2,
            "misérable": -1.3,
            "triste": -1.3,
            "malheureux": -1.3,
            "fâché": -1.3,
            "en colère": -1.3,
            "frustré": -1.3,
            "déçu": -1.3,
            "insatisfait": -1.3,
            "malheureusement": -1.2,
        }
    
    def load_intensifiers(self) -> Dict[str, float]:
        """Load intensifier words that increase sentiment strength."""
        return {
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
            "utterly": 1.9,
            "greatly": 1.7,
            "highly": 1.7,
            "intensely": 1.8,
            "exceedingly": 1.8,
            "exceptionally": 1.8,
            "extraordinarily": 1.9,
            "remarkably": 1.7,
            "notably": 1.6,
            "significantly": 1.6,
            "considerably": 1.6,
            "substantially": 1.6,
            "tremendously": 1.8,
            "immensely": 1.8,
            "hugely": 1.7,
            "vastly": 1.7,
            "massively": 1.7,
            "incredibly": 1.8,
            "unbelievably": 1.8,
            "amazingly": 1.7,
            "astonishingly": 1.7,
            "phenomenally": 1.8,
            "profoundly": 1.7,
            "deeply": 1.6,
            "thoroughly": 1.6,
            "truly": 1.6,
            "genuinely": 1.6,
            "undoubtedly": 1.6,
            "unquestionably": 1.6,
            "undeniably": 1.6,
            "indisputably": 1.6,
            "definitely": 1.5,
            "certainly": 1.5,
            "surely": 1.5,
            "indeed": 1.5,
            "literally": 1.6,
            "practically": 1.5,
            "virtually": 1.5,
            "essentially": 1.5,
            "fundamentally": 1.5,
            "desperately": 1.7,
            "terribly": 1.7,
            "awfully": 1.7,
            "dreadfully": 1.7,
            "horribly": 1.7,
            "miserably": 1.7,
            "woefully": 1.7,
            "painfully": 1.6,
            "bitterly": 1.6,
            "severely": 1.6,
            "gravely": 1.6,
            "seriously": 1.5,
            "critically": 1.6,
            "crucially": 1.5,
            "vitally": 1.5,
            "importantly": 1.4,
            "significantly": 1.5,
            "notably": 1.4,
            "markedly": 1.4,
            "strikingly": 1.5,
            "particularly": 1.4,
            "especially": 1.5,
            "specifically": 1.3,
            "precisely": 1.3,
            "exactly": 1.3,
            "perfectly": 1.6,
            "ideally": 1.5,
            "optimally": 1.5,
            "flawlessly": 1.6,
            "impeccably": 1.6,
            "supremely": 1.7,
            "superbly": 1.6,
            "excellently": 1.6,
            "brilliantly": 1.6,
            "wonderfully": 1.6,
            "fantastically": 1.7,
            "magnificently": 1.7,
            "gloriously": 1.7,
            "splendidly": 1.6,
            "beautifully": 1.5,
            "gorgeously": 1.6,
            "exquisitely": 1.6,
            "delightfully": 1.5,
            "pleasantly": 1.4,
            "satisfyingly": 1.5,
            "gratifyingly": 1.5,
            "admirably": 1.5,
            "commendably": 1.5,
            "impressively": 1.5,
            "spectacularly": 1.7,
            "dramatically": 1.6,
            "radically": 1.6,
            "profoundly": 1.6,
            "immeasurably": 1.7,
            "infinitely": 1.8,
            "endlessly": 1.7,
            "eternally": 1.7,
            "perpetually": 1.6,
            "persistently": 1.5,
            "consistently": 1.4,
            "constantly": 1.5,
            "continually": 1.5,
            "continuously": 1.5,
            "relentlessly": 1.6,
            "incessantly": 1.6,
            "ceaselessly": 1.6,
            "tirelessly": 1.5,
            "unfailingly": 1.5,
            "unerringly": 1.5,
            "invariably": 1.4,
            "reliably": 1.3,
            "dependably": 1.3,
            "successfully": 1.4,
            "effectively": 1.4,
            "efficiently": 1.4,
            "productively": 1.4,
            "fruitfully": 1.4,
            "beneficially": 1.4,
            "advantageously": 1.4,
            "favorably": 1.4,
            "positively": 1.4,
            "affirmatively": 1.3,
            "wholeheartedly": 1.6,
            "passionately": 1.6,
            "enthusiastically": 1.5,
            "eagerly": 1.4,
            "zealously": 1.5,
            "fervently": 1.5,
            "ardently": 1.5,
            "intensely": 1.5,
            "fiercely": 1.5,
            "vehemently": 1.5,
            "vigorously": 1.4,
            "forcefully": 1.4,
            "powerfully": 1.5,
            "strongly": 1.5,
            "robustly": 1.4,
            "solidly": 1.3,
            "firmly": 1.3,
            "steadfastly": 1.3,
            "resolutely": 1.3,
            "determinedly": 1.4,
            "decisively": 1.3,
            "conclusively": 1.3,
            "definitively": 1.3,
            "unequivocally": 1.4,
            "unmistakably": 1.4,
            "explicitly": 1.3,
            "clearly": 1.2,
            "plainly": 1.2,
            "obviously": 1.3,
            "evidently": 1.2,
            "apparently": 1.1,
            "seemingly": 1.1,
            "ostensibly": 1.1,
            "reputedly": 1.1,
            "allegedly": 1.1,
            "supposedly": 1.1,
            "presumably": 1.1,
            "likely": 1.1,
            "probably": 1.1,
            "possibly": 1.0,
            "potentially": 1.0,
            "conceivably": 1.0,
            "imaginably": 1.0,
            "hopefully": 1.0,
            "ideally": 1.1,
            "optimistically": 1.1,
            "positively": 1.2,
            "affirmatively": 1.1,
            "constructively": 1.1,
            "productively": 1.1,
            "effectively": 1.1,
            "efficiently": 1.1,
            "successfully": 1.2,
            "triumphantly": 1.3,
            "victoriously": 1.3,
            "gloriously": 1.3,
            "magnificently": 1.3,
            "spectacularly": 1.3,
            
            # French intensifiers
            "très": 1.8,
            "extrêmement": 2.0,
            "vraiment": 1.7,
            "complètement": 1.8,
            "totalement": 1.8,
            "absolument": 1.9,
            "parfaitement": 1.7,
            "entièrement": 1.7,
            "tellement": 1.6,
            "si": 1.5,
            "trop": 1.5,
            "beaucoup": 1.5,
            "énormément": 1.8,
            "incroyablement": 1.8,
            "terriblement": 1.7,
            "horriblement": 1.7,
            "affreusement": 1.7,
            "fortement": 1.6,
            "gravement": 1.6,
        }
    
    def load_negations(self) -> List[str]:
        """Load negation words that flip sentiment."""
        return [
            # Arabic negations
            "لا", "لم", "لن", "ليس", "ليست", "لسنا", "لستم", "لسن", "لست", "لسن", 
            "ما", "غير", "بلا", "بدون", "ما كاين", "ماكاين", "ماكاينش", "مكاينش",
            "مش", "مشي", "ماشي", "مو", "موش", "ماهو", "ماهي",
            
            # English negations
            "not", "no", "never", "without", "none", "neither", "nor", "nothing",
            "nobody", "nowhere", "barely", "hardly", "scarcely", "seldom", "rarely",
            "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot",
            "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't", "hasn't",
            "haven't", "hadn't", "doesn't", "don't", "didn't",
            
            # French negations
            "ne", "pas", "non", "sans", "jamais", "aucun", "aucune", "ni", "personne", "rien",
        ]
    
    def load_sentiment_mapping(self) -> Dict[str, str]:
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
            0: "neutral",
            "3": "neutral",  # In 5-point scale, 3 is neutral
            3: "neutral",
        }
    
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
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform extremely aggressive sentiment analysis to avoid neutral bias.
        This function is designed to classify more aggressively into positive
        and negative categories.
        """
        if not text:
            return {"label": "neutral", "score": 0.5}
            
        # Normalize and clean text
        text = text.lower() if any(c.isascii() for c in text) else text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into words
        words = text.split()
        
        # Detect negations
        negation_positions = []
        for i, word in enumerate(words):
            if word in self.negations:
                # Mark this position and a few words after as negated
                negation_positions.extend(range(i, min(i+4, len(words))))
        
        # Calculate sentiment scores
        pos_score = 0.0
        neg_score = 0.0
        pos_words = []
        neg_words = []
        intensifier_multiplier = 1.0
        
        for i, word in enumerate(words):
            # Check for intensifiers
            if word in self.intensifiers:
                intensifier_multiplier = self.intensifiers[word]
                continue
                
            # Check positive lexicon
            if word in self.positive_words:
                word_score = self.positive_words[word] * intensifier_multiplier
                # If in negation context, flip the sentiment
                if i in negation_positions:
                    neg_score += word_score
                    neg_words.append(word)
                else:
                    pos_score += word_score
                    pos_words.append(word)
                # Reset intensifier
                intensifier_multiplier = 1.0
                
            # Check negative lexicon
            elif word in self.negative_words:
                word_score = abs(self.negative_words[word]) * intensifier_multiplier
                # If in negation context, flip the sentiment
                if i in negation_positions:
                    pos_score += word_score
                    pos_words.append(word)
                else:
                    neg_score += word_score
                    neg_words.append(word)
                # Reset intensifier
                intensifier_multiplier = 1.0
        
        # Apply biases to counter neutral dominance
        pos_score *= self.positive_bias
        neg_score *= self.negative_bias
        
        # Calculate net sentiment
        total_score = pos_score - neg_score
        
        # Determine sentiment with very low thresholds
        # This is key to avoid the "everything is neutral" problem
        if total_score > 0.01:  # Very low threshold for positive
            sentiment = "positive"
            confidence = min(0.5 + (pos_score / 5.0), 1.0)
        elif total_score < -0.01:  # Very low threshold for negative
            sentiment = "negative"
            confidence = min(0.5 + (neg_score / 5.0), 1.0)
        else:
            # If truly no sentiment is detected, mark as neutral
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "label": sentiment,
            "score": confidence,
            "pos_score": pos_score,
            "neg_score": neg_score,
            "total_score": total_score,
            "pos_words": pos_words,
            "neg_words": neg_words,
            "has_negation": len(negation_positions) > 0
        }
    
    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load data from Qdrant collection."""
        log(f"Loading data from collection: {self.collection_name}")
        
        # Check for existing cache files
        records_cache = self.cache_dir / f"{self.collection_name}_records_all.json"
        if records_cache.exists():
            try:
                with open(records_cache, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                    log(f"Loaded {len(records)} records from cache")
                    if limit:
                        return records[:limit]
                    return records
            except Exception as e:
                log(f"Error loading cache: {e}")
        
        # If no cache or error, load from Qdrant
        if not self.client:
            log("No Qdrant client available")
            return []
            
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            total_count = self.client.count(self.collection_name).count
            log(f"Found {total_count} records in collection")
            
            # Load records in batches
            records = []
            offset = None
            batch_size = 500
            loaded = 0
            
            while loaded < (limit or total_count):
                # Scroll through records
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=min(batch_size, (limit or total_count) - loaded),
                    offset=offset
                )
                
                if isinstance(result, tuple):
                    points, offset = result
                else:
                    points = result.points
                    offset = result.next_page_offset
                
                if not points:
                    break
                    
                # Process points
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
                    
                    # Create record
                    record = {
                        'id': str(point_id),
                        'text': text,
                        'ground_truth': ground_truth
                    }
                    
                    records.append(record)
                
                # Update loaded count
                loaded += len(points)
                log(f"Loaded {loaded} records so far")
                
                # Check if we're done
                if offset is None or loaded >= (limit or total_count):
                    break
            
            # Save to cache
            with open(records_cache, 'w', encoding='utf-8') as f:
                json.dump(records, f)
            
            log(f"Saved {len(records)} records to cache")
            return records
            
        except Exception as e:
            log(f"Error loading data from Qdrant: {e}")
            return []
    
    def analyze_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment for all records."""
        log(f"Analyzing sentiment for {len(records)} records")
        
        results = []
        for i, record in enumerate(tqdm.tqdm(records, desc="Analyzing sentiment")):
            # Analyze text
            sentiment_result = self.analyze_text(record['text'])
            
            # Create result record
            result = record.copy()
            result['predicted_sentiment'] = sentiment_result['label']
            result['sentiment_score'] = sentiment_result['score']
            result['sentiment_details'] = sentiment_result
            
            # Add comparison with ground truth
            if result.get('ground_truth') is not None:
                result['correct'] = result['ground_truth'] == result['predicted_sentiment']
            else:
                result['correct'] = None
            
            results.append(result)
            
            # Log progress
            if (i+1) % 1000 == 0:
                log(f"Processed {i+1} records")
        
        return results
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute metrics from analysis results."""
        log("Computing metrics")
        
        # Count distribution of predictions
        prediction_counts = Counter()
        for result in results:
            sentiment = result.get('predicted_sentiment')
            if sentiment:
                prediction_counts[sentiment] += 1
        
        # Calculate percentages
        total = len(results)
        percentages = {k: (v / total) * 100 for k, v in prediction_counts.items()}
        
        # Evaluation metrics
        eval_metrics = {
            "has_ground_truth": False,
            "accuracy": None,
            "confusion_matrix": None,
            "class_metrics": None
        }
        
        # Filter records with ground truth
        records_with_ground_truth = [r for r in results if r.get('ground_truth') is not None]
        
        if records_with_ground_truth:
            eval_metrics["has_ground_truth"] = True
            eval_metrics["ground_truth_count"] = len(records_with_ground_truth)
            
            # Get ground truth and predictions
            y_true = [r['ground_truth'] for r in records_with_ground_truth]
            y_pred = [r['predicted_sentiment'] for r in records_with_ground_truth]
            
            # Calculate accuracy
            correct = sum(1 for r in records_with_ground_truth if r.get('correct', False))
            eval_metrics["accuracy"] = correct / len(records_with_ground_truth)
            
            # Create confusion matrix
            labels = sorted(list(set(y_true + y_pred)))
            confusion = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}
            
            for true, pred in zip(y_true, y_pred):
                confusion[true][pred] += 1
            
            eval_metrics["confusion_matrix"] = confusion
            eval_metrics["labels"] = labels
            
            # Calculate class metrics
            class_metrics = {}
            for label in labels:
                # Calculate true positives, false positives, false negatives
                tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
                fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
                fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
                
                # Calculate precision, recall, F1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics[label] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": sum(1 for t in y_true if t == label)
                }
            
            eval_metrics["class_metrics"] = class_metrics
            
            # Calculate ground truth distribution
            ground_truth_counts = Counter(y_true)
            eval_metrics["ground_truth_distribution"] = {
                "counts": dict(ground_truth_counts),
                "percentages": {k: (v / len(y_true)) * 100 for k, v in ground_truth_counts.items()}
            }
        
        return {
            "total_records": total,
            "sentiment_counts": dict(prediction_counts),
            "sentiment_percentages": percentages,
            "evaluation": eval_metrics
        }
    
    def save_results(self, results: List[Dict[str, Any]], metrics: Dict[str, Any], output_file: str) -> None:
        """Save results to files."""
        log(f"Saving results to {output_file}")
        
        # Save metrics
        metrics_file = f"{output_file}_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        # Save results in chunks
        chunk_size = 1000
        for i in range(0, len(results), chunk_size):
            chunk = results[i:i+chunk_size]
            chunk_file = f"{output_file}_chunk_{i//chunk_size}.json"
            
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, indent=2)
        
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
        
        log(f"Saved results to {metrics_file}, {report_file}, and chunk files")
    
    def run_analysis(self, limit: Optional[int] = None, output_file: str = "sentiment_results") -> Dict[str, Any]:
        """Run the complete sentiment analysis pipeline."""
        start_time = time.time()
        
        # Load data
        records = self.load_data(limit=limit)
        
        if not records:
            log("No records found")
            return {"error": "No records found"}
        
        # Analyze sentiment
        results = self.analyze_records(records)
        
        # Compute metrics
        metrics = self.compute_metrics(results)
        
        # Save results
        self.save_results(results, metrics, output_file)
        
        # Return summary
        end_time = time.time()
        elapsed = end_time - start_time
        
        return {
            "results_count": len(results),
            "metrics": metrics,
            "elapsed_time": elapsed,
            "output_files": [
                f"{output_file}_metrics.json",
                f"{output_file}_report.txt",
                f"{output_file}_chunk_*.json"
            ]
        }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Forced sentiment analysis')
    parser.add_argument('--host', default='localhost', help='Qdrant host')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port')
    parser.add_argument('--collection', default='sentiment_analysis_dataset', help='Collection name')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of records to process')
    parser.add_argument('--output', default='sentiment_results', help='Output file prefix')
    parser.add_argument('--cache_dir', default='./sentiment_cache', help='Directory for caching results')
    parser.add_argument('--positive_bias', type=float, default=2.0, help='Bias factor for positive sentiment')
    parser.add_argument('--negative_bias', type=float, default=1.5, help='Bias factor for negative sentiment')
    parser.add_argument('--no_force', action='store_true', help='Do not force new analysis (use existing results if available)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("FORCED SENTIMENT ANALYSIS")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = ForcedSentimentAnalysis(
        collection_name=args.collection,
        qdrant_host=args.host,
        qdrant_port=args.port,
        force_new_analysis=not args.no_force,
        positive_bias=args.positive_bias,
        negative_bias=args.negative_bias,
        cache_dir=args.cache_dir
    )
    
    # Run analysis
    results = analyzer.run_analysis(
        limit=args.limit,
        output_file=args.output
    )
    
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

if __name__ == "__main__":
    main()