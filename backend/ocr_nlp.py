"""
OCR + NLP PIPELINE
Extraction de texte (EasyOCR) + Keywords (TF-IDF)
"""

import logging
from typing import List, Dict
import re
from collections import Counter
import io
from PIL import Image

logger = logging.getLogger(__name__)

class OCRNLPPipeline:
    """
    Pipeline OCR + NLP pour extraction de métadonnées
    """
    
    def __init__(self):
        self.ocr_reader = None
        self._init_ocr()
        
        # Stop words français (pour filtrage keywords)
        self.stop_words = set([
            'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou', 
            'mais', 'donc', 'car', 'à', 'au', 'aux', 'pour', 'par', 'sur',
            'dans', 'ce', 'cette', 'ces', 'qui', 'que', 'quoi', 'dont', 'où',
            'se', 'sa', 'son', 'ses', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes',
            'il', 'elle', 'ils', 'elles', 'nous', 'vous', 'je', 'tu', 'on',
            'est', 'sont', 'a', 'ont', 'fait', 'être', 'avoir', 'faire',
            'plus', 'moins', 'très', 'bien', 'mal', 'tout', 'tous', 'toute'
        ])
    
    def _init_ocr(self):
        """Initialise EasyOCR (lazy loading)"""
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['fr', 'en'], gpu=False)
            logger.info("✅ EasyOCR initialized (French + English)")
        except ImportError:
            logger.warning("⚠️ EasyOCR not installed. OCR disabled.")
            logger.info("Install with: pip install easyocr")
        except Exception as e:
            logger.warning(f"⚠️ EasyOCR initialization failed: {e}")
    
    def extract_text(self, image_bytes: bytes) -> Dict:
        """
        Extrait le texte d'une image avec EasyOCR
        
        Returns:
            {'text': str, 'confidence': float, 'detected_blocks': int}
        """
        if self.ocr_reader is None:
            return {
                'text': '',
                'confidence': 0.0,
                'detected_blocks': 0,
                'error': 'OCR not available'
            }
        
        try:
            # Convertir bytes en image PIL
            image = Image.open(io.BytesIO(image_bytes))
            
            # OCR
            results = self.ocr_reader.readtext(image)
            
            # Extraire texte et confiances
            texts = []
            confidences = []
            
            for (bbox, text, conf) in results:
                texts.append(text)
                confidences.append(conf)
            
            full_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            logger.info(f"📄 OCR extracted {len(results)} text blocks")
            
            return {
                'text': full_text,
                'confidence': round(avg_confidence, 3),
                'detected_blocks': len(results)
            }
            
        except Exception as e:
            logger.error(f"❌ OCR extraction failed: {str(e)}")
            return {
                'text': '',
                'confidence': 0.0,
                'detected_blocks': 0,
                'error': str(e)
            }
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """
        Extrait les mots-clés importants du texte
        Méthode simple mais efficace: fréquence + filtrage
        
        Args:
            text: Texte à analyser
            top_k: Nombre de keywords à retourner
        
        Returns:
            Liste des top keywords
        """
        if not text or len(text.strip()) == 0:
            return []
        
        try:
            # Nettoyer et tokeniser
            text = text.lower()
            # Garder uniquement lettres et espaces
            text = re.sub(r'[^a-zàâäéèêëïîôùûüÿç\s]', ' ', text)
            words = text.split()
            
            # Filtrer stop words et mots courts
            words = [w for w in words if w not in self.stop_words and len(w) > 3]
            
            if not words:
                return []
            
            # Compter fréquences
            word_freq = Counter(words)
            
            # Top K mots
            top_words = [word for word, count in word_freq.most_common(top_k)]
            
            logger.info(f"🔑 Extracted {len(top_words)} keywords")
            return top_words
            
        except Exception as e:
            logger.error(f"❌ Keyword extraction failed: {str(e)}")
            return []
    
    def analyze_document_type(self, text: str) -> Dict[str, float]:
        """
        Analyse le texte pour détecter le type de document
        Retourne des scores pour chaque classe basés sur keywords
        
        Cette fonction peut améliorer la prédiction du CNN
        """
        text_lower = text.lower()
        
        # Patterns par type de document
        patterns = {
            'Invoice': [
                'facture', 'invoice', 'montant', 'total', 'tva', 'ht', 'ttc',
                'paiement', 'échéance', 'référence', 'n°', 'client'
            ],
            'Receipt': [
                'reçu', 'receipt', 'ticket', 'caisse', 'CB', 'espèces',
                'merci', 'au revoir', 'magasin', 'commerce'
            ],
            'Report': [
                'rapport', 'report', 'analyse', 'résultats', 'conclusion',
                'introduction', 'méthodologie', 'données', 'étude', 'synthèse'
            ],
            'Drawing': [
                'plan', 'schéma', 'dessin', 'technique', 'échelle', 'vue',
                'coupe', 'détail', 'dimension', 'référence technique'
            ]
        }
        
        scores = {}
        
        for doc_type, keywords in patterns.items():
            # Compter combien de keywords sont présents
            matches = sum(1 for kw in keywords if kw in text_lower)
            # Score normalisé
            score = matches / len(keywords)
            scores[doc_type] = round(score, 3)
        
        return scores
    
    def extract_metadata(self, text: str) -> Dict:
        """
        Extrait des métadonnées structurées du texte
        (dates, montants, références, etc.)
        """
        metadata = {
            'has_date': False,
            'has_amount': False,
            'has_reference': False,
            'detected_dates': [],
            'detected_amounts': [],
            'detected_references': []
        }
        
        try:
            # Détecter dates (formats simples)
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
            dates = re.findall(date_pattern, text)
            if dates:
                metadata['has_date'] = True
                metadata['detected_dates'] = dates[:3]  # Max 3
            
            # Détecter montants (€, EUR, $)
            amount_pattern = r'\b\d+[.,]\d{2}\s*[€$]|\b\d+\s*EUR\b'
            amounts = re.findall(amount_pattern, text)
            if amounts:
                metadata['has_amount'] = True
                metadata['detected_amounts'] = amounts[:3]
            
            # Détecter références (N°, REF, etc.)
            ref_pattern = r'(?:N°|REF|Reference|Réf\.?)\s*:?\s*([A-Z0-9-]+)'
            refs = re.findall(ref_pattern, text, re.IGNORECASE)
            if refs:
                metadata['has_reference'] = True
                metadata['detected_references'] = refs[:3]
            
        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
        
        return metadata
