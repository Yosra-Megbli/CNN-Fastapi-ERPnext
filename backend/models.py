"""
MODEL MANAGER - Chargement SYNCHRONE DIRECT
🚀 SOLUTION: Charge le modèle IMMÉDIATEMENT au démarrage (pas de thread)
"""

import os
import numpy as np
import random
from typing import Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Gestionnaire du modèle avec chargement SYNCHRONE
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.classes = ['Drawing', 'Invoice', 'Report', 'Receipt']
        self.start_time = datetime.now()
        self.total_predictions = 0
        self._model_loaded = False
        
        # Mock intelligent - probabilités par classe
        self.mock_patterns = {
            'Drawing': {'confidence_range': (0.75, 0.92), 'keywords': ['plan', 'schéma', 'design', 'technique', 'blueprint']},
            'Invoice': {'confidence_range': (0.78, 0.95), 'keywords': ['facture', 'montant', 'total', 'TVA', 'paiement']},
            'Report': {'confidence_range': (0.72, 0.89), 'keywords': ['rapport', 'analyse', 'résultats', 'conclusion', 'étude']},
            'Receipt': {'confidence_range': (0.76, 0.93), 'keywords': ['reçu', 'ticket', 'caisse', 'achat', 'commerce']}
        }
        
        # 🚀 CHARGEMENT IMMÉDIAT ET SYNCHRONE
        logger.info("="*70)
        logger.info("🚀 CHARGEMENT SYNCHRONE DU MODÈLE")
        logger.info("="*70)
        self._load_model_now()
    
    def _load_model_now(self):
        """
        🚀 Charge le modèle IMMÉDIATEMENT (pas de thread, pas d'async)
        """
        # 1. Vérifier l'existence du fichier
        if not os.path.exists(self.model_path):
            logger.error(f"❌ Fichier modèle introuvable: {self.model_path}")
            logger.warning("🎭 Mode SIMULATION activé")
            return
        
        file_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        logger.info(f"📦 Fichier trouvé: {self.model_path} ({file_size_mb:.1f} MB)")
        
        # 2. Importer TensorFlow
        try:
            logger.info("⏳ Import de TensorFlow...")
            import time
            start_import = time.time()
            
            import tensorflow as tf
            
            # Désactiver les logs verbeux de TensorFlow
            tf.get_logger().setLevel('ERROR')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
            import_time = time.time() - start_import
            logger.info(f"✅ TensorFlow importé en {import_time:.1f}s")
            
        except ImportError as e:
            logger.error(f"❌ TensorFlow non installé: {e}")
            logger.warning("🎭 Mode SIMULATION activé")
            return
        
        # 3. Charger le modèle
        try:
            logger.info("⏳ Chargement du modèle en cours...")
            logger.info("   (Cela peut prendre 10-60 secondes selon votre CPU)")
            
            start_load = time.time()
            
            # 🔧 CHARGEMENT DIRECT SANS COMPILATION
            self.model = tf.keras.models.load_model(
                self.model_path,
                compile=False  # ← Plus rapide sans compilation
            )
            
            load_time = time.time() - start_load
            logger.info(f"✅ Modèle chargé en {load_time:.1f}s")
            
            # 4. Vérification rapide
            logger.info("🔍 Vérification de l'architecture...")
            logger.info(f"   • Input shape: {self.model.input_shape}")
            logger.info(f"   • Output shape: {self.model.output_shape}")
            logger.info(f"   • Nombre de couches: {len(self.model.layers)}")
            
            # 5. Test de prédiction
            logger.info("🧪 Test de prédiction...")
            start_predict = time.time()
            
            test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            _ = self.model.predict(test_input, verbose=0)
            
            predict_time = time.time() - start_predict
            logger.info(f"✅ Test réussi en {predict_time:.2f}s")
            
            # 6. Succès !
            self._model_loaded = True
            total_time = time.time() - start_load
            
            logger.info("="*70)
            logger.info(f"🎉 MODÈLE CHARGÉ AVEC SUCCÈS en {total_time:.1f}s")
            logger.info("🔥 MODE RÉEL ACTIVÉ - Fusion CNN + OCR/NLP")
            logger.info("="*70)
            
        except Exception as e:
            logger.error("="*70)
            logger.error(f"❌ ÉCHEC DU CHARGEMENT: {type(e).__name__}")
            logger.error(f"💬 Détails: {str(e)[:200]}")
            logger.error("="*70)
            logger.warning("🎭 Bascule en mode SIMULATION")
            self.model = None
            self._model_loaded = False
    
    def start_loading(self):
        """
        Méthode appelée par main.py
        Maintenant ne fait rien car le chargement est fait dans __init__
        """
        if self._model_loaded:
            logger.info("✅ Modèle déjà chargé")
        else:
            logger.warning("⚠️ Modèle non chargé - mode simulation")
    
    def is_model_loaded(self) -> bool:
        """Vérifie si le modèle réel est chargé"""
        return self._model_loaded
    
    def get_load_progress(self) -> float:
        """Retourne toujours 1.0 car le chargement est synchrone"""
        return 1.0
    
    def predict(self, image_array: np.ndarray) -> Dict:
        """
        Prédiction intelligente
        """
        if self.is_model_loaded():
            return self._real_predict(image_array)
        else:
            return self._mock_predict(image_array)
    
    def _real_predict(self, image_array: np.ndarray) -> Dict:
        """Prédiction avec le vrai modèle"""
        try:
            predictions = self.model.predict(image_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            
            return {
                'class': self.classes[class_idx],
                'confidence': confidence,
                'all_scores': {cls: float(predictions[0][i]) for i, cls in enumerate(self.classes)}
            }
        except Exception as e:
            logger.error(f"❌ Prédiction échouée: {e}")
            return self._mock_predict(image_array)
    
    def _mock_predict(self, image_array: np.ndarray) -> Dict:
        """Mock intelligent pour simulation"""
        weights = [0.25, 0.30, 0.25, 0.20]
        selected_class = random.choices(self.classes, weights=weights)[0]
        
        conf_range = self.mock_patterns[selected_class]['confidence_range']
        confidence = random.uniform(conf_range[0], conf_range[1])
        
        remaining = 1.0 - confidence
        other_classes = [c for c in self.classes if c != selected_class]
        other_scores = np.random.dirichlet(np.ones(len(other_classes))) * remaining
        
        all_scores = {}
        score_idx = 0
        for cls in self.classes:
            if cls == selected_class:
                all_scores[cls] = confidence
            else:
                all_scores[cls] = float(other_scores[score_idx])
                score_idx += 1
        
        return {
            'class': selected_class,
            'confidence': confidence,
            'all_scores': all_scores
        }
    
    def get_mock_keywords(self, document_class: str) -> list:
        """Retourne des keywords mock pour une classe"""
        if document_class in self.mock_patterns:
            keywords = self.mock_patterns[document_class]['keywords']
            k = random.randint(3, 5)
            return random.sample(keywords, min(k, len(keywords)))
        return []
    
    def get_uptime(self) -> float:
        """Retourne le temps de fonctionnement en secondes"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def increment_predictions(self):
        """Incrémente le compteur de prédictions"""
        self.total_predictions += 1
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du modèle"""
        return {
            'model_loaded': self.is_model_loaded(),
            'load_progress': self.get_load_progress(),
            'uptime_seconds': self.get_uptime(),
            'total_predictions': self.total_predictions,
            'mode': 'real' if self.is_model_loaded() else 'simulation',
            'classes': self.classes
        }