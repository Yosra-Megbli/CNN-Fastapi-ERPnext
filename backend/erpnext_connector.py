"""
ERPNEXT CONNECTOR - Connexion réelle avec ERPNext
Remplace la simulation SQLite par une vraie intégration
"""

import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ERPNextConnector:
    """
    Connecteur pour communiquer avec ERPNext via REST API
    """
    
    def __init__(self, url: str, api_key: str, api_secret: str):
        """
        Initialise la connexion ERPNext
        
        Args:
            url: URL de base ERPNext (ex: http://localhost:8080)
            api_key: Clé API générée dans ERPNext
            api_secret: Secret API
        """
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        
        # Headers pour authentification
        self.session.headers.update({
            'Authorization': f'token {api_key}:{api_secret}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        logger.info(f"✅ ERPNext Connector initialized: {self.url}")
    
    def test_connection(self) -> bool:
        """
        Teste la connexion à ERPNext
        
        Returns:
            True si connexion OK, False sinon
        """
        try:
            response = self.session.get(
                f"{self.url}/api/method/frappe.auth.get_logged_user",
                timeout=5
            )
            
            if response.status_code == 200:
                user = response.json().get('message', 'Unknown')
                logger.info(f"✅ Connected to ERPNext as: {user}")
                return True
            else:
                logger.error(f"❌ ERPNext connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ ERPNext connection error: {str(e)}")
            return False
    
    def create_ai_document(self, doc_data: Dict) -> Optional[str]:
        """
        Crée un document AI_Document dans ERPNext
        
        Args:
            doc_data: {
                'document_class': str,
                'filename': str,
                'file_hash': str,
                'confidence_score': float,
                'keywords': str,
                'summary': str,
                'ocr_text': str,
                'uploaded_by': str
            }
        
        Returns:
            Nom du document créé ou None si échec
        """
        try:
            # Préparer les données pour ERPNext
            erp_doc = {
                'doctype': 'AI_Document',
                'document_class': doc_data.get('document_class'),
                'filename': doc_data.get('filename'),
                'file_hash': doc_data.get('file_hash', ''),
                'confidence_score': doc_data.get('confidence_score', 0.0),
                'keywords': doc_data.get('keywords', ''),
                'summary': doc_data.get('summary', ''),
                'ocr_text': doc_data.get('ocr_text', ''),
                'uploaded_by': 'Administrator',  # Force Administrator au lieu de 'admin',
                'upload_date': datetime.now().isoformat(),
                'is_encrypted': doc_data.get('is_encrypted', 0)
            }
            
            # Envoyer à ERPNext
            response = self.session.post(
                f"{self.url}/api/resource/AI_Document",
                json=erp_doc,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                doc_name = result.get('data', {}).get('name')
                logger.info(f"✅ AI_Document created: {doc_name}")
                return doc_name
            else:
                logger.error(f"❌ Failed to create AI_Document: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error creating AI_Document: {str(e)}")
            return None
    
    def get_document(self, doc_name: str) -> Optional[Dict]:
        """
        Récupère un document par son nom
        
        Args:
            doc_name: Nom du document dans ERPNext
        
        Returns:
            Données du document ou None
        """
        try:
            response = self.session.get(
                f"{self.url}/api/resource/AI_Document/{doc_name}",
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json().get('data', {})
            else:
                logger.error(f"❌ Document not found: {doc_name}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting document: {str(e)}")
            return None
    
    def get_documents(self, filters: Optional[Dict] = None, limit: int = 50) -> List[Dict]:
        """
        Liste les documents avec filtres
        
        Args:
            filters: Filtres ERPNext (ex: {'document_class': 'Invoice'})
            limit: Nombre max de résultats
        
        Returns:
            Liste de documents
        """
        try:
            params = {
                'doctype': 'AI_Document',
                'fields': '["*"]',
                'limit_page_length': limit,
                'order_by': 'creation desc'
            }
            
            if filters:
                import json
                params['filters'] = json.dumps(filters)
            
            response = self.session.get(
                f"{self.url}/api/resource/AI_Document",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json().get('data', [])
            else:
                logger.error(f"❌ Failed to get documents: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error getting documents: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict:
        """
        Récupère les statistiques des documents
        
        Returns:
            {
                'total': int,
                'by_class': {'Invoice': 10, ...},
                'avg_confidence': float
            }
        """
        try:
            all_docs = self.get_documents(limit=1000)
            
            stats = {
                'total': len(all_docs),
                'by_class': {},
                'avg_confidence': 0.0
            }
            
            # Compter par classe
            total_confidence = 0.0
            for doc in all_docs:
                doc_class = doc.get('document_class', 'Unknown')
                stats['by_class'][doc_class] = stats['by_class'].get(doc_class, 0) + 1
                total_confidence += doc.get('confidence_score', 0.0)
            
            # Moyenne de confiance
            if stats['total'] > 0:
                stats['avg_confidence'] = total_confidence / stats['total']
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting statistics: {str(e)}")
            return {'total': 0, 'by_class': {}, 'avg_confidence': 0.0}
    
    def delete_document(self, doc_name: str) -> bool:
        """
        Supprime un document
        
        Args:
            doc_name: Nom du document à supprimer
        
        Returns:
            True si succès
        """
        try:
            response = self.session.delete(
                f"{self.url}/api/resource/AI_Document/{doc_name}",
                timeout=5
            )
            
            if response.status_code in [200, 202]:
                logger.info(f"✅ Document deleted: {doc_name}")
                return True
            else:
                logger.error(f"❌ Failed to delete: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error deleting document: {str(e)}")
            return False
    
    def check_duplicate(self, file_hash: str) -> Optional[Dict]:
        """
        Vérifie si un document avec ce hash existe déjà
        
        Args:
            file_hash: SHA-256 hash du fichier
        
        Returns:
            Document existant ou None
        """
        try:
            docs = self.get_documents(
                filters={'file_hash': file_hash},
                limit=1
            )
            
            if docs:
                logger.warning(f"⚠️ Duplicate detected: {file_hash}")
                return docs[0]
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error checking duplicate: {str(e)}")
            return None
    
    def bulk_insert(self, documents: List[Dict]) -> Dict:
        """
        Insertion en masse de documents
        
        Args:
            documents: Liste de doc_data
        
        Returns:
            {'success': int, 'failed': int, 'errors': [...]}
        """
        result = {
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        for doc_data in documents:
            doc_name = self.create_ai_document(doc_data)
            
            if doc_name:
                result['success'] += 1
            else:
                result['failed'] += 1
                result['errors'].append({
                    'filename': doc_data.get('filename'),
                    'error': 'Failed to create'
                })
        
        logger.info(f"📊 Bulk insert: {result['success']} success, {result['failed']} failed")
        return result