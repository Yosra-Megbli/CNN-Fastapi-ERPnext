"""
DATABASE MANAGER - SQLite avec 4 tables séparées
Drawing, Invoice, Report, Receipt
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# MODELS PYDANTIC
# ============================================================================
class DocumentInsert(BaseModel):
    """Modèle pour l'insertion d'un document"""
    document_class: str  # Drawing, Invoice, Report, Receipt
    filename: str
    confidence_score: float
    keywords: List[str]
    summary: str
    ocr_text: Optional[str] = None
    uploaded_by: Optional[str] = "admin"

# ============================================================================
# DATABASE MANAGER
# ============================================================================
class DatabaseManager:
    """
    Gestionnaire de base de données SQLite
    4 tables séparées pour les 4 classes de documents
    """
    
    def __init__(self, db_path: str = "archive.db"):
        self.db_path = db_path
        self.classes = ['Drawing', 'Invoice', 'Report', 'Receipt']
        self.connection = None
        logger.info(f"📊 Database initialized: {db_path}")
    
    def get_connection(self):
        """Crée ou retourne la connexion SQLite"""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Pour dict-like access
        return self.connection
    
    def create_tables(self):
        """
        Crée les 4 tables séparées si elles n'existent pas
        Une table par classe de document
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Schéma identique pour les 4 tables
        table_schema = """
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                keywords TEXT,
                summary TEXT,
                ocr_text TEXT,
                uploaded_by TEXT DEFAULT 'admin',
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        
        for class_name in self.classes:
            cursor.execute(table_schema.format(table_name=class_name))
            logger.info(f"✅ Table '{class_name}' ready")
        
        conn.commit()
        logger.info("📊 All database tables created successfully")
    
    def insert_document(self, doc: DocumentInsert) -> int:
        """
        Insert un document dans la table correspondante
        Retourne l'ID du document inséré
        """
        # Valider la classe
        if doc.document_class not in self.classes:
            raise ValueError(f"Invalid class: {doc.document_class}. Must be one of {self.classes}")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Convertir keywords list en string
        keywords_str = ", ".join(doc.keywords) if doc.keywords else ""
        
        # Insertion dans la table appropriée
        query = f"""
            INSERT INTO {doc.document_class} 
            (filename, confidence_score, keywords, summary, ocr_text, uploaded_by)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        
        cursor.execute(query, (
            doc.filename,
            doc.confidence_score,
            keywords_str,
            doc.summary,
            doc.ocr_text,
            doc.uploaded_by
        ))
        
        conn.commit()
        inserted_id = cursor.lastrowid
        
        logger.info(f"✅ Document inserted into {doc.document_class} table (ID: {inserted_id})")
        return inserted_id
    
    def get_documents_by_class(self, class_name: str, limit: int = 50) -> List[Dict]:
        """Récupère les documents d'une classe spécifique"""
        if class_name not in self.classes:
            raise ValueError(f"Invalid class: {class_name}")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = f"""
            SELECT * FROM {class_name}
            ORDER BY upload_date DESC
            LIMIT ?
        """
        
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        
        # Convertir en liste de dicts
        documents = []
        for row in rows:
            doc = dict(row)
            doc['document_class'] = class_name  # Ajouter la classe
            documents.append(doc)
        
        return documents
    
    def get_all_documents(self, limit: int = 50) -> List[Dict]:
        """
        Récupère les documents de TOUTES les tables
        Triés par date (plus récents en premier)
        """
        all_docs = []
        
        for class_name in self.classes:
            docs = self.get_documents_by_class(class_name, limit=limit)
            all_docs.extend(docs)
        
        # Trier par date (plus récents en premier)
        all_docs.sort(key=lambda x: x['upload_date'], reverse=True)
        
        # Limiter au nombre demandé
        return all_docs[:limit]
    
    def get_statistics(self) -> Dict:
        """
        Retourne les statistiques pour chaque classe
        Nombre de documents par table
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = {
            'total': 0,
            'by_class': {},
            'last_update': datetime.now().isoformat()
        }
        
        for class_name in self.classes:
            query = f"SELECT COUNT(*) as count FROM {class_name}"
            cursor.execute(query)
            count = cursor.fetchone()[0]
            
            stats['by_class'][class_name] = count
            stats['total'] += count
        
        return stats
    
    def get_recent_activity(self, days: int = 7) -> List[Dict]:
        """
        Activité récente (derniers X jours)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        activity = []
        
        for class_name in self.classes:
            query = f"""
                SELECT 
                    '{class_name}' as document_class,
                    COUNT(*) as count,
                    DATE(upload_date) as date
                FROM {class_name}
                WHERE upload_date >= datetime('now', '-{days} days')
                GROUP BY DATE(upload_date)
                ORDER BY date DESC
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            for row in rows:
                activity.append(dict(row))
        
        return activity
    
    def delete_document(self, class_name: str, doc_id: int) -> bool:
        """Supprime un document"""
        if class_name not in self.classes:
            raise ValueError(f"Invalid class: {class_name}")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = f"DELETE FROM {class_name} WHERE id = ?"
        cursor.execute(query, (doc_id,))
        conn.commit()
        
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"🗑️  Document {doc_id} deleted from {class_name}")
        
        return deleted
    
    def search_documents(self, search_term: str, limit: int = 20) -> List[Dict]:
        """
        Recherche dans toutes les tables
        Cherche dans filename, keywords, summary
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        results = []
        search_pattern = f"%{search_term}%"
        
        for class_name in self.classes:
            query = f"""
                SELECT * FROM {class_name}
                WHERE filename LIKE ? 
                   OR keywords LIKE ?
                   OR summary LIKE ?
                ORDER BY upload_date DESC
                LIMIT ?
            """
            cursor.execute(query, (search_pattern, search_pattern, search_pattern, limit))
            rows = cursor.fetchall()
            
            for row in rows:
                doc = dict(row)
                doc['document_class'] = class_name
                results.append(doc)
        
        return results[:limit]
    
    def get_confidence_distribution(self) -> Dict:
        """Distribution des scores de confiance par classe"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        distribution = {}
        
        for class_name in self.classes:
            query = f"""
                SELECT 
                    AVG(confidence_score) as avg_confidence,
                    MIN(confidence_score) as min_confidence,
                    MAX(confidence_score) as max_confidence,
                    COUNT(*) as count
                FROM {class_name}
            """
            cursor.execute(query)
            row = cursor.fetchone()
            
            distribution[class_name] = {
                'average': round(row[0], 4) if row[0] else 0,
                'min': round(row[1], 4) if row[1] else 0,
                'max': round(row[2], 4) if row[2] else 0,
                'count': row[3]
            }
        
        return distribution
    
    def close(self):
        """Ferme la connexion"""
        if self.connection:
            self.connection.close()
            logger.info("📊 Database connection closed")
