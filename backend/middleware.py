"""
MIDDLEWARE - Request Logging
Log toutes les requêtes : method, timestamp, IP, endpoint, status
"""

from fastapi import Request
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)

async def log_requests_middleware(request: Request, call_next):
    """
    Middleware pour logger toutes les requêtes
    Capture : method, timestamp, IP, endpoint, status code, duration
    """
    # Informations de la requête
    start_time = time.time()
    timestamp = datetime.now().isoformat()
    method = request.method
    endpoint = request.url.path
    client_ip = request.client.host if request.client else "unknown"
    
    # Log de la requête entrante
    logger.info(f"[REQUEST] {timestamp} | {method} {endpoint} | IP: {client_ip}")
    
    # Exécuter la requête
    try:
        response = await call_next(request)
        
        # Calculer la durée
        duration = time.time() - start_time
        
        # Log de la réponse
        logger.info(
            f"[RESPONSE] {timestamp} | {method} {endpoint} | "
            f"Status: {response.status_code} | Duration: {duration:.3f}s | IP: {client_ip}"
        )
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"[ERROR] {timestamp} | {method} {endpoint} | "
            f"Error: {str(e)} | Duration: {duration:.3f}s | IP: {client_ip}"
        )
        raise

class RequestLogger:
    """
    Classe pour gérer les logs structurés des requêtes
    Peut être étendue pour sauvegarder dans une base de données
    """
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    def log_request(self, request: Request, response_status: int, duration: float):
        """
        Log une requête avec toutes les informations
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'method': request.method,
            'endpoint': request.url.path,
            'ip_address': request.client.host if request.client else "unknown",
            'user_agent': request.headers.get('user-agent', 'unknown'),
            'status_code': response_status,
            'duration_seconds': round(duration, 3)
        }
        
        # Log dans fichier/console
        self.logger.info(
            f"REQUEST | "
            f"Time: {log_entry['timestamp']} | "
            f"Method: {log_entry['method']} | "
            f"Endpoint: {log_entry['endpoint']} | "
            f"IP: {log_entry['ip_address']} | "
            f"Status: {log_entry['status_code']} | "
            f"Duration: {log_entry['duration_seconds']}s"
        )
        
        # Optionnel : Sauvegarder dans DB
        if self.db_manager:
            self._save_to_database(log_entry)
        
        return log_entry
    
    def _save_to_database(self, log_entry):
        """
        Sauvegarde le log dans une table SQLite
        """
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Créer la table si elle n'existe pas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    method TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    status_code INTEGER,
                    duration_seconds REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insérer le log
            cursor.execute("""
                INSERT INTO request_logs 
                (timestamp, method, endpoint, ip_address, user_agent, status_code, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                log_entry['timestamp'],
                log_entry['method'],
                log_entry['endpoint'],
                log_entry['ip_address'],
                log_entry['user_agent'],
                log_entry['status_code'],
                log_entry['duration_seconds']
            ))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to save log to database: {e}")
    
    def get_recent_logs(self, limit: int = 100):
        """
        Récupère les logs récents depuis la DB
        """
        if not self.db_manager:
            return []
        
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM request_logs 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            
            logs = []
            for row in rows:
                logs.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'method': row[2],
                    'endpoint': row[3],
                    'ip_address': row[4],
                    'user_agent': row[5],
                    'status_code': row[6],
                    'duration_seconds': row[7],
                    'created_at': row[8]
                })
            
            return logs
            
        except Exception as e:
            self.logger.error(f"Failed to get logs from database: {e}")
            return []
    
    def get_statistics(self):
        """
        Statistiques des requêtes
        """
        if not self.db_manager:
            return {}
        
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Total requêtes
            cursor.execute("SELECT COUNT(*) FROM request_logs")
            total_requests = cursor.fetchone()[0]
            
            # Requêtes par méthode
            cursor.execute("""
                SELECT method, COUNT(*) as count 
                FROM request_logs 
                GROUP BY method
            """)
            by_method = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Requêtes par status code
            cursor.execute("""
                SELECT status_code, COUNT(*) as count 
                FROM request_logs 
                GROUP BY status_code
            """)
            by_status = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Durée moyenne
            cursor.execute("SELECT AVG(duration_seconds) FROM request_logs")
            avg_duration = cursor.fetchone()[0]
            
            return {
                'total_requests': total_requests,
                'by_method': by_method,
                'by_status': by_status,
                'avg_duration_seconds': round(avg_duration, 3) if avg_duration else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
