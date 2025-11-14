"""
JWT AUTHENTICATION
Token-based security pour l'API
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION JWT
# ============================================================================
SECRET_KEY = "arkeyez-secret-key-2025-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 heures

# ============================================================================
# JWT FUNCTIONS
# ============================================================================
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Crée un token JWT
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    logger.info(f"🔐 Token created for user: {data.get('sub')}")
    return encoded_jwt

def verify_token(token: str) -> dict:
    """
    Vérifie et décode un token JWT
    Raise HTTPException si invalide
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            raise credentials_exception
        
        # Vérifier expiration
        exp = payload.get("exp")
        if exp is None:
            raise credentials_exception
        
        if datetime.utcnow() > datetime.fromtimestamp(exp):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return payload
        
    except JWTError as e:
        logger.warning(f"⚠️ JWT verification failed: {str(e)}")
        raise credentials_exception

def decode_token(token: str) -> Optional[dict]:
    """
    Décode un token sans lever d'exception
    Retourne None si invalide
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None