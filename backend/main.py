# ============================================================================
# CHARGEMENT .ENV - DOIT ÊTRE EN PREMIER
# ============================================================================
from dotenv import load_dotenv
import os

# Charger .env AVANT tout le reste
load_dotenv()

# Vérifier que les variables sont bien chargées
print("=" * 70)
print("🔍 CHECKING ENVIRONMENT VARIABLES")
print("=" * 70)
print(f"ERPNEXT_URL: {os.getenv('ERPNEXT_URL', 'NOT SET')}")
print(f"ERPNEXT_API_KEY: {os.getenv('ERPNEXT_API_KEY', 'NOT SET')}")
print(f"ERPNEXT_API_SECRET: {os.getenv('ERPNEXT_API_SECRET', 'NOT SET')[:20] if os.getenv('ERPNEXT_API_SECRET') else 'NOT SET'}...")
print("=" * 70)

"""
ARKEYEZDOC v2.0 - Backend FastAPI
WebSocket Real-time + ERPNext Integration + CNN+OCR Fusion

🔧 MODIFICATIONS:
- Insert into ERPNext ONLY (pas de fallback local DB)
- Erreur claire si ERPNext non connecté
- Endpoint debug ERPNext ajouté
"""

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from contextlib import asynccontextmanager
import uvicorn
import logging
from datetime import datetime
import io
import base64
import json
from PIL import Image
import numpy as np
import fitz  # PyMuPDF
import asyncio

# Local imports
from models import ModelManager
from database import DatabaseManager, DocumentInsert
from auth import verify_token, create_access_token
from ocr_nlp import OCRNLPPipeline
from middleware import log_requests_middleware, RequestLogger
from erpnext_connector import ERPNextConnector

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL MANAGERS (initialisés avant lifespan)
# ============================================================================
model_manager = ModelManager(model_path="../models/final_model_complete.h5")
db_manager = DatabaseManager(db_path="archive.db")
ocr_nlp = OCRNLPPipeline()
request_logger = RequestLogger(db_manager=db_manager)

# ERPNext Connector
ERPNEXT_URL = os.getenv("ERPNEXT_URL", "http://localhost:8080")
ERPNEXT_API_KEY = os.getenv("ERPNEXT_API_KEY", "")
ERPNEXT_API_SECRET = os.getenv("ERPNEXT_API_SECRET", "")

erpnext_connector = None
if ERPNEXT_API_KEY and ERPNEXT_API_SECRET:
    erpnext_connector = ERPNextConnector(ERPNEXT_URL, ERPNEXT_API_KEY, ERPNEXT_API_SECRET)
    if erpnext_connector.test_connection():
        logger.info("✅ ERPNext connector initialized successfully")
    else:
        logger.warning("⚠️ ERPNext connection failed")
        erpnext_connector = None
else:
    logger.warning("⚠️ ERPNext credentials not set")

# ============================================================================
# LIFESPAN CONTEXT (remplace @app.on_event - deprecated)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # ==================== STARTUP ====================
    logger.info("=" * 70)
    logger.info("🚀 Starting ArkeyezDoc v2.0...")
    logger.info("=" * 70)
    logger.info("📡 WebSocket streaming: ENABLED")
    logger.info(f"🔗 ERPNext: {'CONNECTED' if erpnext_connector else 'NOT CONNECTED'}")
    logger.info("📊 Database: Initializing tables...")
    
    # Créer les tables
    db_manager.create_tables()
    logger.info("✅ Database tables ready")
    
    # Charger le modèle
    logger.info("🤖 Model: Starting background loading...")
    model_manager.start_loading()
    
    # Attendre que le modèle soit chargé (avec timeout et feedback)
    max_wait = 60  # 60 secondes max
    waited = 0
    check_interval = 2  # Vérifier toutes les 2 secondes
    
    logger.info(f"⏳ Waiting for model to load (max {max_wait}s)...")
    
    while not model_manager.is_model_loaded() and waited < max_wait:
        # Vérifier si le chargement a échoué
        if model_manager._model_load_failed:
            logger.warning("⚠️ Model loading failed - running in SIMULATION mode")
            break
        
        # Vérifier si toujours en chargement
        if not model_manager.is_loading and not model_manager.is_model_loaded():
            logger.warning("⚠️ Model loading stopped without success - SIMULATION mode")
            break
        
        await asyncio.sleep(check_interval)
        waited += check_interval
        
        # Log toutes les 10 secondes
        if waited % 10 == 0:
            progress = model_manager.get_load_progress() * 100
            logger.info(f"⏳ Model loading... {progress:.0f}% ({waited}s / {max_wait}s)")
    
    # Vérifier le résultat final
    if model_manager.is_model_loaded():
        logger.info("=" * 70)
        logger.info("✅ MODEL LOADED - Real CNN predictions enabled")
        logger.info("=" * 70)
    elif waited >= max_wait:
        logger.warning("=" * 70)
        logger.warning("⚠️ MODEL LOADING TIMEOUT - Running in SIMULATION mode")
        logger.warning("=" * 70)
    else:
        logger.warning("=" * 70)
        logger.warning("⚠️ MODEL NOT LOADED - Running in SIMULATION mode")
        logger.warning("=" * 70)
    
    logger.info("🎯 API Ready!")
    logger.info("=" * 70)
    
    yield  # L'application tourne ici
    
    # ==================== SHUTDOWN ====================
    logger.info("=" * 70)
    logger.info("🛑 Shutting down ArkeyezDoc...")
    logger.info("=" * 70)
    db_manager.close()
    logger.info("✅ Database closed")
    logger.info("👋 Shutdown complete")
    logger.info("=" * 70)

# ============================================================================
# CONFIGURATION
# ============================================================================
app = FastAPI(
    title="ArkeyezDoc",
    description="AI-Powered ERP Documents Classification with Real-time Streaming",
    version="2.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Middleware
app.middleware("http")(log_requests_middleware)

# ============================================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================================
class ConnectionManager:
    """Manages WebSocket connections for real-time streaming"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"✅ WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"❌ WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send broadcast: {e}")
                self.disconnect(connection)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")

ws_manager = ConnectionManager()

# ============================================================================
# FRONTEND
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

if os.path.exists(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

@app.get("/")
async def root():
    """Serve dashboard"""
    frontend_path = os.path.join(FRONTEND_DIR, "dashboard.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "ArkeyezDoc API v2.0", "docs": "/api/v1/docs"}

# ============================================================================
# CUSTOM DOCS
# ============================================================================
@app.get("/api/v1/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <link type="text/css" rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
        <title>ArkeyezDoc API Documentation</title>
        <style>
            .dashboard-btn {
                position: fixed;
                top: 10px;
                right: 20px;
                z-index: 9999;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                cursor: pointer;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .dashboard-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            }
        </style>
    </head>
    <body>
        <button class="dashboard-btn" onclick="window.location.href='/'">🏠 Dashboard</button>
        <div id="swagger-ui"></div>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
        <script>
            window.onload = function() {
                window.ui = SwaggerUIBundle({
                    url: '/api/v1/openapi.json',
                    dom_id: '#swagger-ui',
                    deepLinking: true,
                    presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
                    layout: "BaseLayout"
                });
            };
        </script>
    </body>
    </html>
    """)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class LoginRequest(BaseModel):
    username: str
    password: str

class FileClassificationResult(BaseModel):
    filename: str
    document_class: str
    confidence: float
    cnn_confidence: float
    ocr_boost: float
    fusion_applied: bool
    keywords: List[str]
    summary: str
    ocr_text: Optional[str]
    page_number: Optional[int] = None
    image_base64: str

class ClassificationResponse(BaseModel):
    results: List[FileClassificationResult]
    total_files: int
    total_pages: int
    is_simulation: bool
    fusion_enabled: bool
    timestamp: str

class StatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_loaded: bool
    load_progress: float
    uptime_seconds: float
    total_predictions: int
    test_accuracy: float
    mode: str
    fusion_enabled: bool
    erpnext_connected: bool
    websocket_clients: int
    model_info: dict

class InsertResponse(BaseModel):
    success: bool
    message: str
    inserted_id: Optional[str]
    erpnext_name: Optional[str]

# ============================================================================
# CNN + OCR FUSION
# ============================================================================
CLASS_KEYWORDS = {
    'Invoice': ['invoice', 'bill', 'facture', 'total', 'amount', 'montant', 'payment', 'tax', 'tva'],
    'Drawing': ['drawing', 'diagram', 'plan', 'blueprint', 'technical', 'schematic'],
    'Report': ['report', 'rapport', 'analysis', 'summary', 'findings', 'conclusion'],
    'Receipt': ['receipt', 'reçu', 'ticket', 'proof', 'purchase', 'transaction']
}

def analyze_ocr_for_class(keywords: List[str], ocr_text: str = "") -> dict:
    """Analyse OCR text to predict document class"""
    class_scores = {}
    keywords_lower = [kw.lower() for kw in keywords]
    ocr_text_lower = ocr_text.lower() if ocr_text else ""
    
    for class_name, class_keywords in CLASS_KEYWORDS.items():
        score = 0
        for kw in keywords_lower:
            for class_kw in class_keywords:
                if class_kw in kw or kw in class_kw:
                    score += 1
        if ocr_text:
            for class_kw in class_keywords:
                if class_kw in ocr_text_lower:
                    score += 0.5
        class_scores[class_name] = score / len(class_keywords) if class_keywords else 0
    
    best_class = max(class_scores, key=class_scores.get) if class_scores else None
    best_score = class_scores[best_class] if best_class else 0
    
    return {'predicted_class': best_class, 'confidence': best_score, 'all_scores': class_scores}

def fusion_cnn_ocr(cnn_prediction: dict, keywords: List[str], ocr_text: str = "") -> dict:
    """Fusion CNN and OCR predictions"""
    cnn_class = cnn_prediction['class']
    cnn_confidence = cnn_prediction['confidence']
    
    ocr_analysis = analyze_ocr_for_class(keywords, ocr_text)
    ocr_class = ocr_analysis['predicted_class']
    ocr_confidence = ocr_analysis['confidence']
    
    final_class = cnn_class
    final_confidence = cnn_confidence
    ocr_boost = 0.0
    fusion_applied = False
    
    if ocr_class and ocr_confidence > 0.1:
        fusion_applied = True
        if cnn_class == ocr_class:
            boost = min(0.05 + (ocr_confidence * 0.03), 0.08)
            final_confidence = min(cnn_confidence + boost, 0.99)
            ocr_boost = boost
        elif ocr_confidence > 0.3:
            penalty = 0.03
            final_confidence = max(cnn_confidence - penalty, 0.60)
            ocr_boost = -penalty
    
    return {
        'class': final_class,
        'confidence': final_confidence,
        'cnn_confidence': cnn_confidence,
        'ocr_boost': ocr_boost,
        'fusion_applied': fusion_applied
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    """Convert PDF to list of images"""
    images = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    pdf_document.close()
    return images

def process_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def image_to_base64(image: Image.Image, max_size: int = 400) -> str:
    """Convert image to base64 string"""
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"

# ============================================================================
# WEBSOCKET ENDPOINT - REAL-TIME CLASSIFICATION
# ============================================================================
@app.websocket("/ws/classify")
async def websocket_classify(websocket: WebSocket):
    """
    WebSocket endpoint for real-time document classification
    
    Protocol:
    Client sends: {"type": "classify", "image": "base64_string", "filename": "doc.jpg"}
    Server sends: {"type": "progress", "step": "ocr", "progress": 50}
    Server sends: {"type": "result", "data": {...classification_result...}}
    """
    await ws_manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "classify":
                try:
                    # Progress: Start
                    await ws_manager.send_personal_message({
                        "type": "progress",
                        "step": "start",
                        "progress": 0,
                        "message": "Classification started..."
                    }, websocket)
                    
                    # Decode image
                    image_data = data.get("image", "").split(",")[1] if "," in data.get("image", "") else data.get("image", "")
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    filename = data.get("filename", "unknown.jpg")
                    
                    # Progress: Image loaded
                    await ws_manager.send_personal_message({
                        "type": "progress",
                        "step": "loaded",
                        "progress": 20,
                        "message": "Image loaded..."
                    }, websocket)
                    
                    # CNN Prediction
                    img_array = process_image(image)
                    cnn_prediction = model_manager.predict(img_array)
                    
                    await ws_manager.send_personal_message({
                        "type": "progress",
                        "step": "cnn",
                        "progress": 50,
                        "message": "CNN classification complete..."
                    }, websocket)
                    
                    # OCR Extraction
                    keywords = []
                    ocr_text = None
                    if model_manager.is_model_loaded():
                        await ws_manager.send_personal_message({
                            "type": "progress",
                            "step": "ocr",
                            "progress": 70,
                            "message": "Extracting text (OCR)..."
                        }, websocket)
                        
                        ocr_result = ocr_nlp.extract_text(image_bytes)
                        ocr_text = ocr_result.get('text', '')
                        if ocr_text:
                            keywords = ocr_nlp.extract_keywords(ocr_text, top_k=5)
                    else:
                        keywords = model_manager.get_mock_keywords(cnn_prediction['class'])
                    
                    # Fusion
                    await ws_manager.send_personal_message({
                        "type": "progress",
                        "step": "fusion",
                        "progress": 90,
                        "message": "Fusing CNN + OCR..."
                    }, websocket)
                    
                    fused_result = fusion_cnn_ocr(cnn_prediction, keywords, ocr_text)
                    
                    summary = f"{fused_result['class']} ({fused_result['confidence']*100:.1f}%)"
                    if fused_result['fusion_applied']:
                        summary += f" [Fusion: {fused_result['ocr_boost']*100:+.1f}%]"
                    
                    # Final Result
                    result = {
                        "type": "result",
                        "data": {
                            "filename": filename,
                            "document_class": fused_result['class'],
                            "confidence": fused_result['confidence'],
                            "cnn_confidence": fused_result['cnn_confidence'],
                            "ocr_boost": fused_result['ocr_boost'],
                            "fusion_applied": fused_result['fusion_applied'],
                            "keywords": keywords,
                            "summary": summary,
                            "ocr_text": ocr_text,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    
                    await ws_manager.send_personal_message(result, websocket)
                    model_manager.increment_predictions()
                    
                except Exception as e:
                    logger.error(f"WebSocket classification error: {e}")
                    await ws_manager.send_personal_message({
                        "type": "error",
                        "message": str(e)
                    }, websocket)
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)

# ============================================================================
# ENDPOINTS - AUTH
# ============================================================================
@app.post("/api/v1/login")
async def login(request: LoginRequest):
    """User login"""
    if request.username == "admin" and request.password == "arkeyez2025":
        token = create_access_token({"sub": request.username})
        return {"access_token": token, "token_type": "bearer", "username": request.username}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

# ============================================================================
# ENDPOINTS - CLASSIFICATION (HTTP for backward compatibility)
# ============================================================================
@app.post("/api/v1/classify-multi", response_model=ClassificationResponse)
async def classify_multiple_documents(
    files: List[UploadFile] = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Multi-file classification (HTTP version)"""
    verify_token(credentials.credentials)
    
    results = []
    total_pages = 0
    fusion_enabled = model_manager.is_model_loaded()
    
    for file in files:
        filename = file.filename
        contents = await file.read()
        
        if file.content_type == 'application/pdf' or filename.lower().endswith('.pdf'):
            images = pdf_to_images(contents)
            total_pages += len(images)
            
            for page_num, image in enumerate(images, start=1):
                img_array = process_image(image)
                cnn_prediction = model_manager.predict(img_array)
                
                keywords = []
                ocr_text = None
                if model_manager.is_model_loaded():
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    ocr_result = ocr_nlp.extract_text(img_byte_arr.getvalue())
                    ocr_text = ocr_result.get('text', '')
                    if ocr_text:
                        keywords = ocr_nlp.extract_keywords(ocr_text, top_k=5)
                else:
                    keywords = model_manager.get_mock_keywords(cnn_prediction['class'])
                
                fused_result = fusion_cnn_ocr(cnn_prediction, keywords, ocr_text)
                
                summary = f"Page {page_num}: {fused_result['class']} ({fused_result['confidence']*100:.1f}%)"
                image_base64 = image_to_base64(image)
                
                results.append(FileClassificationResult(
                    filename=f"{filename} - Page {page_num}",
                    document_class=fused_result['class'],
                    confidence=fused_result['confidence'],
                    cnn_confidence=fused_result['cnn_confidence'],
                    ocr_boost=fused_result['ocr_boost'],
                    fusion_applied=fused_result['fusion_applied'],
                    keywords=keywords,
                    summary=summary,
                    ocr_text=ocr_text,
                    page_number=page_num,
                    image_base64=image_base64
                ))
                
                model_manager.increment_predictions()
        
        elif file.content_type.startswith('image/'):
            image = Image.open(io.BytesIO(contents))
            img_array = process_image(image)
            cnn_prediction = model_manager.predict(img_array)
            
            ocr_text = None
            keywords = []
            if model_manager.is_model_loaded():
                try:
                    ocr_result = ocr_nlp.extract_text(contents)
                except Exception as e:
                    logger.error(f"OCR error: {e}")
                    ocr_result = {'text': ''}
                ocr_text = ocr_result.get('text', '')
                if ocr_text:
                    keywords = ocr_nlp.extract_keywords(ocr_text, top_k=5)
            else:
                keywords = model_manager.get_mock_keywords(cnn_prediction['class'])
            
            fused_result = fusion_cnn_ocr(cnn_prediction, keywords, ocr_text)
            
            summary = f"{fused_result['class']} ({fused_result['confidence']*100:.1f}%)"
            image_base64 = image_to_base64(image)
            
            results.append(FileClassificationResult(
                filename=filename,
                document_class=fused_result['class'],
                confidence=fused_result['confidence'],
                cnn_confidence=fused_result['cnn_confidence'],
                ocr_boost=fused_result['ocr_boost'],
                fusion_applied=fused_result['fusion_applied'],
                keywords=keywords,
                summary=summary,
                ocr_text=ocr_text,
                image_base64=image_base64
            ))
            
            total_pages += 1
            model_manager.increment_predictions()
    
    return ClassificationResponse(
        results=results,
        total_files=len(files),
        total_pages=total_pages,
        is_simulation=not model_manager.is_model_loaded(),
        fusion_enabled=fusion_enabled,
        timestamp=datetime.now().isoformat()
    )

# ============================================================================
# ENDPOINTS - ERPNEXT (INSERTION UNIQUEMENT - PAS DE FALLBACK LOCAL)
# ============================================================================
@app.post("/api/v1/erpnext/insert", response_model=InsertResponse)
async def insert_document(
    doc: DocumentInsert,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    🔴 MODIFICATION CRITIQUE: Insert document into ERPNext ONLY
    
    - Si ERPNext connecté: Insertion dans ERPNext ✅
    - Si ERPNext NON connecté: ERREUR 503 ❌ (pas de fallback local DB)
    """
    verify_token(credentials.credentials)
    
    # Vérifier si ERPNext est connecté
    if not erpnext_connector:
        logger.error("❌ ERPNext NOT CONNECTED - Cannot insert document")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "ERPNext not connected",
                "message": "ERPNext connector is not available. Please configure ERPNEXT_URL, ERPNEXT_API_KEY, and ERPNEXT_API_SECRET environment variables.",
                "action": "Check your .env file and restart the application"
            }
        )
    
    try:
        # Préparer les données pour ERPNext
        doc_data = {
            'document_class': doc.document_class,
            'filename': doc.filename,
            'file_hash': '',
            'confidence_score': doc.confidence_score,
            'keywords': ', '.join(doc.keywords) if doc.keywords else '',
            'summary': doc.summary,
            'ocr_text': doc.ocr_text or '',
            'uploaded_by': doc.uploaded_by or 'Administrator'
        }
        
        logger.info(f"📤 Inserting document into ERPNext: {doc.filename}")
        
        # Insertion dans ERPNext
        erpnext_name = erpnext_connector.create_ai_document(doc_data)
        
        if erpnext_name:
            logger.info(f"✅ Document successfully inserted into ERPNext: {erpnext_name}")
            return InsertResponse(
                success=True,
                message=f"Document successfully inserted into ERPNext",
                inserted_id=erpnext_name,
                erpnext_name=erpnext_name
            )
        else:
            logger.error("❌ ERPNext insertion failed - No document name returned")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "ERPNext insertion failed",
                    "message": "The document could not be inserted into ERPNext. Please check ERPNext logs."
                }
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ ERPNext insertion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "ERPNext insertion failed",
                "message": str(e),
                "type": type(e).__name__
            }
        )

@app.post("/api/v1/erpnext/insert-bulk")
async def insert_documents_bulk(
    documents: List[DocumentInsert],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Insert multiple documents into ERPNext
    """
    verify_token(credentials.credentials)
    
    if not erpnext_connector:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "ERPNext not connected",
                "message": "ERPNext connector is not available."
            }
        )
    
    try:
        doc_list = []
        for doc in documents:
            doc_data = {
                'document_class': doc.document_class,
                'filename': doc.filename,
                'file_hash': '',
                'confidence_score': doc.confidence_score,
                'keywords': ', '.join(doc.keywords) if doc.keywords else '',
                'summary': doc.summary,
                'ocr_text': doc.ocr_text or '',
                'uploaded_by': 'Administrator'
            }
            doc_list.append(doc_data)
        
        result = erpnext_connector.bulk_insert(doc_list)
        
        return {
            "success": True,
            "message": f"Bulk insert completed: {result['success']} success, {result['failed']} failed",
            "details": result
        }
        
    except Exception as e:
        logger.error(f"❌ Bulk insert error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Bulk insert failed", "message": str(e)}
        )

@app.get("/api/v1/erpnext/history")
async def get_history(
    limit: int = 50,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get document history from ERPNext or local DB"""
    verify_token(credentials.credentials)
    
    try:
        if erpnext_connector:
            documents = erpnext_connector.get_documents(limit=limit)
            return {"success": True, "count": len(documents), "documents": documents, "source": "erpnext"}
        else:
            documents = db_manager.get_all_documents(limit=limit)
            return {"success": True, "count": len(documents), "documents": documents, "source": "local"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/v1/erpnext/stats")
async def get_stats(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get statistics from ERPNext or local DB"""
    verify_token(credentials.credentials)
    
    try:
        if erpnext_connector:
            stats = erpnext_connector.get_statistics()
            return {"success": True, "statistics": stats, "source": "erpnext"}
        else:
            stats = db_manager.get_statistics()
            return {"success": True, "statistics": stats, "source": "local"}
    except Exception as e:
        raise HTTPException(500, str(e))

# ============================================================================
# ENDPOINTS - STATUS
# ============================================================================
@app.get("/api/v1/status", response_model=StatusResponse)
async def get_status():
    """Get API status and model information"""
    model_info = {
        "architecture": "ResNet50 + OCR/NLP Fusion",
        "input_size": "224x224x3",
        "classes": ["Invoice", "Drawing", "Report", "Receipt"],
        "framework": "TensorFlow/Keras"
    }
    
    return StatusResponse(
        status="online",
        model_loaded=model_manager.is_model_loaded(),
        load_progress=model_manager.get_load_progress(),
        uptime_seconds=model_manager.get_uptime(),
        total_predictions=model_manager.total_predictions,
        test_accuracy=0.80 if not model_manager.is_model_loaded() else 0.85,
        mode="simulation" if not model_manager.is_model_loaded() else "real",
        fusion_enabled=model_manager.is_model_loaded(),
        erpnext_connected=erpnext_connector is not None,
        websocket_clients=len(ws_manager.active_connections),
        model_info=model_info
    )

@app.get("/api/v1/statuts", response_model=StatusResponse)
async def get_statuts():
    """Alias français pour compatibilité"""
    return await get_status()

# ============================================================================
# ENDPOINTS - DEBUG & DIAGNOSTICS
# ============================================================================
@app.get("/api/v1/debug/model")
async def debug_model(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """🔧 Debug endpoint - Model loading diagnostics"""
    verify_token(credentials.credentials)
    
    load_status = model_manager.get_load_status()
    
    return {
        "model_manager": {
            "is_loading": model_manager.is_loading,
            "is_model_loaded": model_manager.is_model_loaded(),
            "load_progress": model_manager.get_load_progress(),
            "load_failed": model_manager._model_load_failed,
            "total_predictions": model_manager.total_predictions,
            "uptime_seconds": model_manager.get_uptime()
        },
        "file_info": {
            "model_path": model_manager.model_path,
            "absolute_path": os.path.abspath(model_manager.model_path),
            "file_exists": os.path.exists(model_manager.model_path),
            "file_size_mb": os.path.getsize(model_manager.model_path) / (1024*1024) if os.path.exists(model_manager.model_path) else 0
        },
        "load_status": load_status,
        "model_object": str(type(model_manager.model)),
        "classes": model_manager.classes
    }

@app.get("/api/v1/debug/erpnext")
async def debug_erpnext(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """🔧 Debug endpoint - ERPNext connection diagnostics"""
    verify_token(credentials.credentials)
    
    return {
        "erpnext_connector": {
            "is_connected": erpnext_connector is not None,
            "url": ERPNEXT_URL,
            "api_key_configured": bool(ERPNEXT_API_KEY),
            "api_secret_configured": bool(ERPNEXT_API_SECRET)
        },
        "connection_test": erpnext_connector.test_connection() if erpnext_connector else False,
        "environment_variables": {
            "ERPNEXT_URL": ERPNEXT_URL,
            "ERPNEXT_API_KEY": f"{ERPNEXT_API_KEY[:10]}..." if ERPNEXT_API_KEY else "NOT SET",
            "ERPNEXT_API_SECRET": f"{ERPNEXT_API_SECRET[:10]}..." if ERPNEXT_API_SECRET else "NOT SET"
        },
        "message": "✅ ERPNext connected" if erpnext_connector else "❌ ERPNext NOT connected - Configure .env file"
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "components": {
            "api": "ok",
            "database": "ok" if db_manager else "error",
            "model": "loaded" if model_manager.is_model_loaded() else "simulation",
            "erpnext": "connected" if erpnext_connector else "disconnected",
            "websocket": f"{len(ws_manager.active_connections)} clients"
        }
    }

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True, 
        log_level="info"
    )