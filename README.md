#  ArkeyezDoc v2.0 - AI-Powered Document Classification for ERPNext

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![ERPNext](https://img.shields.io/badge/ERPNext-Integration-blue?style=flat)](https://erpnext.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)](https://www.python.org/)

> **Intelligent document classification system based on CNN + OCR/NLP**, with native ERPNext integration and real-time WebSocket streaming.

---

##  Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [ERPNext Configuration](#-erpnext-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)

---

##  Overview

**ArkeyezDoc** is an AI system that automatically classifies documents (invoices, technical drawings, reports, receipts) using:

- **CNN (ResNet50)**: Visual classification through deep learning
- **OCR (EasyOCR)**: Multilingual text extraction (FR/EN)
- **NLP**: Semantic analysis and keyword extraction
- **CNN+OCR Fusion**: Intelligent combination for accuracy boost
- **ERPNext Integration**: Direct insertion into ERPNext via REST API
- **WebSocket Streaming**: Real-time processing tracking

### Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 85-92% |
| **Supported Classes** | 4 (Invoice, Drawing, Report, Receipt) |
| **Processing Time** | ~2-3s per document |
| **Supported Formats** | PDF, JPG, PNG |

---

## Features

###  Core Features

- ✅ **Multi-document Classification**: Batch processing (multi-page PDF supported)
- ✅ **CNN+OCR Fusion**: Confidence boost up to +8% through textual analysis
- ✅ **ERPNext Native**: Direct insertion into `AI_Document` DocType
- ✅ **WebSocket Real-time**: Progressive tracking (0% → 100%) with detailed steps
- ✅ **JWT Authentication**: Secured API with Bearer tokens
- ✅ **Web Dashboard**: Modern and responsive user interface
- ✅ **Complete REST API**: Integrated Swagger UI

### 🔧 Advanced Features

- 🔄 **Duplicate Detection**: SHA-256 hash to prevent duplications
- 📈 **Real-time Statistics**: Document count per class, average confidence
- 🐛 **Debug Endpoints**: Model and ERPNext connection diagnostics
- 🔐 **Security**: Support for sensitive document encryption
- 📝 **Metadata Extraction**: Automatic detection of dates, amounts, references

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (HTML/JS)                       │
│                   Dashboard + WebSocket                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP REST + WebSocket
┌──────────────────────▼──────────────────────────────────────┐
│                  BACKEND (FastAPI)                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │   Auth     │  │  WebSocket │  │  Classification    │    │
│  │   (JWT)    │  │  Manager   │  │  Engine            │    │
│  └────────────┘  └────────────┘  └────────────────────┘    │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │   Model    │  │  OCR/NLP   │  │  ERPNext           │    │
│  │ (ResNet50) │  │ (EasyOCR)  │  │  Connector         │    │
│  └────────────┘  └────────────┘  └────────────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │ REST API
┌──────────────────────▼──────────────────────────────────────┐
│                    ERPNEXT SERVER                            │
│               DocType: AI_Document                           │
└─────────────────────────────────────────────────────────────┘
```

###  Components

| Component | Technology | Role |
|-----------|------------|------|
| **Backend API** | FastAPI | General orchestration |
| **Model Manager** | TensorFlow/Keras | CNN loading & inference |
| **OCR Pipeline** | EasyOCR | Text extraction |
| **NLP Engine** | TF-IDF | Keyword extraction |
| **ERPNext Connector** | Requests | REST communication with ERPNext |
| **WebSocket Manager** | FastAPI WebSocket | Real-time streaming |
| **Database** | SQLite (local) / ERPNext | Document storage |

---

## 📥 Installation

### 1️⃣ Prerequisites

- **Python 3.10+**
- **ERPNext v14+** (local or remote server)
- **4GB RAM minimum** (8GB recommended for model)
- **Git**

### 2️⃣ Clone Repository

```bash
git clone https://github.com/Yosra-Megbli/CNN-Fastapi-ERPnext.git
cd CNN-Fastapi-ERPnext
```

### 3️⃣ Create Virtual Environment

```bash
# Create environment
python -m venv env

# Activate (Windows)
env\Scripts\activate

# Activate (Linux/Mac)
source env/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 5️⃣ Model Configuration

Place your CNN model in the `models/` folder:

```bash
models/
└── final_model_complete.h5  # Your trained model
```

> ⚠️ **Important**: Model must be in `.h5` format and compatible with TensorFlow 2.15

---

## 🔧 ERPNext Configuration

### Step 1: Run Setup Script

```bash
cd backend
python erpnext_setup.py
```

This script will:
1. ✅ Create the `AI_Document` DocType in ERPNext
2. ✅ Generate API credentials (API Key + Secret)
3. ✅ Test connection with a test document

### Step 2: Configure Environment Variables

Create a `.env` file in the `backend/` folder:

```bash
# backend/.env
ERPNEXT_URL=http://localhost:8080
ERPNEXT_API_KEY=your_generated_api_key
ERPNEXT_API_SECRET=your_generated_api_secret
```

> 💡 **Tip**: Credentials are displayed at the end of the `erpnext_setup.py` script

### `AI_Document` DocType Structure

| Field | Type | Description |
|-------|------|-------------|
| `filename` | Data | File name (unique key) |
| `document_class` | Select | Invoice/Drawing/Report/Receipt |
| `file_hash` | Data | SHA-256 hash (duplicate detection) |
| `confidence_score` | Float | Confidence score (0-1) |
| `keywords` | Small Text | Extracted keywords |
| `summary` | Long Text | Classification summary |
| `ocr_text` | Long Text | OCR extracted text |
| `uploaded_by` | Link (User) | User |
| `upload_date` | Datetime | Upload date |
| `is_encrypted` | Check | Encrypted document? |

---

## 🚀 Usage

### Start Server

```bash
cd backend
python main.py
```

Server starts on **http://127.0.0.1:8000**

### Access Interfaces

| Interface | URL | Description |
|-----------|-----|-------------|
| **Dashboard** | http://127.0.0.1:8000/ | User interface |
| **API Docs** | http://127.0.0.1:8000/api/v1/docs | Swagger documentation |
| **Health Check** | http://127.0.0.1:8000/api/v1/health | API status |

### 🔐 Authentication

1. **Login** via API:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "arkeyez2025"}'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "username": "admin"
}
```

2. **Use token** for requests:

```bash
curl -X GET "http://127.0.0.1:8000/api/v1/status" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."
```

---

## 📡 API Documentation

### Main Endpoints

#### 1️⃣ Document Classification

**POST** `/api/v1/classify-multi`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/classify-multi" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "files=@invoice.pdf" \
  -F "files=@drawing.jpg"
```

**Response:**
```json
{
  "results": [
    {
      "filename": "invoice.pdf - Page 1",
      "document_class": "Invoice",
      "confidence": 0.92,
      "cnn_confidence": 0.89,
      "ocr_boost": 0.03,
      "fusion_applied": true,
      "keywords": ["invoice", "amount", "total", "tax", "client"],
      "summary": "Page 1: Invoice (92.0%) [Fusion: +3.0%]",
      "ocr_text": "INVOICE N° 2024-001...",
      "page_number": 1,
      "image_base64": "data:image/jpeg;base64,/9j/4AAQ..."
    }
  ],
  "total_files": 1,
  "total_pages": 1,
  "is_simulation": false,
  "fusion_enabled": true,
  "timestamp": "2025-11-14T15:30:00"
}
```

#### 2️⃣ Insert into ERPNext

**POST** `/api/v1/erpnext/insert`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/erpnext/insert" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "invoice_2024_001.pdf",
    "document_class": "Invoice",
    "confidence_score": 0.92,
    "keywords": ["invoice", "total"],
    "summary": "Invoice (92%)",
    "ocr_text": "INVOICE...",
    "uploaded_by": "Administrator"
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "Document successfully inserted into ERPNext",
  "inserted_id": "invoice_2024_001.pdf",
  "erpnext_name": "invoice_2024_001.pdf"
}
```

#### 3️⃣ Document History

**GET** `/api/v1/erpnext/history?limit=50`

#### 4️⃣ Statistics

**GET** `/api/v1/erpnext/stats`

```json
{
  "success": true,
  "statistics": {
    "total": 120,
    "by_class": {
      "Invoice": 45,
      "Drawing": 30,
      "Report": 25,
      "Receipt": 20
    },
    "avg_confidence": 0.87
  },
  "source": "erpnext"
}
```

### WebSocket Endpoint

**WS** `/ws/classify`

```javascript
const ws = new WebSocket('ws://127.0.0.1:8000/ws/classify');

// Send image
ws.send(JSON.stringify({
  type: "classify",
  image: "data:image/jpeg;base64,/9j/4AAQ...",
  filename: "document.jpg"
}));

// Receive updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === "progress") {
    console.log(`${data.step}: ${data.progress}%`);
  } else if (data.type === "result") {
    console.log("Classification complete:", data.data);
  }
};
```

---

## 📂 Project Structure

```
CNN-Fastapi-ERPnext/
├── backend/
│   ├── main.py                 # Main FastAPI application
│   ├── models.py               # ModelManager (CNN)
│   ├── database.py             # DatabaseManager (local SQLite)
│   ├── auth.py                 # JWT authentication
│   ├── ocr_nlp.py              # OCR + NLP pipeline
│   ├── erpnext_connector.py    # ERPNext REST API client
│   ├── erpnext_setup.py        # ERPNext configuration script
│   ├── middleware.py           # Request logging
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Configuration (git ignored)
│   └── archive.db              # Local database (fallback)
├── frontend/
│   ├── dashboard.html          # Main dashboard
│   ├── styles.css              # CSS styles
│   └── script.js               # JavaScript logic
├── models/
│   └── final_model_complete.h5 # Trained CNN model
├── dataset/                    # Training dataset (optional)
├── output/                     # Test outputs
├── README.md                   # This file
└── .gitignore
```

---

## 🐳 Deployment

### Option 1: Linux Server (Ubuntu)

```bash
# Install Python 3.10+
sudo apt update
sudo apt install python3.10 python3-pip

# Clone and install
git clone https://github.com/Yosra-Megbli/CNN-Fastapi-ERPnext.git
cd CNN-Fastapi-ERPnext/backend
pip3 install -r requirements.txt

# Configure .env
nano .env

# Launch with systemd
sudo nano /etc/systemd/system/arkeyezdoc.service
```

**Service file:**
```ini
[Unit]
Description=ArkeyezDoc API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/CNN-Fastapi-ERPnext/backend
Environment="PATH=/path/to/env/bin"
ExecStart=/path/to/env/bin/python main.py

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl start arkeyezdoc
sudo systemctl enable arkeyezdoc
```

### Option 2: Docker (Coming Soon)

---

## 🐛 Troubleshooting

### ❌ Problem: "ERPNext not connected"

**Solution:**
1. Verify ERPNext is accessible: `curl http://localhost:8080`
2. Check credentials in `.env`
3. Test connection: `GET /api/v1/debug/erpnext`

### ❌ Problem: "Model loading timeout"

**Solution:**
1. Verify `.h5` file exists: `ls -lh models/`
2. Check available RAM: `free -h`
3. Simulation mode activates automatically on failure

### ❌ Problem: "OCR not available"

**Solution:**
```bash
pip install easyocr opencv-python-headless
```

### ❌ Problem: WebSocket disconnected

**Solution:**
1. Check CORS in `main.py`
2. Use `ws://` (not `wss://`) locally
3. Check logs: `/api/v1/debug/model`

---

## 📝 Changelog

### v2.0.0 (2025-11-14)
- ✨ Native ERPNext integration
- ✨ Real-time WebSocket streaming
- ✨ CNN + OCR/NLP fusion
- 🐛 Fix: Asynchronous model loading
- 📚 Complete documentation

### v1.0.0 (2024)
- 🎉 Initial release
- ✅ Basic CNN classification
- ✅ REST API

---

## 👥 Authors

- **Yosra Megbli** - [@Yosra-Megbli](https://github.com/Yosra-Megbli)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- **ERPNext**: Open-source ERP framework
- **FastAPI**: Modern web framework
- **TensorFlow**: Deep learning
- **EasyOCR**: Open-source OCR

---

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/Yosra-Megbli/CNN-Fastapi-ERPnext/issues)
- 📧 **Email**: support@arkeyezdoc.com
- 💬 **Discord**: [Join Server](#)

---

<div align="center">
  <strong>Made with ❤️ by Yosra Megbli</strong>
  <br><br>
  <a href="https://github.com/Yosra-Megbli/CNN-Fastapi-ERPnext">⭐ Star this project</a>
</div>
