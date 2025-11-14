# 🚀 ArkeyezDoc v2.0 - AI-Powered Document Classification for ERPNext

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![ERPNext](https://img.shields.io/badge/ERPNext-Integration-blue?style=flat)](https://erpnext.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)](https://www.python.org/)

> **Système intelligent de classification de documents basé sur CNN + OCR/NLP**, avec intégration native ERPNext et streaming temps réel WebSocket.

---

## 📋 Table des Matières

- [Vue d'ensemble](#-vue-densemble)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration ERPNext](#-configuration-erpnext)
- [Utilisation](#-utilisation)
- [API Documentation](#-api-documentation)
- [Structure du Projet](#-structure-du-projet)
- [Déploiement](#-déploiement)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Vue d'ensemble

**ArkeyezDoc** est un système d'IA qui classifie automatiquement les documents (factures, dessins techniques, rapports, reçus) en utilisant :

- **CNN (ResNet50)** : Classification visuelle par deep learning
- **OCR (EasyOCR)** : Extraction de texte multilingue (FR/EN)
- **NLP** : Analyse sémantique et extraction de mots-clés
- **Fusion CNN+OCR** : Combinaison intelligente pour boost de précision
- **ERPNext Integration** : Insertion directe dans ERPNext via REST API
- **WebSocket Streaming** : Suivi en temps réel du traitement

### 📊 Performances

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 85-92% |
| **Classes supportées** | 4 (Invoice, Drawing, Report, Receipt) |
| **Temps de traitement** | ~2-3s par document |
| **Formats supportés** | PDF, JPG, PNG |

---

## ✨ Fonctionnalités

### 🔥 Core Features

- ✅ **Classification multi-documents** : Traitement par batch (PDF multi-pages supporté)
- ✅ **Fusion CNN+OCR** : Boost de confiance jusqu'à +8% grâce à l'analyse textuelle
- ✅ **ERPNext Native** : Insertion directe dans le DocType `AI_Document`
- ✅ **WebSocket Real-time** : Suivi progressif (0% → 100%) avec étapes détaillées
- ✅ **Authentication JWT** : API sécurisée avec tokens Bearer
- ✅ **Dashboard Web** : Interface utilisateur moderne et responsive
- ✅ **API REST complète** : Swagger UI intégré

### 🔧 Fonctionnalités Avancées

- 🔄 **Détection de doublons** : Hash SHA-256 pour éviter les duplications
- 📈 **Statistiques en temps réel** : Nombre de documents par classe, confiance moyenne
- 🐛 **Debug Endpoints** : Diagnostics modèle et connexion ERPNext
- 🔐 **Sécurité** : Support du chiffrement de documents sensibles
- 📝 **Extraction de métadonnées** : Dates, montants, références automatiquement détectés

---

## 🏗️ Architecture

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

### 📦 Composants

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| **Backend API** | FastAPI | Orchestration générale |
| **Model Manager** | TensorFlow/Keras | Chargement & inférence CNN |
| **OCR Pipeline** | EasyOCR | Extraction de texte |
| **NLP Engine** | TF-IDF | Extraction keywords |
| **ERPNext Connector** | Requests | Communication REST avec ERPNext |
| **WebSocket Manager** | FastAPI WebSocket | Streaming temps réel |
| **Database** | SQLite (local) / ERPNext | Stockage documents |

---

## 📥 Installation

### 1️⃣ Prérequis

- **Python 3.10+**
- **ERPNext v14+** (serveur local ou distant)
- **4GB RAM minimum** (8GB recommandé pour le modèle)
- **Git**

### 2️⃣ Cloner le Repository

```bash
git clone https://github.com/Yosra-Megbli/CNN-Fastapi-ERPnext.git
cd CNN-Fastapi-ERPnext
```

### 3️⃣ Créer l'Environnement Virtuel

```bash
# Créer l'environnement
python -m venv env

# Activer (Windows)
env\Scripts\activate

# Activer (Linux/Mac)
source env/bin/activate
```

### 4️⃣ Installer les Dépendances

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 5️⃣ Configuration du Modèle

Placez votre modèle CNN dans le dossier `models/` :

```bash
models/
└── final_model_complete.h5  # Votre modèle entraîné
```

> ⚠️ **Important** : Le modèle doit être au format `.h5` et compatible TensorFlow 2.15

---

## 🔧 Configuration ERPNext

### Étape 1 : Lancer le Script de Setup

```bash
cd backend
python erpnext_setup.py
```

Ce script va :
1. ✅ Créer le DocType `AI_Document` dans ERPNext
2. ✅ Générer les API credentials (API Key + Secret)
3. ✅ Tester la connexion avec un document de test

### Étape 2 : Configurer les Variables d'Environnement

Créez un fichier `.env` dans le dossier `backend/` :

```bash
# backend/.env
ERPNEXT_URL=http://localhost:8080
ERPNEXT_API_KEY=votre_api_key_generee
ERPNEXT_API_SECRET=votre_api_secret_genere
```

> 💡 **Astuce** : Les credentials sont affichés à la fin du script `erpnext_setup.py`

### Structure du DocType `AI_Document`

| Champ | Type | Description |
|-------|------|-------------|
| `filename` | Data | Nom du fichier (clé unique) |
| `document_class` | Select | Invoice/Drawing/Report/Receipt |
| `file_hash` | Data | SHA-256 hash (détection doublons) |
| `confidence_score` | Float | Score de confiance (0-1) |
| `keywords` | Small Text | Mots-clés extraits |
| `summary` | Long Text | Résumé de la classification |
| `ocr_text` | Long Text | Texte extrait par OCR |
| `uploaded_by` | Link (User) | Utilisateur |
| `upload_date` | Datetime | Date d'upload |
| `is_encrypted` | Check | Document chiffré ? |

---

## 🚀 Utilisation

### Démarrer le Serveur

```bash
cd backend
python main.py
```

Le serveur démarre sur **http://127.0.0.1:8000**

### Accéder aux Interfaces

| Interface | URL | Description |
|-----------|-----|-------------|
| **Dashboard** | http://127.0.0.1:8000/ | Interface utilisateur |
| **API Docs** | http://127.0.0.1:8000/api/v1/docs | Documentation Swagger |
| **Health Check** | http://127.0.0.1:8000/api/v1/health | Status de l'API |

### 🔐 Authentification

1. **Login** via l'API :

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "arkeyez2025"}'
```

Réponse :
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "username": "admin"
}
```

2. **Utiliser le token** pour les requêtes :

```bash
curl -X GET "http://127.0.0.1:8000/api/v1/status" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."
```

---

## 📡 API Documentation

### Endpoints Principaux

#### 1️⃣ Classification de Documents

**POST** `/api/v1/classify-multi`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/classify-multi" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "files=@facture.pdf" \
  -F "files=@dessin.jpg"
```

**Response :**
```json
{
  "results": [
    {
      "filename": "facture.pdf - Page 1",
      "document_class": "Invoice",
      "confidence": 0.92,
      "cnn_confidence": 0.89,
      "ocr_boost": 0.03,
      "fusion_applied": true,
      "keywords": ["facture", "montant", "total", "tva", "client"],
      "summary": "Page 1: Invoice (92.0%) [Fusion: +3.0%]",
      "ocr_text": "FACTURE N° 2024-001...",
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

#### 2️⃣ Insertion dans ERPNext

**POST** `/api/v1/erpnext/insert`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/erpnext/insert" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "facture_2024_001.pdf",
    "document_class": "Invoice",
    "confidence_score": 0.92,
    "keywords": ["facture", "total"],
    "summary": "Invoice (92%)",
    "ocr_text": "FACTURE...",
    "uploaded_by": "Administrator"
  }'
```

**Response :**
```json
{
  "success": true,
  "message": "Document successfully inserted into ERPNext",
  "inserted_id": "facture_2024_001.pdf",
  "erpnext_name": "facture_2024_001.pdf"
}
```

#### 3️⃣ Historique des Documents

**GET** `/api/v1/erpnext/history?limit=50`

#### 4️⃣ Statistiques

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

// Envoyer une image
ws.send(JSON.stringify({
  type: "classify",
  image: "data:image/jpeg;base64,/9j/4AAQ...",
  filename: "document.jpg"
}));

// Recevoir les updates
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

## 📂 Structure du Projet

```
CNN-Fastapi-ERPnext/
├── backend/
│   ├── main.py                 # FastAPI application principale
│   ├── models.py               # ModelManager (CNN)
│   ├── database.py             # DatabaseManager (SQLite local)
│   ├── auth.py                 # JWT authentication
│   ├── ocr_nlp.py              # OCR + NLP pipeline
│   ├── erpnext_connector.py    # ERPNext REST API client
│   ├── erpnext_setup.py        # Script de configuration ERPNext
│   ├── middleware.py           # Request logging
│   ├── requirements.txt        # Dependencies Python
│   ├── .env                    # Configuration (git ignored)
│   └── archive.db              # Base locale (fallback)
├── frontend/
│   ├── dashboard.html          # Dashboard principal
│   ├── styles.css              # Styles CSS
│   └── script.js               # Logique JavaScript
├── models/
│   └── final_model_complete.h5 # Modèle CNN entraîné
├── dataset/                    # Dataset d'entraînement (optionnel)
├── output/                     # Outputs de tests
├── README.md                   # Ce fichier
└── .gitignore
```

---

## 🐳 Déploiement

### Option 1 : Serveur Linux (Ubuntu)

```bash
# Installer Python 3.10+
sudo apt update
sudo apt install python3.10 python3-pip

# Cloner et installer
git clone https://github.com/Yosra-Megbli/CNN-Fastapi-ERPnext.git
cd CNN-Fastapi-ERPnext/backend
pip3 install -r requirements.txt

# Configurer .env
nano .env

# Lancer avec systemd
sudo nano /etc/systemd/system/arkeyezdoc.service
```

**Fichier service :**
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

### Option 2 : Docker (à venir)

---

## 🐛 Troubleshooting

### ❌ Problème : "ERPNext not connected"

**Solution :**
1. Vérifier que ERPNext est accessible : `curl http://localhost:8080`
2. Vérifier les credentials dans `.env`
3. Tester la connexion : `GET /api/v1/debug/erpnext`

### ❌ Problème : "Model loading timeout"

**Solution :**
1. Vérifier que le fichier `.h5` existe : `ls -lh models/`
2. Vérifier la RAM disponible : `free -h`
3. Le mode simulation s'active automatiquement si échec

### ❌ Problème : "OCR not available"

**Solution :**
```bash
pip install easyocr opencv-python-headless
```

### ❌ Problème : WebSocket déconnecté

**Solution :**
1. Vérifier les CORS dans `main.py`
2. Utiliser `ws://` (pas `wss://`) en local
3. Vérifier les logs : `/api/v1/debug/model`

---

## 📝 Changelog

### v2.0.0 (2025-11-14)
- ✨ Intégration ERPNext native
- ✨ WebSocket streaming temps réel
- ✨ Fusion CNN + OCR/NLP
- 🐛 Fix: Chargement asynchrone du modèle
- 📚 Documentation complète

### v1.0.0 (2024)
- 🎉 Version initiale
- ✅ Classification CNN basique
- ✅ API REST

---

## 👥 Auteurs

- **Yosra Megbli** - [@Yosra-Megbli](https://github.com/Yosra-Megbli)

---

## 📄 License

MIT License - Voir [LICENSE](LICENSE) pour détails

---

## 🙏 Remerciements

- **ERPNext** : Framework ERP open-source
- **FastAPI** : Framework web moderne
- **TensorFlow** : Deep learning
- **EasyOCR** : OCR open-source

---

## 📞 Support

- 🐛 **Issues** : [GitHub Issues](https://github.com/Yosra-Megbli/CNN-Fastapi-ERPnext/issues)
- 📧 **Email** : support@arkeyezdoc.com
- 💬 **Discord** : [Rejoindre le serveur](#)

---

<div align="center">
  <strong>Made with ❤️ by Yosra Megbli</strong>
  <br><br>
  <a href="https://github.com/Yosra-Megbli/CNN-Fastapi-ERPnext">⭐ Star ce projet</a>
</div>
