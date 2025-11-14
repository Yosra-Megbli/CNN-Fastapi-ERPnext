import React, { useState, useEffect, useRef } from 'react';
import { Upload, CheckCircle, AlertCircle, TrendingUp, FileText, Database, BarChart3, Clock, Zap, Loader2 } from 'lucide-react';

const API_URL = 'http://localhost:8000';

export default function IntelligentArchiveDashboard() {
  const [token, setToken] = useState(null);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [modelStatus, setModelStatus] = useState({ loaded: false, progress: 0, mode: 'simulation' });
  const [uploadedFile, setUploadedFile] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);
  const [notification, setNotification] = useState(null);
  const fileInputRef = useRef(null);

  // Check model status every 2 seconds
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch(`${API_URL}/api/v1/status`);
        const data = await res.json();
        setModelStatus({
          loaded: data.model_loaded,
          progress: data.load_progress,
          mode: data.mode
        });
      } catch (err) {
        console.error('Status check failed:', err);
      }
    };

    if (token) {
      checkStatus();
      const interval = setInterval(checkStatus, 2000);
      return () => clearInterval(interval);
    }
  }, [token]);

  // Load history and stats
  useEffect(() => {
    if (token) {
      loadHistory();
      loadStats();
    }
  }, [token]);

  const showNotification = (message, type = 'success') => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 4000);
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch(`${API_URL}/api/v1/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      if (res.ok) {
        const data = await res.json();
        setToken(data.access_token);
        showNotification('✅ Connexion réussie!');
      } else {
        showNotification('❌ Identifiants invalides', 'error');
      }
    } catch (err) {
      showNotification('❌ Erreur de connexion', 'error');
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedFile(file);
      setClassificationResult(null);
    }
  };

  const handleClassify = async () => {
    if (!uploadedFile) return;

    setIsClassifying(true);
    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const res = await fetch(`${API_URL}/api/v1/classify`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData
      });
      const data = await res.json();
      setClassificationResult(data);
      showNotification('✅ Classification terminée!');
    } catch (err) {
      showNotification('❌ Erreur de classification', 'error');
    } finally {
      setIsClassifying(false);
    }
  };

  const handleInsertToERPNext = async () => {
    if (!classificationResult) return;

    try {
      const res = await fetch(`${API_URL}/api/v1/erpnext/insert`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          document_class: classificationResult.document_class,
          filename: uploadedFile.name,
          confidence_score: classificationResult.confidence,
          keywords: classificationResult.keywords,
          summary: classificationResult.summary,
          ocr_text: classificationResult.ocr_text
        })
      });
      const data = await res.json();
      if (data.success) {
        showNotification(`✅ Document inséré dans ${classificationResult.document_class}!`);
        loadHistory();
        loadStats();
      }
    } catch (err) {
      showNotification('❌ Erreur d\'insertion', 'error');
    }
  };

  const loadHistory = async () => {
    try {
      const res = await fetch(`${API_URL}/api/v1/erpnext/history?limit=10`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await res.json();
      setHistory(data.documents || []);
    } catch (err) {
      console.error('Failed to load history:', err);
    }
  };

  const loadStats = async () => {
    try {
      const res = await fetch(`${API_URL}/api/v1/erpnext/stats`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await res.json();
      setStats(data.statistics);
    } catch (err) {
      console.error('Failed to load stats:', err);
    }
  };

  // Login Screen
  if (!token) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-600 via-pink-500 to-orange-400 flex items-center justify-center p-4">
        <div className="backdrop-blur-xl bg-white/10 rounded-3xl p-8 shadow-2xl border border-white/20 max-w-md w-full">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-white mb-2">ARKEYEZ</h1>
            <p className="text-white/80">Intelligent Archive Indexer</p>
          </div>
          <form onSubmit={handleLogin} className="space-y-4">
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full px-4 py-3 rounded-xl bg-white/20 backdrop-blur-sm border border-white/30 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-white/50"
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-3 rounded-xl bg-white/20 backdrop-blur-sm border border-white/30 text-white placeholder-white/60 focus:outline-none focus:ring-2 focus:ring-white/50"
            />
            <button
              type="submit"
              className="w-full py-3 bg-white text-purple-600 font-bold rounded-xl hover:bg-white/90 transition-all transform hover:scale-105"
            >
              Se connecter
            </button>
          </form>
          <p className="text-white/60 text-sm text-center mt-4">
            Demo: admin / arkeyez2025
          </p>
        </div>
      </div>
    );
  }

  // Main Dashboard
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 via-pink-500 to-orange-400 p-6">
      {/* Notification */}
      {notification && (
        <div className={`fixed top-6 right-6 z-50 backdrop-blur-xl rounded-2xl p-4 shadow-2xl border animate-slide-in ${
          notification.type === 'success' 
            ? 'bg-green-500/20 border-green-300/30 text-white' 
            : 'bg-red-500/20 border-red-300/30 text-white'
        }`}>
          {notification.message}
        </div>
      )}

      {/* Header */}
      <div className="backdrop-blur-xl bg-white/10 rounded-3xl p-6 mb-6 shadow-2xl border border-white/20">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-1">Intelligent Archive Indexer</h1>
            <p className="text-white/80">AI-Powered Document Classification</p>
          </div>
          <div className="text-right">
            {modelStatus.loaded ? (
              <div className="flex items-center gap-2 text-white">
                <CheckCircle className="text-green-300" size={24} />
                <span className="font-semibold">Model Ready</span>
              </div>
            ) : (
              <div className="text-white">
                <div className="flex items-center gap-2 mb-2">
                  <Loader2 className="animate-spin text-yellow-300" size={20} />
                  <span className="text-sm">Using Simulation Mode</span>
                </div>
                <div className="bg-white/20 rounded-full h-2 w-48">
                  <div 
                    className="bg-gradient-to-r from-yellow-300 to-green-300 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${modelStatus.progress * 100}%` }}
                  />
                </div>
                <p className="text-xs text-white/70 mt-1">{Math.round(modelStatus.progress * 100)}% loaded</p>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Section */}
        <div className="lg:col-span-2">
          <div className="backdrop-blur-xl bg-white/10 rounded-3xl p-6 shadow-2xl border border-white/20 mb-6">
            <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
              <Upload size={24} />
              Upload Document
            </h2>
            
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-white/30 rounded-2xl p-12 text-center cursor-pointer hover:border-white/60 hover:bg-white/5 transition-all"
            >
              <Upload className="mx-auto mb-4 text-white/60" size={48} />
              <p className="text-white font-semibold mb-2">
                {uploadedFile ? uploadedFile.name : 'Click to upload or drag & drop'}
              </p>
              <p className="text-white/60 text-sm">PNG, JPG, JPEG (max 10MB)</p>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />

            {uploadedFile && (
              <button
                onClick={handleClassify}
                disabled={isClassifying}
                className="w-full mt-4 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-bold rounded-xl hover:from-blue-600 hover:to-purple-700 transition-all transform hover:scale-105 disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {isClassifying ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    Classification en cours...
                  </>
                ) : (
                  <>
                    <Zap size={20} />
                    Classifier le document
                  </>
                )}
              </button>
            )}
          </div>

          {/* Results */}
          {classificationResult && (
            <div className="backdrop-blur-xl bg-white/10 rounded-3xl p-6 shadow-2xl border border-white/20">
              <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                <FileText size={24} />
                Résultats de Classification
              </h2>
              
              {classificationResult.is_simulation && (
                <div className="bg-yellow-500/20 border border-yellow-300/30 rounded-xl p-3 mb-4 text-white text-sm">
                  ⚠️ Mode simulation actif - Le modèle réel est en cours de chargement
                </div>
              )}

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="bg-white/10 rounded-xl p-4">
                  <p className="text-white/70 text-sm mb-1">Classe détectée</p>
                  <p className="text-2xl font-bold text-white">{classificationResult.document_class}</p>
                </div>
                <div className="bg-white/10 rounded-xl p-4">
                  <p className="text-white/70 text-sm mb-1">Confiance</p>
                  <p className="text-2xl font-bold text-white">{(classificationResult.confidence * 100).toFixed(1)}%</p>
                </div>
              </div>

              {classificationResult.keywords.length > 0 && (
                <div className="mb-4">
                  <p className="text-white/70 text-sm mb-2">Mots-clés</p>
                  <div className="flex flex-wrap gap-2">
                    {classificationResult.keywords.map((kw, i) => (
                      <span key={i} className="px-3 py-1 bg-white/20 rounded-full text-white text-sm">
                        {kw}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="mb-4">
                <p className="text-white/70 text-sm mb-2">Résumé</p>
                <p className="text-white">{classificationResult.summary}</p>
              </div>

              <button
                onClick={handleInsertToERPNext}
                className="w-full py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white font-bold rounded-xl hover:from-green-600 hover:to-emerald-700 transition-all transform hover:scale-105 flex items-center justify-center gap-2"
              >
                <Database size={20} />
                Insérer dans ERPNext
              </button>
            </div>
          )}
        </div>

        {/* Stats & History Sidebar */}
        <div className="space-y-6">
          {/* Stats */}
          {stats && (
            <div className="backdrop-blur-xl bg-white/10 rounded-3xl p-6 shadow-2xl border border-white/20">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <BarChart3 size={20} />
                Statistiques
              </h3>
              <div className="space-y-3">
                <div className="bg-white/10 rounded-xl p-3">
                  <p className="text-white/70 text-sm">Total Documents</p>
                  <p className="text-2xl font-bold text-white">{stats.total}</p>
                </div>
                {Object.entries(stats.by_class).map(([cls, count]) => (
                  <div key={cls} className="flex justify-between items-center bg-white/5 rounded-lg p-3">
                    <span className="text-white">{cls}</span>
                    <span className="text-white font-bold">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* History */}
          <div className="backdrop-blur-xl bg-white/10 rounded-3xl p-6 shadow-2xl border border-white/20">
            <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <Clock size={20} />
              Historique Récent
            </h3>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {history.map((doc, i) => (
                <div key={i} className="bg-white/10 rounded-lg p-3 hover:bg-white/20 transition-all">
                  <div className="flex justify-between items-start mb-1">
                    <span className="text-white font-semibold text-sm">{doc.document_class}</span>
                    <span className="text-white/60 text-xs">{doc.confidence_score.toFixed(2)}</span>
                  </div>
                  <p className="text-white/80 text-xs truncate">{doc.filename}</p>
                  <p className="text-white/50 text-xs">{new Date(doc.upload_date).toLocaleString('fr-FR')}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes slide-in {
          from { transform: translateX(100%); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
        .animate-slide-in {
          animation: slide-in 0.3s ease-out;
        }
      `}</style>
    </div>
  );
}