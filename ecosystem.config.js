/**
 * RAG Medan v3 - PM2 Ecosystem Configuration
 * 
 * Menjalankan semua services dengan PM2
 * 
 * Usage:
 *   pm2 start ecosystem.config.js           # Start all services
 *   pm2 start ecosystem.config.js --only orchestrator
 *   pm2 stop all
 *   pm2 logs orchestrator
 *   pm2 logs rag-text
 *   pm2 status
 *   pm2 restart all
 */

const path = require('path');
const os = require('os');

// ── Ubah HANYA nilai ini untuk mengganti port orchestrator ──
const ORCHESTRATOR_PORT = 5000;

// Detect OS and set correct Python path
const isWindows = os.platform() === 'win32';
const pythonPath = isWindows 
  ? path.join(__dirname, '.venv', 'Scripts', 'python.exe')
  : path.join(__dirname, '.venv', 'bin', 'python');

module.exports = {
  apps: [
    // ============== ORCHESTRATOR ==============
    {
      name: "orchestrator",
      script: pythonPath,
      args: `-m uvicorn orchestrator.orchestrator:app --host 0.0.0.0 --port ${ORCHESTRATOR_PORT}`,
      cwd: __dirname,
      interpreter: "none",
      env: {
        PYTHONPATH: __dirname,
        ORCHESTRATOR_PORT: ORCHESTRATOR_PORT,
        PYTHONMALLOC: "malloc",
        MALLOC_TRIM_THRESHOLD_: "100000"
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "./logs/orchestrator-error.log",
      out_file: "./logs/orchestrator-out.log",
      merge_logs: true
    },
    
    // ============== RAG TEXT SERVICE ==============
    {
      name: "rag-text",
      script: pythonPath,
      args: "-m uvicorn services.rag_text.main:app --host 0.0.0.0 --port 5010",
      cwd: __dirname,
      interpreter: "none",
      env: {
        PYTHONPATH: __dirname,
        TEXT_SERVICE_PORT: 5010,
        PYTHONMALLOC: "malloc",
        MALLOC_TRIM_THRESHOLD_: "100000"
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "2G",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "./logs/rag-text-error.log",
      out_file: "./logs/rag-text-out.log",
      merge_logs: true
    },
    
    // ============== RAG DOCUMENT SERVICE ==============
    {
      name: "rag-document",
      script: pythonPath,
      args: "-m uvicorn services.rag_document.main:app --host 0.0.0.0 --port 5011",
      cwd: __dirname,
      interpreter: "none",
      env: {
        PYTHONPATH: __dirname,
        DOCUMENT_SERVICE_PORT: 5011,
        PYTHONMALLOC: "malloc",
        MALLOC_TRIM_THRESHOLD_: "100000"
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "3G",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "./logs/rag-document-error.log",
      out_file: "./logs/rag-document-out.log",
      merge_logs: true
    },
    
    // ============== RAG WEB SERVICE ==============
    {
      name: "rag-web",
      script: pythonPath,
      args: "-m uvicorn services.rag_web.main:app --host 0.0.0.0 --port 5012",
      cwd: __dirname,
      interpreter: "none",
      env: {
        PYTHONPATH: __dirname,
        WEB_SERVICE_PORT: 5012,
        PYTHONMALLOC: "malloc",
        MALLOC_TRIM_THRESHOLD_: "100000"
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "2G",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "./logs/rag-web-error.log",
      out_file: "./logs/rag-web-out.log",
      merge_logs: true
    },
    
    // ============== RAG USULAN SERVICE ==============
    {
      name: "rag-usulan",
      script: pythonPath,
      args: "-m uvicorn services.rag_usulan.main:app --host 0.0.0.0 --port 5013",
      cwd: __dirname,
      interpreter: "none",
      env: {
        PYTHONPATH: __dirname,
        USULAN_SERVICE_PORT: 5013,
        PYTHONMALLOC: "malloc",
        MALLOC_TRIM_THRESHOLD_: "100000"
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "2G",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "./logs/rag-usulan-error.log",
      out_file: "./logs/rag-usulan-out.log",
      merge_logs: true
    },
    
    // ============== EMBEDDING SERVICE (optional, enable with USE_SHARED_EMBEDDING=true) ==============
    {
      name: "embedding-service",
      script: pythonPath,
      args: "-m uvicorn services.embedding_service.main:app --host 0.0.0.0 --port 5014",
      cwd: __dirname,
      interpreter: "none",
      env: {
        PYTHONPATH: __dirname,
        EMBEDDING_SERVICE_PORT: 5014,
        USE_SHARED_EMBEDDING: "true",
        PYTHONMALLOC: "malloc",
        MALLOC_TRIM_THRESHOLD_: "100000"
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "3G",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "./logs/embedding-service-error.log",
      out_file: "./logs/embedding-service-out.log",
      merge_logs: true
    },

    // ============== LIGHTRAG SERVER (v4) ==============
    {
      name: "lightrag-server",
      script: path.join(__dirname, '.venv', isWindows ? 'Scripts' : 'bin', isWindows ? 'lightrag-server.exe' : 'lightrag-server'),
      args: "",
      cwd: path.join(__dirname, 'lightrag'),
      interpreter: "none",
      env: {
        PYTHONMALLOC: "malloc",
        MALLOC_TRIM_THRESHOLD_: "100000"
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "4G",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "./logs/lightrag-server-error.log",
      out_file: "./logs/lightrag-server-out.log",
      merge_logs: true
    },

    // ============== LIGHTRAG ADAPTER (v4) ==============
    {
      name: "lightrag-adapter",
      script: pythonPath,
      args: "-m uvicorn services.lightrag_adapter.main:app --host 0.0.0.0 --port 5015",
      cwd: __dirname,
      interpreter: "none",
      env: {
        PYTHONPATH: __dirname,
        LIGHTRAG_ADAPTER_PORT: 5015,
        PYTHONMALLOC: "malloc",
        MALLOC_TRIM_THRESHOLD_: "100000"
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "./logs/lightrag-adapter-error.log",
      out_file: "./logs/lightrag-adapter-out.log",
      merge_logs: true
    }
  ]
};
