/**
 * RAG Medan v3 - PM2 Ecosystem Configuration
 * 
 * Menjalankan semua services dengan PM2
 * 
 * Usage:
 *   pm2 start ecosystem.config.js           # Start all services
 *   pm2 start ecosystem.config.js --only orchestrator
 *   pm2 stop all
 *   pm2 logs
 *   pm2 status
 */

module.exports = {
  apps: [
    // ============== ORCHESTRATOR ==============
    {
      name: "orchestrator",
      script: "python",
      args: "-m uvicorn orchestrator.orchestrator:app --host 0.0.0.0 --port 5001",
      cwd: __dirname,
      interpreter: "none",
      env: {
        PYTHONPATH: __dirname,
        ORCHESTRATOR_PORT: 5001
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
      script: "python",
      args: "-m uvicorn services.rag_text.main:app --host 0.0.0.0 --port 5010",
      cwd: __dirname,
      interpreter: "none",
      env: {
        PYTHONPATH: __dirname,
        TEXT_SERVICE_PORT: 5010
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
      script: "python",
      args: "-m uvicorn services.rag_document.main:app --host 0.0.0.0 --port 5011",
      cwd: __dirname,
      interpreter: "none",
      env: {
        PYTHONPATH: __dirname,
        DOCUMENT_SERVICE_PORT: 5011
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
      script: "python",
      args: "-m uvicorn services.rag_web.main:app --host 0.0.0.0 --port 5012",
      cwd: __dirname,
      interpreter: "none",
      env: {
        PYTHONPATH: __dirname,
        WEB_SERVICE_PORT: 5012
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
      script: "python",
      args: "-m uvicorn services.rag_usulan.main:app --host 0.0.0.0 --port 5013",
      cwd: __dirname,
      interpreter: "none",
      env: {
        PYTHONPATH: __dirname,
        USULAN_SERVICE_PORT: 5013
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "2G",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "./logs/rag-usulan-error.log",
      out_file: "./logs/rag-usulan-out.log",
      merge_logs: true
    }
  ]
};
