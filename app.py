"""
RAG Medan v3 - Main Entry Point

Script untuk menjalankan semua services atau service tertentu.

Usage:
    python app.py                    # Start interactive menu
    python app.py orchestrator       # Start orchestrator only
    python app.py text               # Start RAG text service only
    python app.py document           # Start RAG document service only
    python app.py web                # Start RAG web service only
    python app.py usulan             # Start RAG usulan service only
"""
import os
import sys
import argparse
import subprocess

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import config


def start_orchestrator():
    """Start orchestrator service."""
    print(f"\n🚀 Starting Orchestrator on port {config.ORCHESTRATOR_PORT}...")
    os.chdir(PROJECT_ROOT)
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "orchestrator.orchestrator:app",
        "--host", config.API_HOST,
        "--port", str(config.ORCHESTRATOR_PORT)
    ])


def start_text_service():
    """Start RAG text service."""
    print(f"\n🚀 Starting RAG Text Service on port {config.TEXT_SERVICE_PORT}...")
    os.chdir(PROJECT_ROOT)
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "services.rag_text.main:app",
        "--host", config.API_HOST,
        "--port", str(config.TEXT_SERVICE_PORT)
    ])


def start_document_service():
    """Start RAG document service."""
    print(f"\n🚀 Starting RAG Document Service on port {config.DOCUMENT_SERVICE_PORT}...")
    os.chdir(PROJECT_ROOT)
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "services.rag_document.main:app",
        "--host", config.API_HOST,
        "--port", str(config.DOCUMENT_SERVICE_PORT)
    ])


def start_web_service():
    """Start RAG web scraping service."""
    print(f"\n🚀 Starting RAG Web Service on port {config.WEB_SERVICE_PORT}...")
    os.chdir(PROJECT_ROOT)
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "services.rag_web.main:app",
        "--host", config.API_HOST,
        "--port", str(config.WEB_SERVICE_PORT)
    ])


def start_usulan_service():
    """Start RAG usulan service."""
    print(f"\n🚀 Starting RAG Usulan Service on port {config.USULAN_SERVICE_PORT}...")
    os.chdir(PROJECT_ROOT)
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "services.rag_usulan.main:app",
        "--host", config.API_HOST,
        "--port", str(config.USULAN_SERVICE_PORT)
    ])


def show_menu():
    """Show interactive menu."""
    print("\n" + "="*60)
    print("🚀 RAG Medan v3 - Service Launcher")
    print("="*60)
    print(f"\n📍 Services Configuration:")
    print(f"   Orchestrator:        http://{config.API_HOST}:{config.ORCHESTRATOR_PORT}")
    print(f"   RAG Text Service:    http://{config.API_HOST}:{config.TEXT_SERVICE_PORT}")
    print(f"   RAG Document Service: http://{config.API_HOST}:{config.DOCUMENT_SERVICE_PORT}")
    print(f"   RAG Web Service:     http://{config.API_HOST}:{config.WEB_SERVICE_PORT}")
    print(f"   RAG Usulan Service:  http://{config.API_HOST}:{config.USULAN_SERVICE_PORT}")
    print("\n📝 Menu:")
    print("   1. Start Orchestrator")
    print("   2. Start RAG Text Service")
    print("   3. Start RAG Document Service")
    print("   4. Start RAG Web Service")
    print("   5. Start RAG Usulan Service")
    print("   0. Exit")
    print("\n" + "="*60)
    
    choice = input("\n   Enter your choice (0-5): ").strip()
    
    if choice == "1":
        start_orchestrator()
    elif choice == "2":
        start_text_service()
    elif choice == "3":
        start_document_service()
    elif choice == "4":
        start_web_service()
    elif choice == "5":
        start_usulan_service()
    elif choice == "0":
        print("\n👋 Goodbye!")
        sys.exit(0)
    else:
        print("\n❌ Invalid choice. Please try again.")
        show_menu()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Medan v3 Service Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python app.py                    # Start interactive menu
    python app.py orchestrator       # Start orchestrator only
    python app.py text               # Start RAG text service only
    python app.py document           # Start RAG document service only
    python app.py web                # Start RAG web service only
    python app.py usulan             # Start RAG usulan service only
        """
    )
    
    parser.add_argument(
        "service",
        nargs="?",
        choices=["orchestrator", "text", "document", "web", "usulan"],
        help="Service to start (optional, shows menu if not provided)"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*60)
    print("🌐 RAG Medan v3 - Modular RAG System")
    print("   No Fallback - Each Service Independent")
    print("="*60)
    
    if args.service is None:
        show_menu()
    elif args.service == "orchestrator":
        start_orchestrator()
    elif args.service == "text":
        start_text_service()
    elif args.service == "document":
        start_document_service()
    elif args.service == "web":
        start_web_service()
    elif args.service == "usulan":
        start_usulan_service()


if __name__ == "__main__":
    main()
