import os
import sys
import argparse
import logging
import uuid
import tempfile
import requests
from typing import List
from datetime import datetime

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from shared.logging_config import setup_logging
from shared.ocr_utils import (
    extract_text_from_pdf,
    extract_text_from_image,
    extract_text_from_docx,
    extract_text_from_xlsx
)

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import PointStruct, Distance, VectorParams

# Setup logging
logger = setup_logging("document_worker")


def download_file(url: str, dest_path: str) -> bool:
    """Download file dari URL."""
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False


def get_file_extension(url: str) -> str:
    """Get file extension dari URL."""
    path = url.split('?')[0]
    ext = os.path.splitext(path)[1].lower()
    return ext if ext else '.pdf'


def extract_text(file_path: str) -> str:
    """Extract text dari file berdasarkan extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return extract_text_from_image(file_path)
    elif ext in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif ext in ['.xlsx', '.xls']:
        return extract_text_from_xlsx(file_path)
    else:
        # Default to PDF
        return extract_text_from_pdf(file_path)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """Split text into chunks."""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        
        # Find natural break point
        if end < text_len:
            # Try to find paragraph break
            break_point = text.rfind('\n\n', start, end)
            if break_point == -1 or break_point <= start:
                # Try sentence break
                break_point = text.rfind('. ', start, end)
            if break_point == -1 or break_point <= start:
                break_point = end
            else:
                break_point += 1  # Include the break character
            end = break_point
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < text_len else text_len
    
    return chunks


def update_task_status(task_id: str, status: str, progress: str = None, 
                       error: str = None, chunks_count: int = 0):
    """Update task status via sync module."""
    # Import di sini untuk avoid circular import
    from services.rag_document.sync import update_task
    
    updates = {"status": status}
    if progress:
        updates["progress"] = progress
    if error:
        updates["error"] = error
    if chunks_count:
        updates["chunks_count"] = chunks_count
    
    update_task(task_id, updates)


def process_document(task_id: str, doc_id: str, file_url: str, opd_name: str):
    """
    Main processing function.
    1. Download file
    2. OCR/Extract text
    3. Chunk text
    4. Generate embeddings
    5. Store to Qdrant
    """
    logger.info(f"[WORKER] Starting task {task_id} for doc_id={doc_id}")
    
    # Initialize
    model = None
    qdrant = None
    temp_file = None
    
    try:
        # Update status
        update_task_status(task_id, "processing", "Initializing...")
        
        # Load model
        logger.info("[WORKER] Loading embedding model...")
        model = SentenceTransformer(config.EMBEDDING_MODEL_PATH_LARGE)
        
        # Connect to Qdrant
        logger.info("[WORKER] Connecting to Qdrant...")
        if config.QDRANT_API_KEY:
            qdrant = QdrantClient(
                host=config.QDRANT_HOST,
                port=config.QDRANT_PORT,
                api_key=config.QDRANT_API_KEY
            )
        else:
            qdrant = QdrantClient(
                host=config.QDRANT_HOST,
                port=config.QDRANT_PORT
            )
        
        # Ensure collection exists
        collections = qdrant.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if config.COLLECTION_DOCUMENT not in collection_names:
            logger.info(f"[WORKER] Creating collection {config.COLLECTION_DOCUMENT}")
            qdrant.create_collection(
                collection_name=config.COLLECTION_DOCUMENT,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIMENSION_LARGE,
                    distance=Distance.COSINE
                )
            )
        
        # Download file
        update_task_status(task_id, "processing", "Downloading file...")
        logger.info(f"[WORKER] Downloading {file_url}")
        
        ext = get_file_extension(file_url)
        temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        if not download_file(file_url, temp_path):
            raise Exception("Failed to download file")
        
        # OCR/Extract
        update_task_status(task_id, "processing", "Extracting text...")
        logger.info("[WORKER] Extracting text...")
        
        text = extract_text(temp_path)
        
        if not text or len(text.strip()) < 50:
            raise Exception("No text extracted or content too short")
        
        logger.info(f"[WORKER] Extracted {len(text)} characters")
        
        # Chunk
        update_task_status(task_id, "processing", "Chunking text...")
        logger.info("[WORKER] Chunking text...")
        
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        
        if not chunks:
            raise Exception("No chunks created")
        
        logger.info(f"[WORKER] Created {len(chunks)} chunks")
        
        # Delete existing chunks for this doc_id
        update_task_status(task_id, "processing", "Removing old chunks...")
        
        try:
            qdrant.delete(
                collection_name=config.COLLECTION_DOCUMENT,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="doc_id",
                                match=qdrant_models.MatchValue(value=doc_id)
                            )
                        ]
                    )
                )
            )
        except Exception:
            pass  # Might not exist
        
        # Generate embeddings dan store
        update_task_status(task_id, "processing", "Generating embeddings...")
        logger.info("[WORKER] Generating embeddings and storing...")
        
        now = datetime.utcnow().isoformat()
        points = []
        
        for i, chunk_text in enumerate(chunks):
            # E5 model requires 'passage:' prefix for documents
            embedding = model.encode(f"passage: {chunk_text}", convert_to_numpy=True).tolist()
            point_id = str(uuid.uuid4())
            
            payload = {
                "doc_id": doc_id,
                "opd_name": opd_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "content": chunk_text,
                "is_deleted": False,
                "created_at": now,
                "updated_at": now
            }
            
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
            
            # Batch upsert
            if len(points) >= 50:
                qdrant.upsert(collection_name=config.COLLECTION_DOCUMENT, points=points)
                points = []
                update_task_status(task_id, "processing", 
                                   f"Stored {i+1}/{len(chunks)} chunks...")
        
        # Final batch
        if points:
            qdrant.upsert(collection_name=config.COLLECTION_DOCUMENT, points=points)
        
        # Success
        update_task_status(task_id, "completed", "Processing completed", 
                          chunks_count=len(chunks))
        logger.info(f"[WORKER] Completed task {task_id}: {len(chunks)} chunks stored")
        
    except Exception as e:
        logger.exception(f"[WORKER] Error: {e}")
        update_task_status(task_id, "failed", error=str(e))
    
    finally:
        # Cleanup temp file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except:
                pass


def main():
    """Main entry point untuk worker."""
    parser = argparse.ArgumentParser(description="Document OCR Worker")
    parser.add_argument("--task_id", required=True, help="Task ID")
    parser.add_argument("--doc_id", required=True, help="Document ID")
    parser.add_argument("--file_url", required=True, help="File URL")
    parser.add_argument("--opd_name", default="", help="OPD Name")
    
    args = parser.parse_args()
    
    process_document(
        task_id=args.task_id,
        doc_id=args.doc_id,
        file_url=args.file_url,
        opd_name=args.opd_name
    )


if __name__ == "__main__":
    main()
