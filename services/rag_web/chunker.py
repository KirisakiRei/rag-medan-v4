"""
RAG Web Service - Chunker Module
Text chunking untuk web content
"""
import re
import logging
from typing import List
from dataclasses import dataclass, field

logger = logging.getLogger("rag_web.chunker")


@dataclass
class Chunk:
    """Chunk data class."""
    content: str
    index: int
    start_char: int
    end_char: int
    token_count: int
    metadata: dict = field(default_factory=dict)


class Chunker:
    """Text chunker untuk web content."""
    
    PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str, url: str = "") -> List[Chunk]:
        """
        Chunk text menjadi bagian-bagian kecil.
        
        Args:
            text: Text yang akan di-chunk
            url: URL untuk logging
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # Split by paragraphs
        paragraphs = self.PARAGRAPH_PATTERN.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks: List[Chunk] = []
        current_chunk = ""
        current_start = 0
        char_position = 0
        
        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            
            # Jika paragraph terlalu besar, split lagi
            if para_tokens > self.chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, len(chunks), current_start, char_position
                    ))
                    current_chunk = ""
                
                # Split paragraph yang besar
                sub_chunks = self._split_large_paragraph(para)
                for sub_chunk in sub_chunks:
                    chunks.append(self._create_chunk(
                        sub_chunk, len(chunks), char_position, char_position + len(sub_chunk)
                    ))
                    char_position += len(sub_chunk) + 2
            else:
                # Try to combine dengan current chunk
                combined = current_chunk + "\n\n" + para if current_chunk else para
                combined_tokens = self._estimate_tokens(combined)
                
                if combined_tokens <= self.chunk_size:
                    if not current_chunk:
                        current_start = char_position
                    current_chunk = combined
                else:
                    # Flush current chunk dan start baru
                    if current_chunk:
                        chunks.append(self._create_chunk(
                            current_chunk, len(chunks), current_start, char_position
                        ))
                    current_chunk = para
                    current_start = char_position
            
            char_position += len(para) + 2
        
        # Flush remaining
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, len(chunks), current_start, char_position
            ))
        
        logger.info(f"[CHUNKER] Created {len(chunks)} chunks")
        return chunks
    
    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split paragraph besar berdasarkan sentence."""
        sentences = self.SENTENCE_PATTERN.split(paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current = ""
        
        for sentence in sentences:
            combined = current + " " + sentence if current else sentence
            if self._estimate_tokens(combined) <= self.chunk_size:
                current = combined
            else:
                if current:
                    chunks.append(current)
                current = sentence if self._estimate_tokens(sentence) <= self.chunk_size else ""
        
        if current:
            chunks.append(current)
        
        return chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate jumlah tokens."""
        if not text:
            return 0
        return max(len(text.split()), len(text) // 4)
    
    def _create_chunk(self, content: str, index: int, start_char: int, end_char: int) -> Chunk:
        """Create Chunk object."""
        return Chunk(
            content=content.strip(),
            index=index,
            start_char=start_char,
            end_char=end_char,
            token_count=self._estimate_tokens(content)
        )


# Singleton instance
chunker = Chunker(chunk_size=512, chunk_overlap=50)
