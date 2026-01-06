"""
RAG Web Service - Cleaner Module
HTML cleaning dan text extraction
"""
import re
import html
import logging
from typing import Optional

from bs4 import BeautifulSoup, Comment

logger = logging.getLogger("rag_web.cleaner")


class Cleaner:
    """HTML cleaner dan text extractor."""
    
    # Tags yang akan dihapus
    REMOVE_TAGS = [
        "script", "style", "noscript", "iframe", "embed", "object",
        "video", "audio", "canvas", "svg", "map", "figure",
        "nav", "header", "footer", "aside", "form", "button",
        "input", "select", "textarea", "label"
    ]
    
    # Pattern untuk non-content elements
    NON_CONTENT_PATTERNS = [
        r"nav", r"menu", r"sidebar", r"footer", r"header",
        r"comment", r"share", r"social", r"related", r"recommend",
        r"advertisement", r"ads", r"banner", r"popup", r"modal",
        r"cookie", r"newsletter", r"subscribe"
    ]
    
    # Classes yang biasanya berisi content
    ARTICLE_CLASSES = ["content", "article", "post", "entry", "body", "text", "main"]
    
    def __init__(self):
        self.non_content_pattern = re.compile(
            "|".join(self.NON_CONTENT_PATTERNS),
            re.IGNORECASE
        )
    
    def clean(self, raw_html: str, url: str = "") -> str:
        """
        Clean HTML dan extract text.
        
        Args:
            raw_html: Raw HTML string
            url: URL untuk logging
            
        Returns:
            Clean text content
        """
        try:
            soup = BeautifulSoup(raw_html, "html.parser")
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Remove unwanted tags
            for tag in self.REMOVE_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Remove non-content elements
            self._remove_non_content_elements(soup)
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Extract text
            clean_text = self._extract_text(content)
            
            # Normalize whitespace
            clean_text = self._normalize_whitespace(clean_text)
            
            logger.info(f"[CLEANER] Cleaned content: {len(clean_text)} chars")
            return clean_text
            
        except Exception as e:
            logger.error(f"[CLEANER] Error: {e}")
            return self._fallback_clean(raw_html)
    
    def _remove_non_content_elements(self, soup: BeautifulSoup) -> None:
        """Remove elements yang kemungkinan bukan content."""
        for element in soup.find_all(True):
            classes = element.get("class", [])
            if classes:
                class_str = " ".join(classes) if isinstance(classes, list) else classes
                if self.non_content_pattern.search(class_str):
                    element.decompose()
                    continue
            
            element_id = element.get("id", "")
            if element_id and self.non_content_pattern.search(element_id):
                element.decompose()
    
    def _extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Extract main content area."""
        # Try article tag
        article = soup.find("article")
        if article and len(article.get_text(strip=True)) > 200:
            return article
        
        # Try main tag
        main = soup.find("main")
        if main and len(main.get_text(strip=True)) > 200:
            return main
        
        # Try common content classes
        for class_name in self.ARTICLE_CLASSES:
            content_div = soup.find("div", class_=re.compile(class_name, re.I))
            if content_div and len(content_div.get_text(strip=True)) > 200:
                return content_div
        
        # Fallback to body
        body = soup.find("body")
        return body if body else soup
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract text dari soup."""
        texts = []
        block_tags = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "blockquote", "pre"]
        
        for element in soup.find_all(block_tags):
            text = element.get_text(separator=" ", strip=True)
            if text and len(text) > 10:
                texts.append(text)
        
        if not texts:
            texts = [soup.get_text(separator=" ", strip=True)]
        
        return "\n\n".join(texts)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace dalam text."""
        text = html.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    
    def _fallback_clean(self, raw_html: str) -> str:
        """Fallback cleaning jika BeautifulSoup gagal."""
        text = re.sub(r"<script[^>]*>.*?</script>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        return self._normalize_whitespace(text)
    
    def extract_title(self, raw_html: str) -> Optional[str]:
        """Extract title dari HTML."""
        soup = BeautifulSoup(raw_html, "html.parser")
        
        # Try title tag
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        
        # Try og:title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()
        
        # Try h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        
        return None


# Singleton instance
cleaner = Cleaner()
