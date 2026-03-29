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
    
    # Classes yang biasanya berisi content (diperluas dengan pola CMS Indonesia)
    ARTICLE_CLASSES = [
        # Generic
        "content", "article", "post", "entry", "body", "text", "main",
        # Joomla
        "item-page", "article-content", "com-content-article",
        # WordPress
        "entry-content", "post-content", "wp-content",
        # Gov / Portal Indonesia custom
        "isi-berita", "konten-berita", "berita-isi", "artikel-isi",
        "konten-utama", "halaman-konten", "isi-konten", "detail-berita",
        "berita-detail", "konten-detail", "isi-artikel", "detail-artikel",
    ]
    
    def __init__(self):
        self.non_content_pattern = re.compile(
            "|".join(self.NON_CONTENT_PATTERNS),
            re.IGNORECASE
        )
    
    def clean_with_selector(self, raw_html: str, css_selector: str, url: str = "") -> str:
        """Clean HTML menggunakan CSS selector. Fallback ke auto-detect jika selector gagal."""
        try:
            soup = BeautifulSoup(raw_html, "html.parser")

            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            for tag in self.REMOVE_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()

            selected = soup.select(css_selector)

            if not selected:
                logger.warning(f"[CLEANER] Selector '{css_selector}' tidak menemukan elemen, fallback ke auto-detect")
                return self.clean(raw_html, url)

            combined_text_parts = []
            for element in selected:
                text = self._extract_text(element)
                text = self._normalize_whitespace(text)
                if text:
                    combined_text_parts.append(text)

            combined = "\n\n".join(combined_text_parts)

            if len(combined.strip()) < 100:
                logger.warning(
                    f"[CLEANER] Selector '{css_selector}' menghasilkan konten terlalu pendek "
                    f"({len(combined)} chars), fallback ke auto-detect"
                )
                return self.clean(raw_html, url)

            logger.info(f"[CLEANER] CSS selector '{css_selector}' berhasil: {len(combined)} chars")
            return combined.strip()

        except Exception as e:
            logger.error(f"[CLEANER] Error pada clean_with_selector: {e}, fallback ke auto-detect")
            return self.clean(raw_html, url)

    def clean(self, raw_html: str, url: str = "") -> str:
        """Clean HTML dan extract text (auto-detect mode)."""
        try:
            soup = BeautifulSoup(raw_html, "html.parser")
            
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            for tag in self.REMOVE_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()
            
            self._remove_non_content_elements(soup)
            content = self._extract_main_content(soup)
            clean_text = self._extract_text(content)
            clean_text = self._normalize_whitespace(clean_text)
            
            logger.info(f"[CLEANER] Cleaned content: {len(clean_text)} chars")
            return clean_text
            
        except Exception as e:
            logger.error(f"[CLEANER] Error: {e}")
            return self._fallback_clean(raw_html)
    
    def _remove_non_content_elements(self, soup: BeautifulSoup) -> None:
        """Remove elements yang kemungkinan bukan content."""
        for element in list(soup.find_all(True)):
            try:
                classes = element.get("class", [])
                if classes:
                    class_str = " ".join(classes) if isinstance(classes, list) else classes
                    if self.non_content_pattern.search(class_str):
                        element.decompose()
                        continue

                element_id = element.get("id", "")
                if element_id and self.non_content_pattern.search(element_id):
                    element.decompose()
            except Exception:
                # Some descendants can become detached after parent decompose; skip safely.
                continue
    
    def _extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Extract main content area dengan fallback bertingkat."""
        # Try article tag
        article = soup.find("article")
        if article and len(article.get_text(strip=True)) > 200:
            return article
        
        # Try main tag
        main = soup.find("main")
        if main and len(main.get_text(strip=True)) > 200:
            return main
        
        # Try common content classes (termasuk CMS Indonesia)
        for class_name in self.ARTICLE_CLASSES:
            content_div = soup.find("div", class_=re.compile(class_name, re.I))
            if content_div and len(content_div.get_text(strip=True)) > 200:
                return content_div

        # Fallback: text-density heuristic
        best_element = None
        best_ratio = 0.0
        for div in soup.find_all("div"):
            html_len = len(str(div))
            text_len = len(div.get_text(strip=True))
            if html_len > 200 and text_len > 150:
                ratio = text_len / html_len
                if ratio > best_ratio and ratio > 0.25:
                    best_ratio = ratio
                    best_element = div

        if best_element is not None:
            logger.debug(f"[CLEANER] Text-density heuristic: ratio={best_ratio:.2f}, chars={len(best_element.get_text(strip=True))}")
            return best_element

        # Final fallback to body
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
