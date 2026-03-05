"""RAG Web Service - FAQ extractor. Supports selector mode and 4-strategy auto-detect."""
import re
import logging
from typing import List, Tuple, Optional

from bs4 import BeautifulSoup

logger = logging.getLogger("rag_web.faq_extractor")

# Minimum panjang teks untuk dianggap valid sebagai pertanyaan / jawaban
_MIN_Q_LEN = 10
_MIN_A_LEN = 20

# Class/ID yang mengindikasikan kontainer FAQ
_FAQ_CONTAINER_CLASSES = re.compile(
    r"faq|accordion|qa|tanya.?jawab|question|jawaban|pertanyaan",
    re.IGNORECASE
)


def extract_faq_pairs(
    raw_html: str,
    question_selector: Optional[str] = None,
    answer_selector: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Ekstrak pasangan (pertanyaan, jawaban) dari HTML."""
    if not raw_html or not raw_html.strip():
        return []

    soup = BeautifulSoup(raw_html, "html.parser")

    # Bersihkan script/style agar tidak masuk ke teks
    for tag in ["script", "style", "noscript"]:
        for el in soup.find_all(tag):
            el.decompose()

    # Mode 1: Operator tentukan selector
    if question_selector and answer_selector:
        pairs = _extract_by_selector(soup, question_selector, answer_selector)
        if pairs:
            logger.info(f"[FAQ] Selector mode: {len(pairs)} pairs ditemukan")
            return pairs
        logger.warning(
            f"[FAQ] Selector mode gagal (q='{question_selector}', a='{answer_selector}'), "
            "lanjut ke auto-detect"
        )

    # Mode 2: Auto-detect — coba semua strategi secara berurutan
    pairs = _extract_dl_pattern(soup)
    if pairs:
        logger.info(f"[FAQ] Auto-detect (dl/dt/dd): {len(pairs)} pairs")
        return pairs

    pairs = _extract_details_pattern(soup)
    if pairs:
        logger.info(f"[FAQ] Auto-detect (details/summary): {len(pairs)} pairs")
        return pairs

    pairs = _extract_faq_container(soup)
    if pairs:
        logger.info(f"[FAQ] Auto-detect (faq container): {len(pairs)} pairs")
        return pairs

    pairs = _extract_heading_paragraph(soup)
    if pairs:
        logger.info(f"[FAQ] Auto-detect (heading+paragraph): {len(pairs)} pairs")
        return pairs

    logger.warning("[FAQ] Tidak ada struktur FAQ yang terdeteksi")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Private extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalisasi whitespace pada teks yang diekstrak."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_by_selector(
    soup: BeautifulSoup,
    q_selector: str,
    a_selector: str,
) -> List[Tuple[str, str]]:
    """Ekstrak pair menggunakan dua CSS selector."""
    try:
        questions = soup.select(q_selector)
        answers = soup.select(a_selector)
    except Exception as e:
        logger.error(f"[FAQ] Error saat select: {e}")
        return []

    if not questions or not answers:
        return []

    pairs = []
    for q_el, a_el in zip(questions, answers):
        q_text = _clean_text(q_el.get_text(separator=" "))
        a_text = _clean_text(a_el.get_text(separator=" "))
        if len(q_text) >= _MIN_Q_LEN and len(a_text) >= _MIN_A_LEN:
            pairs.append((q_text, a_text))

    if len(questions) != len(answers):
        logger.warning(
            f"[FAQ] Jumlah pertanyaan ({len(questions)}) != jawaban ({len(answers)}). "
            f"Hanya {len(pairs)} pasangan yang valid diambil."
        )

    return pairs


def _extract_dl_pattern(soup: BeautifulSoup) -> List[Tuple[str, str]]:
    """Ekstrak dari pola <dl><dt>pertanyaan</dt><dd>jawaban</dd></dl>."""
    pairs = []
    for dl in soup.find_all("dl"):
        children = [c for c in dl.children if c.name in ("dt", "dd")]
        i = 0
        while i < len(children):
            if children[i].name == "dt":
                q_text = _clean_text(children[i].get_text(separator=" "))
                a_parts = []
                j = i + 1
                while j < len(children) and children[j].name == "dd":
                    a_parts.append(_clean_text(children[j].get_text(separator=" ")))
                    j += 1
                a_text = "\n".join(a_parts)
                if len(q_text) >= _MIN_Q_LEN and len(a_text) >= _MIN_A_LEN:
                    pairs.append((q_text, a_text))
                i = j
            else:
                i += 1
    return pairs


def _extract_details_pattern(soup: BeautifulSoup) -> List[Tuple[str, str]]:
    """Ekstrak dari pola HTML5 <details>/<summary>."""
    pairs = []
    for details in soup.find_all("details"):
        summary = details.find("summary")
        if not summary:
            continue
        q_text = _clean_text(summary.get_text(separator=" "))

        # Ambil semua teks di dalam <details> selain <summary>
        answer_parts = []
        for child in details.children:
            if child.name and child.name != "summary":
                text = _clean_text(child.get_text(separator=" "))
                if text:
                    answer_parts.append(text)
        a_text = "\n".join(answer_parts)

        if len(q_text) >= _MIN_Q_LEN and len(a_text) >= _MIN_A_LEN:
            pairs.append((q_text, a_text))
    return pairs


def _extract_faq_container(soup: BeautifulSoup) -> List[Tuple[str, str]]:
    """
    Cari kontainer dengan class/id yang mengindikasikan FAQ (accordion, tanya-jawab, dll.),
    lalu cari pasangan heading/paragraf di dalamnya.
    """
    pairs = []

    # Cari semua elemen yang memiliki class/id mengandung kata FAQ
    candidates = []
    for el in soup.find_all(True):
        cls = " ".join(el.get("class", []))
        el_id = el.get("id", "")
        if _FAQ_CONTAINER_CLASSES.search(cls) or _FAQ_CONTAINER_CLASSES.search(el_id):
            candidates.append(el)

        # De-duplikasi: ambil elemen paling dalam (bukan ancestor kandidat lain)
        filtered = []
    for el in candidates:
        is_ancestor = any(el in other.parents for other in candidates if other is not el)
        if not is_ancestor:
            filtered.append(el)

    for container in filtered:
        # Dalam kontainer, cari item individual
        # Pattern: heading/strong/bold diikuti paragraf
        item_pairs = _find_heading_paragraph_in(container)
        pairs.extend(item_pairs)

    return pairs


def _find_heading_paragraph_in(container) -> List[Tuple[str, str]]:
    """Cari pasangan heading+paragraph dalam container tertentu."""
    pairs = []
    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6", "strong", "b", "p"}
    
    # Ambil semua children langsung
    direct_children = [c for c in container.children if c.name]

    i = 0
    while i < len(direct_children):
        child = direct_children[i]
        child_text = _clean_text(child.get_text(separator=" "))

        # Anggap ini sebagai pertanyaan jika singkat dan merupakan heading/strong
        if child.name in {"h3", "h4", "h5", "h6", "strong", "b"}:
            if len(child_text) >= _MIN_Q_LEN:
                a_parts = []
                j = i + 1
                while j < len(direct_children) and direct_children[j].name not in {"h3", "h4", "h5", "h6", "strong", "b"}:
                    a_text = _clean_text(direct_children[j].get_text(separator=" "))
                    if a_text:
                        a_parts.append(a_text)
                    j += 1
                a_text = "\n".join(a_parts)
                if len(a_text) >= _MIN_A_LEN:
                    pairs.append((child_text, a_text))
                i = j
                continue
        i += 1

    return pairs


def _extract_heading_paragraph(soup: BeautifulSoup) -> List[Tuple[str, str]]:
    """Cari pola H3/H4 diikuti paragraf di area konten."""
    # Cari kontainer paling relevan
    search_root = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile("content|article|post", re.I))
        or soup.find("body")
        or soup
    )

    pairs = []
    all_elements = list(search_root.find_all(["h3", "h4", "p"], recursive=False))
    
    if not all_elements:
        all_elements = list(search_root.find_all(["h3", "h4", "p"]))

    i = 0
    while i < len(all_elements):
        el = all_elements[i]
        if el.name in {"h3", "h4"}:
            q_text = _clean_text(el.get_text(separator=" "))
            if len(q_text) >= _MIN_Q_LEN:
                a_parts = []
                j = i + 1
                while j < len(all_elements) and all_elements[j].name == "p":
                    a_text = _clean_text(all_elements[j].get_text(separator=" "))
                    if a_text:
                        a_parts.append(a_text)
                    j += 1
                a_text = "\n".join(a_parts)
                if len(a_text) >= _MIN_A_LEN:
                    pairs.append((q_text, a_text))
                i = j
                continue
        i += 1

    return pairs
