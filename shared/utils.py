"""
RAG Medan v3 - Shared Utilities
Common utility functions used across services
"""
import re
import ast
import json
from typing import Dict, List, Set, Optional, Any

# Stopwords untuk filtering
STOPWORDS: Set[str] = {
    "apa", "bagaimana", "cara", "untuk", "dan", "atau", "yang", "dengan",
    "ke", "dari", "buat", "membuat", "mengurus", "mendaftar", "mencetak",
    "dimana", "kapan", "berapa", "adalah", "itu", "ini", "saya", "kamu"
}

# Sinonim untuk ekspansi query
SYNONYMS: Dict[str, List[str]] = {
    # Singkatan pemerintah
    "ktp": ["kartu tanda penduduk"],
    "kk": ["kartu keluarga"],
    "kadis": ["kepala dinas"],
    "kominfo": ["dinas komunikasi dan informatika", "diskominfo"],
    "dukcapil": ["dinas kependudukan dan catatan sipil", "disdukcapil"],
    "dishub": ["dinas perhubungan"],
    "dinkes": ["dinas kesehatan"],
    "disnaker": ["dinas ketenagakerjaan"],
    "sktm": ["surat keterangan tidak mampu"],
    "siup": ["surat izin usaha perdagangan"],
    "umkm": ["usaha mikro kecil menengah"],
    "pungli": ["pungutan liar"],
    "bansos": ["bantuan sosial"],
    "damkar": ["pemadam kebakaran"],
    "nib": ["nomor induk berusaha"],
    "nisn": ["nomor induk siswa nasional"],
    "pkl": ["praktek kerja lapangan"],
    "skkni": ["standar kompetensi kerja nasional indonesia"],
    "siduta": ["sistem informasi terpadu ketenagakerjaan"],
    # Sinonim bahasa informal -> formal
    "gimana": ["bagaimana", "caranya"],
    "gimn": ["bagaimana", "caranya"],
    "gmn": ["bagaimana", "caranya"],
    "bikin": ["buat", "membuat", "pembuatan"],
    "bkin": ["buat", "membuat", "pembuatan"],
    "ngurus": ["mengurus", "urus", "pengurusan"],
    "ganti": ["ubah", "mengubah", "perubahan", "mengganti", "pergantian"],
    "rubah": ["ubah", "mengubah", "perubahan"],
    "perpanjang": ["perpanjangan", "memperpanjang"],
    "daftar": ["mendaftar", "pendaftaran", "registrasi"],
    "cetak": ["mencetak", "pencetakan", "print"],
    "ambil": ["mengambil", "pengambilan"],
    "syarat": ["persyaratan", "ketentuan", "dokumen"],
    "berkas": ["dokumen", "file", "persyaratan"],
    "prosedur": ["proses", "langkah", "tahapan", "alur"],
    "biaya": ["tarif", "harga", "bayar", "ongkos"],
    "gratis": ["bebas biaya", "tanpa biaya", "free"],
    "lama": ["durasi", "waktu", "berapa hari"],
    "cepat": ["kilat", "express", "segera"],
    # Sinonim aksi
    "buat": ["membuat", "pembuatan", "bikin"],
    "urus": ["mengurus", "pengurusan", "ngurus"],
    "ubah": ["mengubah", "perubahan", "ganti", "mengganti"],
    "perbaiki": ["memperbaiki", "koreksi", "revisi"],
}

# Kategori dan keyword mapping
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "0196f6a8-9cb8-7385-8383-9d4f8fdcd396": [
        "ktp", "kk", "kartu keluarga", "kartu tanda penduduk",
        "akta", "kelahiran", "kematian", "domisili", "sktm", "nik"
    ],
    "0196ccd1-d7f9-7252-b0a1-a67d4bc103a0": [
        "bpjs", "rsud", "puskesmas", "klinik", "vaksin", "pengobatan",
        "berobat", "posyandu", "stunting", "imunisasi"
    ],
    "0196cd16-3a0a-726d-99b4-2e9c6dda5f64": [
        "sekolah", "ppdb", "spmb", "guru", "siswa", "beasiswa", "prestasi", "zonasi", "afirmasi", "nisn"
    ],
    "019707b1-ebb6-708f-ad4d-bfc65d05f299": [
        "pengaduan", "izin", "siup", "bantuan", "masyarakat", "usaha", "nib",
        "kartu prakerja", "kartu kuning", "ak1", "sertifikat", "pajak", "reklame", "magang", "siduta"
    ],
    "0196f6b9-ba96-70f1-a930-3b89e763170f": [
        "kepala dinas", "kadis", "sekretaris", "jabatan", "struktur organisasi"
    ],
    "01970829-1054-72b2-bb31-16a34edd84fc": [
        "aturan", "peraturan", "perwali", "perda", "perpres", "hukum"
    ],
    "0196f6c0-1178-733a-acd8-b8cb62eefe98": [
        "lokasi", "alamat", "kantor", "posisi"
    ],
    "001970853-dd2e-716e-b90c-c4f79270f700": [
        "tugas", "fungsi", "tupoksi", "profil", "visi", "misi"
    ]
}

CATEGORY_NAMES: Dict[str, str] = {
    "0196f6a8-9cb8-7385-8383-9d4f8fdcd396": "Kependudukan",
    "0196ccd1-d7f9-7252-b0a1-a67d4bc103a0": "Kesehatan",
    "0196cd16-3a0a-726d-99b4-2e9c6dda5f64": "Pendidikan",
    "019707b1-ebb6-708f-ad4d-bfc65d05f299": "Layanan Masyarakat",
    "0196f6b9-ba96-70f1-a930-3b89e763170f": "Struktur Organisasi",
    "01970829-1054-72b2-bb31-16a34edd84fc": "Peraturan",
    "0196f6c0-1178-733a-acd8-b8cb62eefe98": "Lokasi Fasilitas Pemerintahan Kota Medan",
    "001970853-dd2e-716e-b90c-c4f79270f700": "Profil"
}

# Lokasi non-Medan untuk filtering
NON_MEDAN_LOCATIONS: List[str] = [
    "jakarta", "bandung", "surabaya", "yogyakarta", "semarang",
    "siantar", "pematangsiantar", "pematang siantar",
    "binjai", "tebing", "tebing tinggi", "aceh", "padang",
    "pekanbaru", "riau", "deliserdang", "deli serdang",
    "langkat", "tanjung morawa", "belawan", "labuhanbatu"
]

# Opinion words untuk filtering
OPINION_WORDS: List[str] = [
    "rajin", "malas", "ganteng", "cantik", "baik", "buruk",
    "terkenal", "paling", "ter", "terbaik", "terburuk",
    "terjelek", "terbodoh", "terrajin"
]


def detect_category(question: str) -> Optional[Dict[str, str]]:
    """Deteksi kategori berdasarkan keyword dalam pertanyaan."""
    question_lower = question.lower()
    for category_id, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in question_lower for keyword in keywords):
            return {"id": category_id, "name": CATEGORY_NAMES.get(category_id, "Unknown")}
    return None


def normalize_text(text: str) -> str:
    """Normalisasi teks - hapus karakter spesial, multiple spaces."""
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_location_terms(text: str) -> str:
    """Hapus referensi lokasi umum dari teks."""
    text = re.sub(r"\bdi\s+kota\s+medan\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdi\s+medan\b", "", text, flags=re.IGNORECASE)
    return text.strip()


def expand_terms(text: str) -> str:
    """Ekspansi singkatan dengan sinonimnya."""
    words = text.lower().split()
    expanded_words = []
    for word in words:
        expanded_words.append(word)
        if word in SYNONYMS:
            expanded_words.extend(SYNONYMS[word])
    return " ".join(expanded_words)


def tokenize_and_filter(text: str) -> List[str]:
    """Tokenize dan filter stopwords."""
    return [
        word.lower() for word in text.split()
        if word.lower() not in STOPWORDS and len(word) > 2
    ]


def keyword_overlap(question_a: str, question_b: str) -> float:
    """Hitung overlap score antara dua teks."""
    # Normalize: hapus tanda baca sebelum expand
    question_a_clean = re.sub(r"[^\w\s]", " ", question_a)
    question_b_clean = re.sub(r"[^\w\s]", " ", question_b)
    
    question_a_expanded = expand_terms(question_a_clean)
    question_b_expanded = expand_terms(question_b_clean)
    tokens_a = set(tokenize_and_filter(question_a_expanded))
    tokens_b = set(tokenize_and_filter(question_b_expanded))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def hard_filter_local(question: str) -> Dict[str, Any]:
    """Local hard filter untuk cek lokasi dan opini."""
    question_lower = question.lower()
    question_normalized = re.sub(r"[^\w\s]", " ", question_lower)
    question_normalized = re.sub(r"\s+", " ", question_normalized)

    # Check non-Medan locations
    for location in NON_MEDAN_LOCATIONS:
        if re.search(rf"\b{re.escape(location)}\b", question_normalized):
            return {
                "valid": False,
                "reason": f"Pertanyaan menyebut daerah di luar Medan ({location.title()})",
                "clean_question": question
            }

    # Check opinion words
    if any(re.search(rf"\b{re.escape(word)}\b", question_normalized) for word in OPINION_WORDS):
        return {
            "valid": False,
            "reason": "Pertanyaan bersifat opini/personal, bukan layanan publik",
            "clean_question": question
        }

    # Check question length
    if len(question_normalized.split()) <= 1:
        return {
            "valid": False,
            "reason": "Pertanyaan terlalu pendek atau tidak jelas",
            "clean_question": question
        }

    return {
        "valid": True,
        "reason": "Lolos hard filter",
        "clean_question": question
    }


def safe_parse_answer_id(raw_value: Any) -> List[str]:
    """Parse answer_id dengan aman dari berbagai format."""
    if not raw_value:
        return []

    if isinstance(raw_value, list):
        clean_list = []
        for item in raw_value:
            try:
                if isinstance(item, str) and item.startswith('"') and item.endswith('"'):
                    clean_list.append(json.loads(item))
                else:
                    clean_list.append(item)
            except Exception:
                clean_list.append(item)
        return clean_list

    try:
        raw_string = str(raw_value).strip()
        if raw_string.startswith("[") and raw_string.endswith("]"):
            parsed_array = ast.literal_eval(raw_string)
            clean_list = []
            for item in parsed_array:
                try:
                    clean_list.append(
                        json.loads(item) if isinstance(item, str) and item.startswith('"') else item
                    )
                except Exception:
                    clean_list.append(item)
            return clean_list
        return [raw_string]
    except Exception:
        return [str(raw_value)]


def format_for_display(text: str) -> str:
    """Format text untuk display dengan paragraph breaks yang jelas."""
    if not text:
        return ""
    
    paragraphs = re.split(r'\n\n+', text)
    cleaned_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if para:
            para = para.replace('\n', ' ')
            para = re.sub(r' {2,}', ' ', para)
            cleaned_paragraphs.append(para)
    
    return '\n\n'.join(cleaned_paragraphs)


def calculate_final_score(dense_score: float, overlap_score: float = 0.0) -> float:
    """Calculate final score dengan weighted average."""
    return round((0.65 * dense_score) + (0.35 * overlap_score), 3)
