# src/pdf_reader.py
# ============================================================
# Lecture et découpage des PDFs en chunks exploitables
# ============================================================

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
from pathlib import Path
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrait tout le texte d'un PDF page par page.
    Retourne le texte brut complet.
    """
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) non installé. Installez-le avec: pip install PyMuPDF")
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    doc = fitz.open(str(pdf_path))
    full_text = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            full_text.append(f"[PAGE {page_num + 1}]\n{text}")
    doc.close()
    return "\n\n".join(full_text)


def split_into_chunks(text: str, chunk_size: int = 1500, overlap: int = 150) -> List[str]:
    """
    Découpe un texte long en chunks de taille `chunk_size` caractères,
    avec un chevauchement de `overlap` pour préserver le contexte inter-chunks.

    Stratégie :
    - On essaie de couper aux fins de paragraphes (double newline) ou de phrases.
    - Si aucune coupure naturelle, on coupe brutalement.

    Pourquoi splitter ?
    - Les LLMs ont une fenêtre de contexte limitée (ex: 4096 tokens ~ 3000-4000 chars).
    - Découper permet de traiter de gros documents sans dépasser la limite.
    - Le chevauchement évite de perdre des informations à la jointure entre chunks.
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Chercher une coupure naturelle (fin de paragraphe)
        if end < text_len:
            # Priorité 1 : double newline
            split_pos = text.rfind("\n\n", start, end)
            if split_pos == -1 or split_pos <= start:
                # Priorité 2 : fin de phrase
                for sep in [". ", "! ", "? ", ".\n"]:
                    split_pos = text.rfind(sep, start, end)
                    if split_pos > start:
                        split_pos += len(sep)
                        break
            if split_pos > start:
                end = split_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Avancer avec chevauchement
        start = end - overlap if end < text_len else text_len

    return chunks


def get_relevant_chunks(
    chunks: List[str],
    indicator: str,
    top_k: int = 3,
) -> List[str]:
    """
    Sélectionne les `top_k` chunks les plus pertinents pour un indicateur donné.
    Méthode simple : comptage des mots-clés de l'indicateur dans chaque chunk.

    Pour un pipeline production, on remplacerait ceci par une recherche vectorielle
    (FAISS + embeddings), mais pour le test, le comptage de keywords suffit.
    """
    # Construire un set de mots-clés à partir de l'indicateur
    keywords = set(indicator.lower().replace("-", " ").replace("/", " ").split())
    # Retirer les stopwords basiques
    stopwords = {"de", "du", "le", "la", "les", "en", "à", "au", "des", "un", "une", "l"}
    keywords -= stopwords

    scored_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = sum(chunk_lower.count(kw) for kw in keywords)
        scored_chunks.append((score, i, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, _, c in scored_chunks[:top_k]]

    # Si aucun chunk pertinent trouvé, prendre les premiers chunks du doc
    if not any(s > 0 for s, _, _ in scored_chunks[:top_k]):
        top_chunks = chunks[:top_k]

    return top_chunks


def load_document(doc_config: Dict, chunk_size: int = 1500, overlap: int = 150) -> Dict:
    """
    Charge un document PDF, l'extrait et le découpe.
    Retourne un dictionnaire enrichi avec le texte et les chunks.
    """
    pdf_path = doc_config["pdf_path"]
    print(f"  📄 Chargement : {pdf_path}")
    full_text = extract_text_from_pdf(pdf_path)
    chunks = split_into_chunks(full_text, chunk_size=chunk_size, overlap=overlap)
    print(f"     → {len(full_text)} caractères | {len(chunks)} chunks")
    return {
        **doc_config,
        "full_text": full_text,
        "chunks": chunks,
    }
