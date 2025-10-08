# ocr_pipeline_api_first.py
# ------------------------------------------------------------
# UFMA • Projeto OCR Acessível (API primeiro para escaneado)
# ------------------------------------------------------------
import os, io, json, base64, time, argparse, logging, re
import requests
from PyPDF2 import PdfReader
from docling.document_converter import DocumentConverter

API_URL = "http://200.137.132.64:5001/v1alpha/convert/source"
OUTDIR  = "output"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------- util -------------------- #
def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)

def read_file_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def postprocess_markdown(md: str) -> str:
    if not md:
        return md
    subs = {"€":"É","Ã¡":"á","Ã£":"ã","Ã©":"é","Ãª":"ê","Ã³":"ó","¢":"ç"}
    for a,b in subs.items():
        md = md.replace(a,b)
    md = re.sub(r'-\s*\n\s*','', md)                         # remove hífen de quebra
    md = re.sub(r'(?<![.!?:;])\n(?=\w)',' ', md)             # junta linhas suaves
    md = re.sub(r'<!--\s*image\s*-->', '[imagem: selo/assinatura/carimbo]', md, flags=re.I)
    return md.strip()

def write_text(path: str, text: str):
    ensure_outdir()
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ---------------- detecção nativo ---------------- #
def is_native_pdf(pdf_path: str, min_chars: int = 80) -> bool:
    """True se o texto é extraível sem OCR (PDF nativo)."""
    try:
        reader = PdfReader(pdf_path)
        buff = []
        for pg in reader.pages:
            try:
                t = pg.extract_text() or ""
            except Exception:
                t = ""
            buff.append(t)
        return len("".join(buff).strip()) >= min_chars
    except Exception:
        return False

# ---------------- Docling API ---------------- #
def _call_docling_api(pdf_path: str, *, do_ocr: bool, timeout=180) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("Extensão inválida, esperava .pdf")

    pdf_b64 = read_file_base64(pdf_path)
    payload = {
        "options": {
            "from_formats": ["pdf"],
            "to_formats":   ["md"],
            "do_ocr": do_ocr,                 # <- chave da mudança
            "pdf_backend": "dlparse_v4",
            "image_export_mode": "placeholder"
        },
        "file_sources": [
            {"base64_string": pdf_b64, "filename": os.path.basename(pdf_path)}
        ]
    }
    r = requests.post(API_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    md = (data.get("document") or {}).get("md_content") or ""
    if not md and "documents" in data:
        docs = data["documents"] or []
        if docs and isinstance(docs[0], dict):
            md = docs[0].get("md_content") or ""
    return md or ""

def extrator_api_nativo(pdf_path: str) -> str:
    return _call_docling_api(pdf_path, do_ocr=False)

def extrator_api_ocr(pdf_path: str) -> str:
    return _call_docling_api(pdf_path, do_ocr=True)

# ---------------- Docling local (com OCR) ---------------- #
def extrator_local_ocr(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("Extensão inválida, esperava .pdf")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown() or ""

# ---------------- orquestração ---------------- #
def processar_pdf(pdf_path: str, min_chars: int = 80) -> str:
    if not os.path.exists(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        raise FileNotFoundError("Forneça um caminho válido para um .pdf.")

    nativo = is_native_pdf(pdf_path, min_chars=min_chars)
    base = os.path.join(OUTDIR, os.path.splitext(os.path.basename(pdf_path))[0])
    md = ""

    try:
        if nativo:
            logging.info("→ Detectado **NATIVO**. 1) API sem OCR → 2) Local+OCR (fallback).")
            md = extrator_api_nativo(pdf_path)
            if not md:
                logging.warning("API (sem OCR) vazia. Fallback: Local+OCR…")
                md = extrator_local_ocr(pdf_path)
        else:
            logging.info("→ Detectado **ESCANEADO**. 1) API COM OCR → 2) Local+OCR → 3) API sem OCR.")
            md = extrator_api_ocr(pdf_path)
            if not md:
                logging.warning("API (com OCR) vazia. Fallback: Local+OCR…")
                md = extrator_local_ocr(pdf_path)
            if not md:
                logging.warning("Local+OCR vazio. Último recurso: API sem OCR…")
                md = extrator_api_nativo(pdf_path)
    except requests.exceptions.RequestException as e:
        logging.error(f"Falha na API: {e}")
        # Mantém a mesma preferência dependendo do tipo:
        if nativo:
            md = extrator_local_ocr(pdf_path)
        else:
            md = extrator_local_ocr(pdf_path) or extrator_api_nativo(pdf_path)

    md = postprocess_markdown(md)
    if md:
        out_md = f"{base}.md"
        write_text(out_md, md)
        logging.info(f"Markdown salvo: {out_md}")
    else:
        logging.error("Conversão retornou vazio.")
    return md

# ---------------- CLI ---------------- #
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Pipeline Docling + OCR (API primeiro para escaneado)")
    p.add_argument("pdf", help="Arquivo PDF")
    p.add_argument("--min-chars", type=int, default=80, help="Limiar para detectar PDF nativo")
    args = p.parse_args()
    processar_pdf(args.pdf, min_chars=args.min_chars)
