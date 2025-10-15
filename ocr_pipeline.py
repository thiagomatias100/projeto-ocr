# ocr_pipeline.py
# UNIVERSIDADE FEDERAL DO MARANHÃO - UFMA
# Autor: Thiago Matias da Silva
#
# Projeto de OCR para Identificação de Caracteres e Extração de Textos para Leitura de Tela
# Estratégia:
#   1) PRIORIDADE: API Docling-Serve (v1) com OCR ligado.
#   2) Se a API retornar vazio ou praticamente só <!--image-->, aplica TRY-HARD (pdf->imagem->Tesseract).
#   3) (Opcional, só para testes) Fallback LOCAL Docling: só se USE_LOCAL_TEST=True.
#
# Saídas:
#   - salva "saida_api.md" quando a API retornar markdown
#   - salva "<basename>_tryhard.md" quando o TRY-HARD produzir markdown

import os
import base64
import json
import time
import requests
from PyPDF2 import PdfReader

from pdf2image import convert_from_path
import numpy as np
import cv2
import pytesseract

# >>> Ajuste se necessário (Windows):
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------------------------- CONFIGURAÇÕES ------------------------------------
API_URL = "http://200.137.132.64:5005/v1/convert/source"  # v1 (prioritária)
USE_LOCAL_TEST = False  # False = não usa Docling local; True = usa Docling local como fallback de TESTE
OCR_LANG_TESS = "por+eng+spa"  # idiomas Tesseract para o try-hard
API_TIMEOUT = 120  # segundos
# -----------------------------------------------------------------------------


def only_placeholders(md: str, min_real_chars: int = 40) -> bool:
    """
    True se o markdown tem praticamente só placeholders <!--image--> e quase nenhum texto real.
    """
    if not md:
        return True
    stripped = md.replace("<!--image-->", "").strip()
    return len(stripped) < min_real_chars


def is_native_pdf(pdf_path: str, min_chars: int = 50) -> bool:
    """
    True se parece nativo (texto extraível com PyPDF2).
    False se escaneado (ou falha na leitura).
    (Usado apenas para diagnóstico. O fluxo principal não depende dele.)
    """
    try:
        reader = PdfReader(pdf_path)
        extracted = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            extracted.append(t)
        text = "".join(extracted).strip()
        return len(text) >= min_chars
    except Exception as e:
        print(f"[detector] Falha PyPDF2: {e}")
        return False


def extrator_api(pdf_path: str) -> str:
    """
    Usa a API Docling-Serve (v1) com OCR ativado (prioritária).
    Retorna o markdown ou "".
    """
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo '{pdf_path}' não foi encontrado.")
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"Arquivo '{pdf_path}' não é .PDF.")

        with open(pdf_path, "rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "options": {
                "from_formats": ["pdf"],
                "to_formats": ["md"],
                # OCR ligado porque a prioridade é a API resolver tudo:
                "do_ocr": True,
                #"ocr_engine": "Tesseract",
                "ocr_lang": ["pt", "en", "es"],
                "pdf_backend": "dlparse_v4",       # backend recomendado
                "image_export_mode": "placeholder" # mantém placeholders no MD
            },
            "sources": [
                {
                    "base64_string": pdf_base64,
                    "filename": os.path.basename(pdf_path),
                    "kind": "file"
                }
            ],
            "target": {"kind": "inbody"}
        }

        resp = requests.post(API_URL, json=payload, timeout=API_TIMEOUT)
        resp.raise_for_status()

        try:
            data = resp.json()
        except json.JSONDecodeError:
            print("|°_°| API retornou texto não-JSON (trecho abaixo):")
            print(resp.text[:1000])
            return ""

        md = (data.get("document") or {}).get("md_content") or ""
        if md:
            with open("saida_api.md", "w", encoding="utf-8") as f:
                f.write(md)
        return md

    except (FileNotFoundError, ValueError) as e:
        print(e)
        return ""
    except requests.exceptions.RequestException as e:
        print("|°~°| API - erro de requisição:", e)
        return ""
    except Exception as e:
        print("|°~°| API - erro inesperado:", e)
        return ""


# ----------------------- TRY-HARD (pdf2image + Tesseract) ---------------------

def pdf_pages_to_numpy_bgr(pdf_path: str, dpi: int = 300) -> list:
    """
    Converte o PDF em uma lista de frames (np.ndarray BGR) usando pdf2image.
    """
    pages_pil = convert_from_path(pdf_path, dpi=dpi)
    frames_bgr = []
    for pil_img in pages_pil:
        rgb = np.array(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frames_bgr.append(bgr)
    return frames_bgr


def preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """
    Pré-processamento robusto para OCR:
    - escala de cinza
    - CLAHE
    - desruído leve (bilateral)
    - sharpen leve
    - binarização adaptativa
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    den = cv2.bilateralFilter(clahe, d=7, sigmaColor=50, sigmaSpace=50)
    blur = cv2.GaussianBlur(den, (0,0), 1.0)
    sharp = cv2.addWeighted(den, 1.5, blur, -0.5, 0)
    binimg = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        35, 15
    )
    return binimg


def ocr_tesseract_bgr_to_text(bgr: np.ndarray, lang: str = OCR_LANG_TESS) -> str:
    """
    OCR com Tesseract. Usa PSM 6 (parágrafos). Ajuste para PSM 4 em páginas multi-coluna.
    """
    proc = preprocess_for_ocr(bgr)
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(proc, lang=lang, config=config)
    return text.replace("\r", "").strip()


def ocr_try_hard_pdf(pdf_path: str, dpi: int = 300) -> str:
    """
    Fallback 'try-hard': rasteriza todas as páginas e roda OCR por Tesseract,
    retornando um Markdown concatenado (com separadores entre páginas).
    """
    try:
        frames = pdf_pages_to_numpy_bgr(pdf_path, dpi=dpi)
        md_pages = []
        for i, bgr in enumerate(frames, start=1):
            txt = ocr_tesseract_bgr_to_text(bgr)
            md_pages.append(f"## Página {i}\n\n{txt if txt else '*(sem texto detectável)*'}\n")
        md_all = "\n---\n".join(md_pages)
        if md_all.strip():
            outp = f"{os.path.splitext(pdf_path)[0]}_tryhard.md"
            with open(outp, "w", encoding="utf-8") as f:
                f.write(md_all)
        return md_all
    except Exception as e:
        print("|°~°| TRY-HARD OCR falhou:", e)
        return ""


# ------------------------- DOCLING LOCAL (TESTE) ------------------------------

def extrator_local_docling(pdf_path: str) -> str:
    """
    Usa Docling local (com OCR quando necessário).
    ATENÇÃO: apenas para TESTES (controlado por USE_LOCAL_TEST).
    """
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo '{pdf_path}' não foi encontrado.")
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"Arquivo '{pdf_path}' não é .PDF.")

        # Import adiado para não exigir a lib quando não for testar
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        md = result.document.export_to_markdown() or ""
        if md:
            with open(f"{os.path.splitext(pdf_path)[0]}_local.md", "w", encoding="utf-8") as f:
                f.write(md)
        return md

    except (FileNotFoundError, ValueError) as e:
        print(e)
        return ""
    except Exception as e:
        print("|°~°| LOCAL Docling - erro:", e)
        return ""


# ------------------------------ ORQUESTRADOR ----------------------------------

def processar_pdf(pdf_path: str) -> str:
    """
    Fluxo PRIORITÁRIO: API -> (se vazio/placeholder) TRY-HARD -> (TESTE opcional) LOCAL
    """
    if not os.path.exists(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        print("|°~°| Forneça um caminho válido para um .pdf existente.")
        return ""

    print("|°_°| Passo 1: API (prioritária)...")
    md = extrator_api(pdf_path)
    if md and not only_placeholders(md):
        print("|°_°| API OK (texto real detectado).")
        return md
    print("|°~°| API retornou vazio ou só placeholders. Indo para TRY-HARD...")

    print("|°_°| Passo 2: TRY-HARD (pdf2image + Tesseract)...")
    md = ocr_try_hard_pdf(pdf_path, dpi=300)
    if md and not only_placeholders(md):
        print("|°_°| TRY-HARD OK (texto real detectado).")
        return md

    if USE_LOCAL_TEST:
        print("|°_°| Passo 3 (TESTE): Docling LOCAL...")
        md = extrator_local_docling(pdf_path)
        if md and not only_placeholders(md):
            print("|°_°| LOCAL OK (texto real detectado).")
            return md

    print("|°~°| Falhou API e TRY-HARD (e LOCAL desativado ou sem texto).")
    return ""


# --------------------------------- MAIN ---------------------------------------

if __name__ == "__main__":
    inicio_wall = time.time()
    inicio_cpu = time.process_time()

    pdf_path = input("ENTRE COM O NOME DO ARQUIVO: ").strip().strip('"').strip("'")
    _ = processar_pdf(pdf_path)

    fim_wall = time.time()
    fim_cpu = time.process_time()
    print(f"|°_°| Tempo decorrido (wall): {fim_wall - inicio_wall:.2f}s")
    print(f"|°_°| Tempo de CPU: {fim_cpu - inicio_cpu:.2f}s")
