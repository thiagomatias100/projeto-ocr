import os
import base64
import json
import requests
from PyPDF2 import PdfReader
from docling.document_converter import DocumentConverter

API_URL = "http://200.137.132.64:5001/v1alpha/convert/source"

# ---------- DETECÇÃO ----------
def is_native_pdf(pdf_path: str, min_chars: int = 50) -> bool:
    """
    True se parece nativo (texto extraível com PyPDF2).
    False se escaneado (ou falha na leitura).
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

# ---------- EXTRATORES ----------
def extrator_api_nativo(pdf_path: str) -> str:
    """
    Usa a API (docling-serve) SEM OCR (ideal p/ PDF nativo).
    Retorna o markdown (string) ou "" se não obtiver.
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
                # OCR desativado p/ nativo
                "do_ocr": False,
                "pdf_backend": "dlparse_v4",
                "image_export_mode": "placeholder"
            },
            "file_sources": [
                {"base64_string": pdf_base64, "filename": os.path.basename(pdf_path)}
            ]
        }

        resp = requests.post(API_URL, json=payload, timeout=120)
        resp.raise_for_status()

        try:
            data = resp.json()
        except json.JSONDecodeError:
            print("[API NATIVO] Resposta não-JSON:")
            print(resp.text[:1000])
            return ""

        md = (data.get("document") or {}).get("md_content") or ""
        if md:
            print("[API NATIVO] Markdown extraído com sucesso.\n")
            print(md)
            with open("saida_api.md", "w", encoding="utf-8") as f:
                f.write(md)
        else:
            print("[API NATIVO] Sem md_content na resposta (trecho abaixo).")
            print(str(data)[:1000])
        return md
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return ""
    except requests.exceptions.RequestException as e:
        print("[API NATIVO] Erro de requisição:", e)
        return ""
    except Exception as e:
        print("[API NATIVO] Erro inesperado:", e)
        return ""

def extrator_local_ocr(pdf_path: str) -> str:
    """
    Usa Docling local (com OCR quando necessário).
    Retorna o markdown (string) ou "" se não obtiver.
    """
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo '{pdf_path}' não foi encontrado.")
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"Arquivo '{pdf_path}' não é .PDF.")

        converter = DocumentConverter()  # seu ambiente já registra OCR engines
        result = converter.convert(pdf_path)
        md = result.document.export_to_markdown() or ""
        if md:
            print("[LOCAL OCR] Markdown extraído com sucesso.\n")
            print(md)
            with open("saida_local_ocr.md", "w", encoding="utf-8") as f:
                f.write(md)
        else:
            print("[LOCAL OCR] Conversão retornou vazio.")
        return md
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return ""
    except Exception as e:
        print("[LOCAL OCR] Erro ao processar com Docling:", e)
        return ""

# ---------- PIPELINE / ORQUESTRADOR ----------
def processar_pdf(pdf_path: str, min_chars: int = 50):
    """
    Regra: NATIVO -> API (leve); ESCANEADO -> LOCAL (OCR).
    Com fallback cruzado se vier vazio.
    """
    if not os.path.exists(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        print("Forneça um caminho válido para um .pdf existente.")
        return

    print("[*] Detectando tipo do PDF...")
    nativo = is_native_pdf(pdf_path, min_chars=min_chars)

    if nativo:
        print("[*] Detectado NATIVO -> usando API (sem OCR).")
        md = extrator_api_nativo(pdf_path)
        if not md:  # fallback
            print("[*] Fallback: tentando LOCAL (OCR)...")
            extrator_local_ocr(pdf_path)
    else:
        print("[*] Detectado ESCANEADO -> usando LOCAL (com OCR).")
        md = extrator_local_ocr(pdf_path)
        if not md:  # fallback
            print("[*] Fallback: tentando API (sem OCR)...")
            extrator_api_nativo(pdf_path)

if __name__ == "__main__":
    pdf_path = input("ENTRE COM O NOME DO ARQUIVO: ").strip().strip('"').strip("'")
    processar_pdf(pdf_path, min_chars=50)
