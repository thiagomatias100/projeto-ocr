# ocr_pipeline_v2.py
# UNIVERSIDADE FEDERAL DO MARANHÃO - UFMA
# Autor: Thiago Matias da Silva
#
# Estratégia solicitada:
#  1) Entrada pode ser PDF ou IMAGEM.
#  2) Se for PDF:
#       - Tenta API. Se vier só <!--image--> => PDF escaneado.
#       - Converte PDF -> PNG (em memória), EXIBE (matplotlib) sem salvar,
#         deixa espaço comentado para pré-processamento futuro,
#         e tenta de novo na API utilizando a imagem PNG em memória.
#         Se ainda falhar, fallback com OpenCV + Tesseract e salva em .md.
#  3) Se for IMAGEM:
#       - EXIBE (matplotlib), tenta API com a imagem.
#         Se falhar, fallback com OpenCV + Tesseract e salva .md.
#
# Saídas:
#  - "saida_api.md" quando a API retornar markdown válido.
#  - "<basename>_ocr.md" quando o fallback local (Tesseract) gerar texto.
#
# Requisitos:
#  - Tesseract instalado (e caminho configurado no Windows).
#  - pdf2image + Poppler para PDF -> imagem.
#  - requests, PyPDF2, OpenCV, matplotlib, numpy.
#  - (Opcional) easyocr no servidor se usar o 'ocr_engine': 'easyocr' na API.

import os
import io
import base64
import json
import time
import imghdr
import requests
from typing import Tuple, List
import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pyttsx3

# --- Acessibilização mínima de Markdown ---
import re
from pathlib import Path

TEXTO_IMAGEM_ALT = "[Descrição: aqui havia uma imagem ou logotipo]"

def acessibilizar_md(md: str,
                     texto_imagem: str = TEXTO_IMAGEM_ALT,
                     substituir_imgs_markdown: bool = False) -> str:
    """
    Deixa o MD mais acessível:
      (1) Marca títulos '# ...' com nível.
      (2) Substitui <!--image--> por texto alternativo.
      (3) (opcional) Substitui '![alt](src)' por texto alternativo.
    """
    def _marca_titulo(m):
        hashes = m.group(1)
        titulo = m.group(2).strip()
        nivel = len(hashes)
        return f"\n[Início do título nível {nivel}: {titulo}]\n"

    md2 = re.sub(r'^(#{1,6})\s*(.+)$', _marca_titulo, md, flags=re.MULTILINE)
    md2 = re.sub(r'<!--\s*image\s*-->', texto_imagem, md2, flags=re.IGNORECASE)

    if substituir_imgs_markdown:
        md2 = re.sub(r'!\[[^\]]*\]\([^)]+\)', texto_imagem, md2)

    return md2


# --------------------------- CONFIG -------------------------------------------

# Windows: aponte o executável do Tesseract se necessário
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- TTS opcional (narração) ---
ENABLE_TTS = True  # defina False se quiser silenciar rápido

def speak(msg: str):
    """Fala uma mensagem curta. Ignora erros se pyttsx3 não estiver instalado."""
    if not ENABLE_TTS:
        return
    try:
        import pyttsx3
        e = pyttsx3.init()
        for v in e.getProperty("voices"):
            name = v.name.lower()
            if "portugu" in name or "brazil" in name or name.startswith("pt"):
                e.setProperty("voice", v.id)
                break
        e.say(msg)
        e.runAndWait()
    except Exception:
        pass  # não quebra o pipeline se algo der errado no TTS


API_ENDPOINTS = [
    "http://200.137.132.64:5005/v1/convert/source",
   #"http://200.137.132.64:5001/v1alpha/convert/source",
]
API_TIMEOUT = 120

# Para Tesseract local (fallback)
TESS_LANG_STR = "por"#+eng+spa"  # idiomas do Tesseract (string única)
TESS_CONFIG = "--oem 3 --psm 6"  # troque p/ --psm 4 se multi-coluna

# -----------------------------------------------------------------------------


def only_placeholders(md: str, min_real_chars: int = 40) -> bool:
    """
    True se o markdown tem praticamente só placeholders (<!--image--> e <!-- page-break -->)
    e quase nenhum texto real.
    """
    if not md:
        return True
    stripped = (
        md.replace("<!--image-->", "")
          .replace("<!-- page-break -->", "")
          .strip()
    )
    return len(stripped) < min_real_chars


def is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def is_image_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def np_bgr_to_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Falha ao codificar PNG em memória.")
    return buf.tobytes()


def pil_to_bgr(pil_img) -> np.ndarray:
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def show_image_bgr(bgr: np.ndarray, title: str = "Visualização"):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def build_payload_for_pdf(pdf_bytes_b64: str) -> dict:
    """
    Payload já compatível com o servidor 'novo'.
    """
    return {
        "options": {
            "from_formats": ["pdf", "image"],
            "to_formats": ["md"],
            "image_export_mode": "placeholder",
            "do_ocr": True,
            "force_ocr": False,
            # Ajuste conforme o servidor:
            "ocr_engine": "easyocr",          # se o servidor usar Tesseract, troque para "tesseract"
            "ocr_lang": ["pt","en","es"],   # EasyOCR usa 'pt'; Tesseract seria "por+eng+spa"
            "pdf_backend": "pypdfium2",
            "table_mode": "fast",
            "table_cell_matching": True,
            "pipeline": "standard",
            "do_table_structure": True,
            "include_images": True,
            "images_scale": 2,
            "md_page_break_placeholder": "<!-- page-break -->",
            "do_code_enrichment": False,
            "do_formula_enrichment": False,
            "do_picture_classification": False,
            "do_picture_description": False,
        },
        "sources": [{
            "base64_string": pdf_bytes_b64,
            "filename": "input.pdf",
            "kind": "file"
        }],
        "target": {"kind": "inbody"}
    }


def build_payload_for_image(img_png_bytes_b64: str, filename: str = "page.png") -> dict:
    """
    Payload para reenviar a IMAGEM (PNG em memória) na API.
    """
    return {
        "options": {
            "from_formats": ["image"],
            "to_formats": ["md"],
            "image_export_mode": "placeholder",
            "do_ocr": True,
            "force_ocr": False,
            "ocr_engine": "easyocr",
            "ocr_lang": ["pt", "en", "es"],
            "pdf_backend": "pypdfium2",
            "table_mode": "fast",
            "table_cell_matching": True,
            "pipeline": "standard",
            "do_table_structure": True,
            "include_images": True,
            "images_scale": 2,
            "md_page_break_placeholder": "<!-- page-break -->",
            "do_code_enrichment": False,
            "do_formula_enrichment": False,
            "do_picture_classification": False,
            "do_picture_description": False,
        },
        "sources": [{
            "base64_string": img_png_bytes_b64,
            "filename": filename,
            "kind": "file"
        }],
        "target": {"kind": "inbody"}
    }


def call_docling_api(payload: dict) -> str:
    """
    Tenta nos endpoints configurados. Retorna MD ou "".
    """
    last_err = None
    for url in API_ENDPOINTS:
        try:
            resp = requests.post(url, json=payload, timeout=API_TIMEOUT)
            if resp.status_code == 422:
                print(f"|°~°| 422 em {url}: {resp.text[:400]}")
                continue
            resp.raise_for_status()
            try:
                data = resp.json()
            except json.JSONDecodeError:
                print("|°~°| Body não-JSON; amostra:", resp.text[:400])
                continue

            md = ""
            if isinstance(data, dict):
                if "document" in data:
                    md = (data["document"] or {}).get("md_content") or ""
                if not md and "documents" in data and isinstance(data["documents"], list) and data["documents"]:
                    md = (data["documents"][0] or {}).get("md_content") or ""
            if md:
                return md
        except requests.exceptions.RequestException as e:
            last_err = e
            print(f"|°~°| API falhou em {url}: {e}")

    if last_err:
        print("|°~°| API - última exceção:", last_err)
    return ""


# ------------------------- Fallback local (OpenCV + Tesseract) ----------------

def preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """
    Pré-processamento (padrão robusto). Você pode afinar depois.
    """
    imagem = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sharp = cv2.addWeighted(bgr, 1.6, imagem, -0.6, 0)
    #img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    #img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    #img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    """""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    den = cv2.bilateralFilter(clahe, d=5, sigmaColor=40, sigmaSpace=40)
    blur = cv2.GaussianBlur(den, (0, 0), 1.0)
    sharp = cv2.addWeighted(den, 1.5, blur, -0.5, 0)
    binimg = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
    )
    """""
    return sharp#bgr#img


def ocr_tesseract_image(bgr: np.ndarray) -> str:
    proc = preprocess_for_ocr(bgr)
    txt = pytesseract.image_to_string(proc, lang=TESS_LANG_STR, config=TESS_CONFIG)
    return (txt or "").replace("\r", "").strip()


# ------------------------------- Fluxos ---------------------------------------

def process_pdf(path_pdf: str) -> str:
    # 1) Tenta API com o PDF direto
    with open(path_pdf, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

    print("|°_°| Passo 1: API (PDF)...")
    md = call_docling_api(build_payload_for_pdf(pdf_b64))

    if md and not only_placeholders(md):
        print("|°_°| API OK (texto real). Salvando saida_api.md ...")
        speak("Salvando o arquivo adaptado para leitura de tela.")
        #MARKDOWN PURO
        """with open("saida_api.md", "w", encoding="utf-8") as w:
            w.write(md)"""
        #MARKDOWN_ACC
        md_acc = acessibilizar_md(md)
        with open("saida_api.md", "w", encoding="utf-8") as w:
            w.write(md_acc)
        return md

    print("|°~°| API sem texto (ou só placeholders). Considerando ESCANEADO.")
    print("|°_°| Passo 2: PDF -> PNG (memória) + exibição (matplotlib) e reenvio na API...")

    # 2) Converte páginas em memória e tenta API com PNGs
    pages_pil = convert_from_path(path_pdf, dpi=300)
    md_pages = []

    for i, pil_img in enumerate(pages_pil, start=1):
        bgr = pil_to_bgr(pil_img)

        # EXIBIÇÃO (sem salvar)
        show_image_bgr(bgr, title=f"Página {i} (pré-exibição)")

        # --- ESPAÇO PARA PRÉ-PROCESSAMENTO FUTURO ---
        # Exemplo (descomentando você ativa):
        # bgr = some_future_preprocess(bgr)

        # Envia a imagem (PNG) para API
        png_bytes = np_bgr_to_png_bytes(bgr)
        img_b64 = base64.b64encode(png_bytes).decode("utf-8")
        md_img = call_docling_api(build_payload_for_image(img_b64, filename=f"page_{i}.png"))

        if md_img and not only_placeholders(md_img):
            md_pages.append(f"<!-- page-break -->\n{md_img}")
        else:
            # Fallback local com Tesseract
            print(f"|°~°| Página {i}: API com imagem falhou/sem texto. Fallback Tesseract local...")
            speak("Não foi possível extrair texto deste arquivo usando a API.")
            txt = ocr_tesseract_image(bgr)
            md_pages.append(f"## Página {i}\n\n{txt if txt else '*(sem texto detectável)*'}\n")

    md_all = "\n---\n".join(md_pages).strip()
    if md_all:
        outp = f"{os.path.splitext(path_pdf)[0]}_ocr.md"
        with open(outp, "w", encoding="utf-8") as w:
            w.write(md_all)
        print(f"|°_°| OCR concluído. Markdown salvo em: {outp}")
        speak(f"OCR concluído. Markdown salvo em: {outp}")
    else:
        print("|°~°| Não foi possível extrair texto.")
        speak("Não foi possível extrair texto.")
    return md_all


def process_image(path_img: str) -> str:
    # Valida formato de imagem (rápido)
    if not imghdr.what(path_img):
        print("|°~°| Arquivo não parece ser uma imagem válida.")
        speak("Arquivo não parece ser uma imagem válida.")
        return ""

    # Carrega com OpenCV (BGR)
    bgr = cv2.imread(path_img, cv2.IMREAD_COLOR)
    if bgr is None:
        print("|°~°| Falha ao carregar a imagem.")
        speak("Falha ao carregar a imagem.")
        return ""

    # EXIBE (sem salvar)
    show_image_bgr(bgr, title=os.path.basename(path_img))

    # --- ESPAÇO PARA PRÉ-PROCESSAMENTO FUTURO ---
    # bgr = some_future_preprocess(bgr)

    # Tenta API com a imagem
    png_bytes = np_bgr_to_png_bytes(bgr)
    img_b64 = base64.b64encode(png_bytes).decode("utf-8")
    print("|°_°| API (IMAGEM)...")
    speak("Usarei a API de imagem")
    md = call_docling_api(build_payload_for_image(img_b64, filename=os.path.basename(path_img)))

    if md and not only_placeholders(md):
        #MARKDOWN_PURO    
        """"
        with open("saida_api.md", "w", encoding="utf-8") as w:
            w.write(md)
            print("|°_°| API OK. Markdown salvo em saida_api.md")
            speak("Markdown salvo")
            return md
        """
        #MARKDOWN_ACC    
        md_acc = acessibilizar_md(md)
        with open("saida_api.md", "w", encoding="utf-8") as w:
              w.write(md_acc)
              print("|°_°| API OK. Markdown salvo em saida_api.md (versão acessível)")
              speak("Markdown salvo e adaptado para leitura de tela.")
        return md_acc

    # Fallback Tesseract local
    print("|°~°| API sem texto. Fallback Tesseract local...")
    speak("Muito Extranho, não encontrei texto. Tentarei de outro modo!")
    txt = ocr_tesseract_image(bgr)
    md_local = f"## {os.path.basename(path_img)}\n\n{txt if txt else '*(sem texto detectável)*'}\n"
    outp = f"{os.path.splitext(path_img)[0]}_ocr.md"
    with open(outp, "w", encoding="utf-8") as w:
        w.write(md_local)
        print(f"|°_°| Markdown salvo em: {outp}")
        speak(f"Seu arquivo está pronto para leitura! Está salvo em:{outp}")
    return md_local


def main():
    inicio_wall = time.time()
    inicio_cpu = time.process_time()

    path = input("ENTRE COM O ARQUIVO (PDF ou IMAGEM): ").strip().strip('"').strip("'")

    if not os.path.exists(path):
        print("|°~°| Caminho inválido.")
        speak("Esse caminho é inválido. Tente outro!")
        
    elif is_pdf(path):
        _ = process_pdf(path)
    elif is_image_path(path):
        _ = process_image(path)
    else:
        # Tenta inferir pelo conteúdo se extensão não ajuda
        kind = imghdr.what(path)
        if kind:
            _ = process_image(path)
        else:
            print("|°~°| Extensão não reconhecida. Use .pdf ou uma imagem (.png/.jpg/.tif...).")
            speak("Deixe-me ver! A extensão do arquivo não reconhecida. Use .pdf ou uma imagem .png")

    fim_wall = time.time()
    fim_cpu = time.process_time()
    print(f"|°_°| Tempo decorrido (wall): {fim_wall - inicio_wall:.2f}s")
    print(f"|°_°| Tempo de CPU: {fim_cpu - inicio_cpu:.2f}s")


if __name__ == "__main__":
    main()
