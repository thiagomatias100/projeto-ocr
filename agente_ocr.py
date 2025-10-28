# agente_ocr.py
# UNIVERSIDADE FEDERAL DO MARANHÃO - UFMA
# Autor: Thiago Matias da Silva
#
# 
#  1) Entrada pode ser PDF ou IMAGEM.
#  2) Se for PDF:
#       - Tenta API. Se vier só <!--image--> => PDF escaneado.
#       - Converte PDF -> PNG (em memória), EXIBE (matplotlib) sem salvar,
#         Pré-processamento futuro,
#         e tenta de novo na API utilizando a imagem PNG em memória.
#         Se ainda falhar, fallback com OpenCV + Tesseract, fazer adapitação para leitura de tela e salva em .md de Acessibilidade.
#  3) Se for IMAGEM:
#       - EXIBE (matplotlib), tenta API com a imagem.
#         Se falhar, fallback com OpenCV + Tesseract fazer adapitação para leitura de tela e salva em .md de Acessibilidade.
#
# Saídas:
#  - "saida_api.md" quando a API retornar markdown válido.(Mudança em caso de ACC)
#  - "<basename>_ocr.md" quando o fallback local (Tesseract) gerar texto.
#
# Requisitos:
#  - Tesseract instalado (e caminho configurado no Windows).
#  - pdf2image + Poppler para PDF -> imagem.
#  - requests, PyPDF2, OpenCV, matplotlib, numpy, PILLOW.
#  - (Opcional) easyocr no servidor se usar o 'ocr_engine': 'easyocr' na API.

import os
import io
import base64
import json
import time
import imghdr #substituir
import requests
from typing import Tuple, List
import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pyttsx3
from PIL import Image
from typing import Tuple


# --- Acessibilização mínima de Markdown ---
import re
from pathlib import Path

#|°¿°| CONTROLADOR DE MENSAGENS DE ACESSIBILIDADE
#TEXTO_IMAGEM_ALT = "[Descrição: aqui havia uma imagem ou logotipo]" - para subistituir a mensagem de retrono de acessibilidade em:(imagens,logotipos etc.)
TEXTO_IMAGEM_ALT = "[Descrição: aqui havia uma imagem ou logotipo]"
TEXTO_PAGE_BREACK = "[Descrição: próxima página]"
#|°¿°| CONTROLADOR DE DOCUMENTO LOCAL - Para ativação da modalidade de verificação de melhorias de extração com pré processamento de imagem.
#essa modalidade tem finalidade de teste locais, logo, a engime do tesseract deve esta sendo apontada em: PC local e instalada.
OCR_MODE = "documento"
#OCR_MODE = "tabela"
#OCR_MODE = "multicoluna"
#OCR_MODE = "baixo_contarste"



#|°¿°| MÉTODO DE COM FUNÇÃO DE ACESSIBILIDADE PARA PÓS PROCESSAMENTO DO ARQUIVO MARKDOWN (.md).
# OBS: Em caso de usá-lo, deverá trocar md por MD_acc   
def acessibilizar_md(md: str,
    texto_imagem: str = TEXTO_IMAGEM_ALT,
    texto_page_break: str = TEXTO_PAGE_BREACK,
    substituir_imgs_markdown: bool = False) -> str:
    """
    Deixa o .md mais acessível: legendas.
    (1) Marca títulos '# ...' com nível.
    (2) Substitui <!--image--> por texto alternativo.
    (3) (opcional) Substitui '![alt](src)' por texto alternativo.
    (4) Suubtitui <!--page_break--> por texto alternativo.
    (A) Correção preventiva: se entrou “€” por acidente, converta de volta.        
    """
    #CORREÇÃO CARACTERES ESPECIAIS.
    md = md.replace("€", "e")
    #METODO DE TRATAMENTO DE MARCADORES DE TITULOS [#,##,###,...,######]
    def _marca_titulo(m):
        hashes = m.group(1)
        titulo = m.group(2).strip()
        nivel = len(hashes)
        return f"\n[Início do título nível {nivel}: {titulo}]\n"
    #NATIVO DO .md
    md2 = re.sub(r'^(#{1,6})\s*(.+)$', _marca_titulo, md, flags=re.MULTILINE)
    md2 = re.sub(r'<!--\s*image\s*-->', texto_imagem, md2, flags=re.IGNORECASE)
    md2 = re.sub(r'<!--\s*page-break\s*-->',texto_page_break,md2,flags=re.IGNORECASE)

    if substituir_imgs_markdown:
        md2 = re.sub(r'!\[[^\]]*\]\([^)]+\)', texto_imagem, md2)
    #RETORNA O NOVO TEXTO ADAPTADO PARA LEITURA DE TELA.
    return md2

#|°¿°| CONFIGURAÇÃO DO MOTOR TESSERACT NO PC (PARA TESTES) 

#|°¿°| Windows: aponte o executável do Tesseract se necessário.
#|°¿°|MECANISMO USADO PARA MEUS TESTES 
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- TTS opcional (narração) --- USO PROVISÓRIO  DE NARRAÇÃO 

ENABLE_TTS = False  # defina False se quiser silenciar rápido

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

"""
|°¿°| CONFIGURAÇÕES  DAS POISSÍVEIS API'S
"""
API_ENDPOINTS = [
    "http://200.137.132.64:5005/v1/convert/source",
   #"http://200.137.132.64:5001/v1alpha/convert/source",
]
API_TIMEOUT = 120
#PARA USO EXCLUSIVO DE TESTE DE QUALIDADE DAS IMAGENS
#CONTROLADOR DE VISUALIZAÇÃO DE IMAGEM PRÉ-PROCESSSAMENTO USE COMO (True OU False) PARA HABILITAR OU DESABILITAR:
#METODO DE PRE-VISUALIZA
#METODO DE SALVAR AS IMAGENS PRÉ-PROCESSADAS
SHOW_PREVIEW = True     # mostra janelas matplotlib
SAVE_PREVIEW = True     # salva arquivos no disco
PREVIEW_MAX_WIDTH = 1800  # redimensiona para não abrir imagens gigantes


"""
PREPARANDO CONFIGURAÇÕES DE LAYOUT COMO (OSD) DEIXEI ATIVA SOMENTE PARA TESTES LOCAIS!
"""
# Para Tesseract local (fallback)
TESS_LANG_STR = "por"  # idiomas do Tesseract (string única)
TESS_CONFIG = "--oem 3 --psm 3"  # troque p/ --psm 4 se multi-coluna
"""
Tester o modos de configuração de página. OBS: geralmente encontro o melhor resultado para documetos em --psm -
Page segmentation modes:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
       bypassing hacks that are Tesseract-specific.
"""

# -----------------------------------------------------------------------------
"""
MÉTODOS AUXILIARES PARA TRATAR POSSÍVEIS PROBLEMAS:
"""
#MODO DE VERIFICAR ESPAÇOS EM BRANCOS E COMENTÁRIOS 
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

#MÉTODO DE VERIFICAÇÃO DE TIPO DE ARQUIVO (.pdf). RETORNA (True OU False)
def is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


#MÉTODO DE VERIFICAÇÃO DE TIPO DE ARQUIVO (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"). RETORNA (True OU False)
def is_image_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

#
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

#PRÉVIEW DE IMAGENS PRÉ-PROCESSAMENTO PARA TESTE:
def preview_preprocess(bgr: np.ndarray, binimg: np.ndarray, enhanced_bgr: np.ndarray,
                       title: str, out_prefix: str = None):
    """
    Mostra e (opcionalmente) salva o resultado do pré-processamento:
    - Coluna 1: original (RGB)
    - Coluna 2: enhanced (RGB)
    - Coluna 3: binário (grayscale)
    """
    import matplotlib.pyplot as plt
    import cv2
    import os

    # Redimensiona preview para não ficar gigante
    def _resize_max(img, maxw=PREVIEW_MAX_WIDTH):
        h, w = img.shape[:2]
        if w <= maxw:
            return img
        scale = maxw / float(w)
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    rgb_orig    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_enh     = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    bin_disp    = binimg if binimg.ndim == 2 else cv2.cvtColor(binimg, cv2.COLOR_BGR2GRAY)

    rgb_orig = _resize_max(rgb_orig)
    rgb_enh  = _resize_max(rgb_enh)
    bin_disp = _resize_max(bin_disp)

    if SHOW_PREVIEW:
        plt.figure(figsize=(14, 5))
        plt.suptitle(title, fontsize=12)
        plt.subplot(1,3,1); plt.imshow(rgb_orig); plt.title("Original"); plt.axis("off")
        plt.subplot(1,3,2); plt.imshow(rgb_enh);  plt.title("Enhanced"); plt.axis("off")
        plt.subplot(1,3,3); plt.imshow(bin_disp, cmap="gray"); plt.title("Binário p/ OCR"); plt.axis("off")
        plt.tight_layout(); plt.show()

    if SAVE_PREVIEW and out_prefix:
        # Garantir pasta existente
        out_dir = os.path.dirname(out_prefix)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # Salva PNGs (binário e enhanced); original você já tem como arquivo de entrada
        out_bin = f"{out_prefix}_pre_bin.png"
        out_enh = f"{out_prefix}_pre_enh.png"

        # salvar em BGR/GRAY corretos
        cv2.imwrite(out_bin, binimg if binimg.ndim == 2 else cv2.cvtColor(binimg, cv2.COLOR_BGR2GRAY))
        cv2.imwrite(out_enh, cv2.cvtColor(rgb_enh, cv2.COLOR_RGB2BGR))  # volta p/ BGR para salvar
        print(f"|°_°| Pré-processamento salvo: {out_bin} | {out_enh}")


#FIM-PRÉVIEW DE IMAGENS PRÉ-PROCESSAMENTO PARA TESTE:


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
            "ocr_engine": "tesseract",
            "ocr_lang": ["por+eng+spa"],
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


#PROCESSAMNETO LOCAL - Fallback local (OpenCV + Tesseract) - EM CASO DE ERRO DA API OU PARA TESTES LOCAL.

def preprocess_for_ocr(bgr: np.ndarray, mode: str = "documento") -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna (binario_para_tesseract, color_enhanced) de acordo com a modalidade.
    Modos: "documento", "multicoluna", "tabela", "baixo_contraste"
    (100% local. MODO SIMPLES)
    """
    import cv2, numpy as np

    def _resize(img, target_short=1400):
        h, w = img.shape[:2]
        s = min(h, w)
        if s >= target_short: return img
        f = target_short / float(s)
        return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)

    def _illum(gray, ksize=31):
        bg = cv2.medianBlur(gray, ksize)
        bg = np.clip(bg, 1, 255)
        return cv2.divide(gray, bg, scale=255)

    def _unsharp(img, sigma=1.0, amount=1.6):
        blur = cv2.GaussianBlur(img, (0,0), sigma)
        return cv2.addWeighted(img, amount, blur, -(amount-1), 0)

    def _sauvola(gray, win=25, k=0.34, R=128.0):
        g = gray.astype(np.float32)
        mean = cv2.boxFilter(g, -1, (win,win), normalize=True)
        sqm  = cv2.boxFilter(g*g, -1, (win,win), normalize=True)
        var  = np.clip(sqm - mean*mean, 0, None)
        std  = np.sqrt(var)
        thr  = mean * (1 + k*((std/R) - 1))
        return (g > thr).astype(np.uint8)*255

    # ——— roteamento de modos (todos simples e estáveis) ———
    mode = (mode or "documento").lower()
    bgr = _resize(bgr, 1500 if mode=="documento" else 1600)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if mode == "documento":
        norm  = _illum(gray)
        den   = cv2.bilateralFilter(norm, 7, 55, 55)
        sharp = _unsharp(den, 1.2, 1.6)
        binim = _sauvola(sharp, 25, 0.34)
        binim = cv2.morphologyEx(binim, cv2.MORPH_OPEN,  np.ones((2,2), np.uint8), 1)
        binim = cv2.morphologyEx(binim, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), 1)
        enhanced = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    elif mode == "multicoluna":
        norm  = _illum(gray)
        clahe = cv2.createCLAHE(2.0, (8,8)).apply(norm)
        sharp = _unsharp(clahe, 1.0, 1.5)
        binim = _sauvola(sharp, 31, 0.30)
        binim = cv2.morphologyEx(binim, cv2.MORPH_CLOSE, np.ones((1,3), np.uint8), 1)
        enhanced = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    elif mode == "tabela":
        norm     = _illum(gray)
        _, binim = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        H, W     = binim.shape
        hor = cv2.morphologyEx(binim, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(max(20,W//40),1)),1)
        ver = cv2.morphologyEx(binim, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(20,H//40))),1)
        binim = cv2.bitwise_or(binim, cv2.bitwise_or(hor, ver))
        enhanced = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    elif mode == "baixo_contraste":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L,a,b = cv2.split(lab)
        L = cv2.createCLAHE(2.5,(8,8)).apply(L)
        lab = cv2.merge([L,a,b])
        bgr2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        gray2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
        sharp = _unsharp(gray2, 1.0, 1.6)
        binim = _sauvola(sharp, 25, 0.30)
        enhanced = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    else:
        norm  = _illum(gray)
        den   = cv2.bilateralFilter(norm, 7, 55, 55)
        sharp = _unsharp(den, 1.2, 1.6)
        binim = _sauvola(sharp, 25, 0.34)
        enhanced = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    return np.ascontiguousarray(binim), enhanced


def ocr_tesseract_image(bgr: np.ndarray, mode: str = "documento") -> str:
    binimg, _ = preprocess_for_ocr(bgr, mode=mode)
    txt = pytesseract.image_to_string(binimg, lang=TESS_LANG_STR, config=TESS_CONFIG)  # você já está usando --psm 6
    return (txt or "").replace("\r", "").strip()
#FIM DO PROCESSAMNETO LOCAL


#FLUXO DE FUNCIONAMENTO DO PROCESSAMENTO (API DOCLING) + (MOTOR OCR+TERSSERACT)  EM CASOS DE (PDF)

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
        return md_acc #md

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
            # Fallback local com Tesseract (novo pipeline + preview)
            print(f"|°~°| Página {i}: API com imagem falhou/sem texto. Fallback Tesseract local...")
            speak("Não foi possível extrair texto desta página usando a API. Tentarei outro método!")

            # 1) pré-processa
            binimg, enhanced = preprocess_for_ocr(bgr, mode=OCR_MODE)

            # 2) preview/salvar por página
            base = os.path.splitext(path_pdf)[0]
            out_prefix = f"{base}_page{i:02d}"
            preview_preprocess(
                bgr, binimg, enhanced,
                title=f"Pré-processamento ({OCR_MODE}) - Página {i}",
                out_prefix=out_prefix
            )

            # 3) OCR em cima do binário
            txt = pytesseract.image_to_string(binimg, lang=TESS_LANG_STR, config=TESS_CONFIG)

            md_pages.append(f"## Página {i}\n\n{txt if txt else '*(sem texto detectável)*'}\n")


    md_all = "\n---\n".join(md_pages).strip()
    if md_all:
        outp = f"{os.path.splitext(path_pdf)[0]}_ocr.md"
        #with open(outp, "w", encoding="utf-8") as w:
         #   w.write(md_all)
        md_acc = acessibilizar_md(md_all)
        with open("saida_api.md", "w", encoding="utf-8") as w:
            w.write(md_acc)

        print(f"|°_°| OCR concluído. Markdown salvo em: {outp}")
        speak(f"OCR concluído. Markdown salvo em: {outp}")
        return md_all

        
    else:
        print("|°~°| Não foi possível extrair texto.")
        speak("Não foi possível extrair texto.")
    return md_all

#  FLUXO DE FUNCIONAMENTO DO PROCESSAMENTO (API DOCLING) + (MOTOR OCR+TERSSERACT)  EM CASOS DE (IMAGENS)
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

# Fallback Tesseract local (LOCAL - USO APENAS PARA TESTE DO AGENTE OS RESULTADOS OBTIDOS AQUI SERÃO USADOS PARA MELHORAMENTOS FUTUROS)
    print("|°~°| API sem texto. Fallback Tesseract local...")
    speak("Muito estranho, não encontrei texto. Tentarei de outro modo!")

    # 1) pré-processa para visualizar E alimentar o Tesseract
    binimg, enhanced = preprocess_for_ocr(bgr, mode=OCR_MODE)

    # 2) preview/salvar (prefixo usa o caminho da imagem de entrada)
    out_prefix = os.path.splitext(path_img)[0]
    preview_preprocess(
        bgr, binimg, enhanced,
        title=f"Pré-processamento ({OCR_MODE}) - {os.path.basename(path_img)}",
        out_prefix=out_prefix  # salva *_pre_bin.png e *_pre_enh.png se SAVE_PREVIEW=True
    )

    # 3) OCR em cima do binário
    txt = pytesseract.image_to_string(binimg, lang=TESS_LANG_STR, config=TESS_CONFIG)

    # 4) salva markdown
    md_local = f"## {os.path.basename(path_img)}\n\n{txt if txt else '*(sem texto detectável)*'}\n"
    outp = f"{os.path.splitext(path_img)[0]}_ocr.md"
    with open(outp, "w", encoding="utf-8") as w:
        w.write(md_local)

    print(f"|°_°| Markdown salvo em: {outp}")
    speak(f"Seu arquivo está pronto para leitura! Está salvo em: {outp}")
    return md_local


#MÉTODO PRINCIPAL DE EXECUÇÃO DO AGENTE_OCR
def main():
    inicio_wall = time.time()
    inicio_cpu = time.process_time()
    #TRATAMENTO DE ESPAÇOS VAZIOS
    path = input("ENTRE COM O ARQUIVO (PDF ou IMAGEM): ").strip().strip('"').strip("'")

    if not os.path.exists(path):
        print("|°~°| Caminho inválido.")
        speak("Deixe-me ver! Esse caminho é inválido ou a extensão do arquivo não é do tipo P D F ou imagem aceita. Tente outro!")
        
    elif is_pdf(path):
        _ = process_pdf(path)
    elif is_image_path(path):
        _ = process_image(path)
    else:
        # Tentar inferir pelo conteúdo SE extensão não ajuda
        kind = imghdr.what(path)
        if kind:
            _ = process_image(path)
        else:
            print("|°~°| Extensão do arquivo não valida ou é não reconhecida. Use .pdf ou uma imagem (.png/.jpg/.tif...).")
            speak("Deixe-me ver! A extensão do arquivo enviado não reconhecida. Use .pdf ou uma imagem .png")

    fim_wall = time.time()
    fim_cpu = time.process_time()
    print(f"|°_°| Tempo decorrido (wall): {fim_wall - inicio_wall:.2f}s")
    print(f"|°_°| Tempo de CPU: {fim_cpu - inicio_cpu:.2f}s")


if __name__ == "__main__":
    main()
