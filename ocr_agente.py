# agente_ocr.py — UFMA — EasyOCR (principal) + Tesseract (secundário)
# v2025-10-28

import os, io, base64, json, time
from typing import Tuple, List, Optional
import numpy as np
import cv2
import requests
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from pdf2image import convert_from_path
import math

# ========== ACESSIBILIDADE (.md) ==========
import re
TEXTO_IMAGEM_ALT   = "[Descrição: aqui havia uma imagem ou logotipo]"
TEXTO_PAGE_BREACK  = "[Descrição: próxima página]"

def acessibilizar_md(md: str,
    texto_imagem: str = TEXTO_IMAGEM_ALT,
    texto_page_break: str = TEXTO_PAGE_BREACK,
    substituir_imgs_markdown: bool = False) -> str:
    md = (md or "").replace("€", "e")
    def _marca_titulo(m):
        hashes = m.group(1); titulo = m.group(2).strip(); nivel = len(hashes)
        return f"\n[Início do título nível {nivel}: {titulo}]\n"
    md2 = re.sub(r'^(#{1,6})\s*(.+)$', _marca_titulo, md, flags=re.MULTILINE)
    md2 = re.sub(r'<!--\s*image\s*-->', texto_imagem, md2, flags=re.IGNORECASE)
    md2 = re.sub(r'<!--\s*page-break\s*-->', texto_page_break, md2, flags=re.IGNORECASE)
    if substituir_imgs_markdown:
        md2 = re.sub(r'!\[[^\]]*\]\([^)]+\)', texto_imagem, md2)
    return md2

def only_placeholders(md: str, min_real_chars: int = 40) -> bool:
    if not md: return True
    stripped = (md.replace("<!--image-->", "")
                  .replace("<!-- page-break -->", "")
                  .strip())
    return len(stripped) < min_real_chars

# --- TTS opcional (narração) ---
ENABLE_TTS = True          # ative/desative aqui
TTS_VOICE_HINTS = ("portugu", "brazil", "pt")  # preferências de voz
TTS_RATE_WPM = 170         # velocidade (palavras/min) ~ 150-190 bom p/ docs
TTS_VOLUME = 1.0           # 0.0..1.0

def speak(msg: str):
    """Fala uma mensagem curta. Não quebra o pipeline se der erro."""
    if not ENABLE_TTS or not msg:
        return
    try:
        import pyttsx3
        e = pyttsx3.init()
        # tenta pt-BR
        sel = None
        for v in e.getProperty("voices"):
            name = (v.name or "").lower()
            if any(h in name for h in TTS_VOICE_HINTS):
                sel = v.id; break
        if sel: e.setProperty("voice", sel)
        if TTS_RATE_WPM: e.setProperty("rate", int(TTS_RATE_WPM))
        if TTS_VOLUME is not None: e.setProperty("volume", float(TTS_VOLUME))
        e.say(msg)
        e.runAndWait()
    except Exception:
        # silencioso por padrão
        pass

# ========== CONFIGURAÇÕES ==========
# Motores (principal e secundário)
PRIMARY_ENGINE   = "easyocr"     # "easyocr" | "tesseract"
SECONDARY_ENGINE = "tesseract"   # "tesseract" | "easyocr"

# Ligar/desligar rotas
USE_API                 = True     # tenta API Docling/Servidor
USE_LOCAL               = False     # tenta OCR local
TRY_SECONDARY_IF_EMPTY  = True     # se vier vazio, tenta motor secundário (mesma rota)

# Pré-processamento (aplicado IGUAL para API e LOCAL, nos DOIS motores)
# "none"  = sem pré-processamento (manda original)
# "basic" = cinza + CLAHE leve + resize + unsharp leve
PREPROCESS_LEVEL = "basic"         # "none" | "basic"|"advanced"
PRE_ADV_DO_DESKEW   = True     # corrigir inclinação (Hough) se necessário
PRE_ADV_REMOVE_LINES= True    # remover linhas horizontais/verticais longas (formulários/tabelas)

# Tesseract (local)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESS_LANG_STR = "por+eng"
TESS_CONFIG   = "--oem 3 --psm 6"  # mude p/ 3/4/6/11 conforme layout

# EasyOCR (local)
EASYOCR_LANGS = ['pt', 'en']
EASYOCR_GPU   = True

# API Docling/Conversor
API_ENDPOINTS = [
    "http://200.137.132.64:5005/v1/convert/source",
]
API_TIMEOUT   = 120

# Visualização salva/preview
SHOW_PREVIEW     = True
SAVE_PREVIEW     = True
PREVIEW_MAX_WIDTH= 1800

# Modo “perfil” (apenas rótulo para logs/arquivos)
OCR_MODE_LABEL = "documento"

# ========== UTIL ==========
def is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")

def is_image_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, FileNotFoundError, OSError):
        return False

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def np_bgr_to_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok: raise RuntimeError("Falha ao codificar PNG.")
    return buf.tobytes()

def show_image_bgr(bgr: np.ndarray, title: str = "Visualização"):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    plt.figure(); plt.imshow(rgb); plt.title(title); plt.axis("off"); plt.show()

# ========== PRÉ-PROCESSAMENTO PADRONIZADO ==========
#advanced
def _resize_min(img, min_side=1200):
    h, w = img.shape[:2]
    s = min(h, w)
    if s >= min_side: return img
    f = min_side / float(s)
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)

def _illumination_correction(gray, ksize=41):
    bg = cv2.medianBlur(gray, ksize)
    bg = np.clip(bg, 1, 255)
    return cv2.divide(gray, bg, scale=255)

def _unsharp(img, sigma=1.0, amount=1.6):
    blur = cv2.GaussianBlur(img, (0,0), sigma)
    return cv2.addWeighted(img, amount, blur, -(amount-1), 0)

def _sauvola(gray, win=31, k=0.32, R=128.0):
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (win,win), normalize=True)
    sqr  = cv2.boxFilter(g*g, -1, (win,win), normalize=True)
    var  = np.clip(sqr - mean*mean, 0, None)
    std  = np.sqrt(var)
    thr  = mean * (1 + k*((std/R) - 1))
    return (g > thr).astype(np.uint8)*255

def _rotate_bound(image, angle_deg):
    (h, w) = image.shape[:2]
    cX, cY = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _estimate_skew_angle(gray, angle_limit=20):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
    if lines is None: return 0.0
    angles = []
    for rho_theta in lines[:200]:
        rho, theta = rho_theta[0]
        deg = (theta * 180.0 / np.pi) - 90.0
        if deg > 90:  deg -= 180
        if deg < -90: deg += 180
        if abs(deg) <= angle_limit:
            angles.append(deg)
    return float(np.median(angles)) if angles else 0.0

def _remove_form_lines(binimg, min_len_ratio=0.45):
    H, W = binimg.shape
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, W//40), 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, H//40)))
    hor = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, hor_kernel, iterations=1)
    ver = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, ver_kernel, iterations=1)
    mask = cv2.bitwise_or(hor, ver)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, num_labels):
        x,y,w,h,area = stats[i]
        if w >= int(min_len_ratio*W) or h >= int(min_len_ratio*H):
            keep[labels==i] = 255
    return cv2.bitwise_and(binimg, cv2.bitwise_not(keep))

def preprocess_advanced(bgr: np.ndarray, *, do_deskew=True, remove_lines=False) -> np.ndarray:
    # 1) tamanho mínimo
    img  = _resize_min(bgr, min_side=1200)
    # 2) cinza + correção de iluminação
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = _illumination_correction(gray, ksize=41)
    # 3) realce local
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8)).apply(norm)
    # 4) denoise leve + unsharp
    den   = cv2.bilateralFilter(clahe, 7, 60, 60)
    sharp = _unsharp(den, sigma=1.0, amount=1.7)
    # 5) binarização primária
    binim = _sauvola(sharp, win=31, k=0.32, R=128.0)

    # 6) deskew (aplica na imagem realçada)
    if do_deskew:
        ang = _estimate_skew_angle(binim, angle_limit=20)
        if abs(ang) >= 0.5:
            rot_gray = _rotate_bound(sharp, -ang)
            binim    = _sauvola(rot_gray, win=31, k=0.32, R=128.0)

    # 7) remoção de linhas (opcional)
    if remove_lines:
        binim = _remove_form_lines(binim, min_len_ratio=0.45)

    # 8) morfologia leve
    binim = cv2.morphologyEx(binim, cv2.MORPH_OPEN,  np.ones((2,2), np.uint8), 1)
    binim = cv2.morphologyEx(binim, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), 1)

    # 9) retorna em BGR (3 canais)
    return cv2.cvtColor(binim, cv2.COLOR_GRAY2BGR)

#advanced
def preprocess_none(bgr: np.ndarray) -> np.ndarray:
    return bgr

def preprocess_basic(bgr: np.ndarray) -> np.ndarray:
    # Cinza + CLAHE leve + resize (para altura mínima) + unsharp leve
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

    # resize se a menor dimensão < 1000 px
    h, w = clahe.shape
    min_target = 1000
    if min(h, w) < min_target:
        scale = min_target / float(min(h, w))
        clahe = cv2.resize(clahe, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # unsharp leve
    blur  = cv2.GaussianBlur(clahe, (0,0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.4, blur, -0.4, 0)

    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

def get_preprocessor(level: str):
    level = (level or "none").lower()
    if level == "basic":
        return preprocess_basic
    if level == "advanced":
        # fecha sobre as flags globais para manter a mesma assinatura (bgr)->bgr
        def _adv(bgr):
            return preprocess_advanced(bgr, do_deskew=PRE_ADV_DO_DESKEW, remove_lines=PRE_ADV_REMOVE_LINES)
        return _adv
    return preprocess_none

def preview_preprocess(orig_bgr: np.ndarray, pre_bgr: np.ndarray, title: str, out_prefix: Optional[str]):
    if not (SHOW_PREVIEW or SAVE_PREVIEW): return
    def _resize_max(img, maxw=PREVIEW_MAX_WIDTH):
        h, w = img.shape[:2]
        if w <= maxw: return img
        s = maxw/float(w)
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    rgb_o = _resize_max(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
    rgb_p = _resize_max(cv2.cvtColor(pre_bgr,  cv2.COLOR_BGR2RGB))

    if SHOW_PREVIEW:
        plt.figure(figsize=(12,5))
        plt.suptitle(title)
        plt.subplot(1,2,1); plt.imshow(rgb_o); plt.title("Original"); plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(rgb_p); plt.title("Pré-processado"); plt.axis("off")
        plt.tight_layout(); plt.show()

    if SAVE_PREVIEW and out_prefix:
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        cv2.imwrite(f"{out_prefix}_orig.png", cv2.cvtColor(rgb_o, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{out_prefix}_pre.png",  cv2.cvtColor(rgb_p, cv2.COLOR_RGB2BGR))

# ========== API (payload com motor dinâmico + mesmo pré-processamento) ==========
def build_payload_pdf(pdf_b64: str, ocr_engine: str) -> dict:
    # Idiomas por motor
    if ocr_engine == "easyocr":
        ocr_lang = ["pt","en"]
    else:
        ocr_lang = ["por+eng"]
    return {
        "options": {
            "from_formats": ["pdf", "image"],
            "to_formats": ["md"],
            "image_export_mode": "placeholder",
            "do_ocr": True,
            "force_ocr": False,
            "ocr_engine": ocr_engine,
            "ocr_lang": ocr_lang,
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
        "sources": [{"base64_string": pdf_b64, "filename": "input.pdf", "kind": "file"}],
        "target": {"kind": "inbody"}
    }

def build_payload_image(img_png_b64: str, filename: str, ocr_engine: str) -> dict:
    if ocr_engine == "easyocr":
        ocr_lang = ["pt","en"]
    else:
        ocr_lang = ["por+eng"]
    return {
        "options": {
            "from_formats": ["image"],
            "to_formats": ["md"],
            "image_export_mode": "placeholder",
            "do_ocr": True,
            "force_ocr": False,
            "ocr_engine": ocr_engine,
            "ocr_lang": ocr_lang,
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
        "sources": [{"base64_string": img_png_b64, "filename": filename, "kind": "file"}],
        "target": {"kind": "inbody"}
    }

def call_api(payload: dict) -> str:
    last_err = None
    for url in API_ENDPOINTS:
        try:
            r = requests.post(url, json=payload, timeout=API_TIMEOUT)
            if r.status_code == 422:
                print(f"|°~°| 422 em {url}: {r.text[:400]}"); continue
            r.raise_for_status()
            data = r.json()
            md = ""
            if isinstance(data, dict):
                if "document" in data:
                    md = (data["document"] or {}).get("md_content") or ""
                if not md and "documents" in data and data["documents"]:
                    md = (data["documents"][0] or {}).get("md_content") or ""
            if md: return md
        except requests.RequestException as e:
            last_err = e
            print(f"|°~°| API falhou em {url}: {e}")
    if last_err: print("|°~°| API - última exceção:", last_err)
    return ""

# ========== OCR LOCAL (mesmo pré-processamento nos dois motores) ==========
def ocr_local_easyocr(bgr: np.ndarray) -> str:
    try:
        import easyocr
    except ImportError:
        print("|°~°| EasyOCR não instalado (pip install easyocr).")
        return ""
    reader = easyocr.Reader(EASYOCR_LANGS, gpu=EASYOCR_GPU)
    # text-only concat
    results = reader.readtext(bgr, detail=1)
    lines = []
    # ordenar por top-left y, depois x (rústico)
    def tl(box): 
        xs = [p[0] for p in box]; ys = [p[1] for p in box]; 
        return (min(ys), min(xs))
    results.sort(key=lambda r: tl(r[0]))
    for box, txt, conf in results:
        if (txt or "").strip():
            lines.append(txt.strip())
    return "\n".join(lines).strip()

def ocr_local_tesseract(bgr: np.ndarray) -> str:
    return (pytesseract.image_to_string(bgr, lang=TESS_LANG_STR, config=TESS_CONFIG) or "").strip()

# ========== FLUXOS ==========
def process_pdf(path_pdf: str) -> str:
    # 1) API com PDF (motor principal)
    if USE_API:
        with open(path_pdf, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode("utf-8")
        print(f"|°_°| API(PDF) [{PRIMARY_ENGINE}]...")
        md = call_api(build_payload_pdf(pdf_b64, PRIMARY_ENGINE))
        if md and not only_placeholders(md):
            md_acc = acessibilizar_md(md)
            with open("saida_api.md", "w", encoding="utf-8") as w: w.write(md_acc)
            return md_acc

        if TRY_SECONDARY_IF_EMPTY:
            print(f"|°~°| Vazio. API(PDF) [{SECONDARY_ENGINE}]...")
            md2 = call_api(build_payload_pdf(pdf_b64, SECONDARY_ENGINE))
            if md2 and not only_placeholders(md2):
                md_acc = acessibilizar_md(md2)
                with open("saida_api.md", "w", encoding="utf-8") as w: w.write(md_acc)
                return md_acc

    # 2) Converte páginas para imagem e repete API por página (aplicando mesmo pré-processamento)
    pages_pil = convert_from_path(path_pdf, dpi=300)
    md_pages: List[str] = []
    preproc = get_preprocessor(PREPROCESS_LEVEL)

    if USE_API:
        for i, pil_img in enumerate(pages_pil, start=1):
            bgr = pil_to_bgr(pil_img)
            bgr_pre = preproc(bgr)
            preview_preprocess(bgr, bgr_pre, f"Pré-processamento ({PREPROCESS_LEVEL}) — pág. {i}", 
                               out_prefix=f"{os.path.splitext(path_pdf)[0]}_p{i:02d}")

            png_b64 = base64.b64encode(np_bgr_to_png_bytes(bgr_pre)).decode("utf-8")
            print(f"|°_°| API(IMG) pág.{i} [{PRIMARY_ENGINE}]...")
            md_img = call_api(build_payload_image(png_b64, f"page_{i}.png", PRIMARY_ENGINE))
            if not md_img and TRY_SECONDARY_IF_EMPTY:
                print(f"|°~°| Vazio. API(IMG) pág.{i} [{SECONDARY_ENGINE}]...")
                md_img = call_api(build_payload_image(png_b64, f"page_{i}.png", SECONDARY_ENGINE))
            if md_img and not only_placeholders(md_img):
                md_pages.append(f"<!-- page-break -->\n{md_img}")

        if md_pages:
            md_all = "\n\n".join(md_pages).strip()
            md_acc = acessibilizar_md(md_all)
            with open("saida_api.md", "w", encoding="utf-8") as w: w.write(md_acc)
            return md_acc

    # 3) LOCAL por página (aplica MESMO pré-processamento, primeiro motor principal, depois secundário)
    if USE_LOCAL:
        out_pages = []
        for i, pil_img in enumerate(pages_pil, start=1):
            bgr = pil_to_bgr(pil_img)
            bgr_pre = preproc(bgr)
            txt = ""
            if PRIMARY_ENGINE == "easyocr":
                txt = ocr_local_easyocr(bgr_pre)
                if (not txt.strip()) and TRY_SECONDARY_IF_EMPTY:
                    txt = ocr_local_tesseract(bgr_pre)
            else:
                txt = ocr_local_tesseract(bgr_pre)
                if (not txt.strip()) and TRY_SECONDARY_IF_EMPTY:
                    txt = ocr_local_easyocr(bgr_pre)
            out_pages.append(f"## Página {i}\n\n{(txt or '*(sem texto detectável)*')}\n")

        md_local = "\n---\n".join(out_pages).strip()
        with open(f"{os.path.splitext(path_pdf)[0]}_ocr.md", "w", encoding="utf-8") as w: w.write(md_local)
        return md_local

    return ""

def process_image(path_img: str) -> str:
    if not is_image_path(path_img) or not is_valid_image(path_img):
        print("|°~°| Arquivo não parece ser uma imagem válida."); return ""

    bgr = cv2.imread(path_img, cv2.IMREAD_COLOR)
    if bgr is None:
        print("|°~°| Falha ao carregar a imagem."); return ""

    preproc = get_preprocessor(PREPROCESS_LEVEL)
    bgr_pre = preproc(bgr)
    preview_preprocess(bgr, bgr_pre, f"Pré-processamento ({PREPROCESS_LEVEL}) — {os.path.basename(path_img)}",
                       out_prefix=os.path.splitext(path_img)[0])

    # 1) API com imagem (principal → secundário)
    if USE_API:
        img_b64 = base64.b64encode(np_bgr_to_png_bytes(bgr_pre)).decode("utf-8")
        print(f"|°_°| API(IMG) [{PRIMARY_ENGINE}]...")
        md = call_api(build_payload_image(img_b64, os.path.basename(path_img), PRIMARY_ENGINE))
        if not md and TRY_SECONDARY_IF_EMPTY:
            print(f"|°~°| Vazio. API(IMG) [{SECONDARY_ENGINE}]...")
            md = call_api(build_payload_image(img_b64, os.path.basename(path_img), SECONDARY_ENGINE))
        if md and not only_placeholders(md):
            md_acc = acessibilizar_md(md)
            with open("saida_api.md", "w", encoding="utf-8") as w: w.write(md_acc)
            return md_acc

    # 2) LOCAL (aplica mesmo pré-processamento)
    if USE_LOCAL:
        if PRIMARY_ENGINE == "easyocr":
            txt = ocr_local_easyocr(bgr_pre)
            if (not txt.strip()) and TRY_SECONDARY_IF_EMPTY:
                txt = ocr_local_tesseract(bgr_pre)
        else:
            txt = ocr_local_tesseract(bgr_pre)
            if (not txt.strip()) and TRY_SECONDARY_IF_EMPTY:
                txt = ocr_local_easyocr(bgr_pre)

        md_local = f"## {os.path.basename(path_img)}\n\n{(txt or '*(sem texto detectável)*')}\n"
        with open(f"{os.path.splitext(path_img)[0]}_ocr.md", "w", encoding="utf-8") as w: w.write(md_local)
        return md_local

    return ""

# ========== MAIN ==========
def main():
    speak("Olá! Envie um arquivo PDF ou imagem para leitura.")
    path = input("ENTRE COM O ARQUIVO (PDF ou IMAGEM): ").strip().strip('"').strip("'")
    if not os.path.exists(path):
        print("|°~°| Caminho inválido.")
        speak("Caminho inválido. Tente novamente.")
        return
    t0 = time.time()
    if is_pdf(path):
        speak("Documento PDF detectado. Iniciando processamento.")
        _ = process_pdf(path)
    elif is_image_path(path) or is_valid_image(path):
        speak("Imagem detectada. Iniciando processamento.")
        _ = process_image(path)
    else:
        print("|°~°| Use .pdf ou uma imagem válida (.png/.jpg/.tif/.bmp/.webp).")
        speak("Formato não suportado. Use PDF ou imagem válida.")
        return
    dt = time.time() - t0
    print(f"|°_°| Tempo total: {dt:.2f}s")
    speak(f"Processamento concluído em {int(dt)} segundos.")


if __name__ == "__main__":
    main()
