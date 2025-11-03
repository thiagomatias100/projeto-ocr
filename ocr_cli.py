#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR CLI — EasyOCR (principal) + Tesseract (secundário)
Uso: ocr-cli [opções] <arquivo>

Exemplos:
  ocr-cli --engine easyocr --api --pre basic ./arquivo.pdf
  ocr-cli --engine easyocr --no-api --local --out md ./foto.jpg
  ocr-cli --engine tesseract --pre none --local ./scan.png
"""

import os, sys, base64, json, time, argparse, re, warnings
from typing import Optional, List
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
# Evitar DecompressionBomb em páginas enormes (Pillow)
Image.MAX_IMAGE_PIXELS = 300_000_000  # ajuste se necessário
try:
    warnings.simplefilter("ignore", Image.DecompressionBombWarning)
except Exception:
    pass

# ---- Dependências opcionais (tratamos com try/except) ----
try:
    import requests
except Exception:
    requests = None

try:
    import pytesseract
except Exception:
    pytesseract = None

# ------- Acessibilidade Markdown -------
def acessibilizar_md(md: str) -> str:
    TEXTO_IMAGEM_ALT  = "[Descrição: aqui havia uma imagem ou logotipo]"
    TEXTO_PAGE_BREAK  = "[Descrição: próxima página]"
    md = (md or "").replace("€","e")
    md = re.sub(r'<!--\s*image\s*-->', TEXTO_IMAGEM_ALT, md, flags=re.I)
    md = re.sub(r'<!--\s*page-break\s*-->', TEXTO_PAGE_BREAK, md, flags=re.I)
    def _marca_titulo(m):
        hashes, titulo = m.group(1), m.group(2).strip()
        return f"\n[Início do título nível {len(hashes)}: {titulo}]\n"
    md = re.sub(r'^(#{1,6})\s*(.+)$', _marca_titulo, md, flags=re.M)
    return md

def md_parece_so_imagem(md: str) -> bool:
    if not md or not md.strip():
        return True
    temp = re.sub(r'<!--\s*page-break\s*-->', '', md, flags=re.I)
    temp = re.sub(r'<!--\s*image\s*-->', '', temp, flags=re.I)
    temp = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', temp)
    return len(temp.strip()) == 0

# ------- Utils -------
def is_pdf(p): return p.lower().endswith(".pdf")
def is_image_path(p): return os.path.splitext(p)[1].lower() in {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"}

def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as im: im.verify()
        return True
    except Exception:
        return False

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def np_bgr_to_png_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok: raise RuntimeError("Falha ao codificar PNG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

# ------- Pré-processamento -------
def boost_then_gray(bgr: np.ndarray, alpha: float = 1.25, beta: float = 12) -> np.ndarray:
    boosted = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)
    gray = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def preprocess(bgr: np.ndarray, level: str, alpha: float = 1.25, beta: float = 25) -> np.ndarray:
    level = (level or "none").lower()
    if level == "none":
        return bgr
    if level == "basic":
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(2.0,(8,8)).apply(gray)
        h,w = clahe.shape
        if min(h,w) < 1000:
            s = 1000.0/min(h,w)
            clahe = cv2.resize(clahe, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        blur  = cv2.GaussianBlur(clahe,(0,0),1.0)
        sharp = cv2.addWeighted(clahe, 1.4, blur, -0.4, 0)
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    if level == "medium":
        return boost_then_gray(bgr, alpha=alpha, beta=beta)
    return bgr

# ------- OCR local -------
def ocr_local_easyocr(bgr: np.ndarray, langs: List[str], gpu: bool, fast: bool) -> str:
    try:
        import easyocr
    except Exception:
        return ""
    reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
    if fast:
        res = reader.readtext(bgr, detail=0, paragraph=False, decoder='greedy', beamWidth=1, rotation_info=[])
        txt = "\n".join([r for r in res if isinstance(r,str)]).strip()
    else:
        res = reader.readtext(bgr, detail=0, paragraph=True)
        txt = "\n".join([r for r in res if isinstance(r,str)]).strip()
    return txt

def ocr_local_tesseract(bgr: np.ndarray, tess_lang: str, tess_config: str) -> str:
    if pytesseract is None: return ""
    return (pytesseract.image_to_string(bgr, lang=tess_lang, config=tess_config) or "").strip()

# ------- API -------
def build_payload_image(img_png_b64: str, filename: str, engine: str,
                        langs_easy: List[str], langs_tess: str, force_ocr: bool) -> dict:
    ocr_lang = (langs_easy if engine == "easyocr" else [langs_tess])
    return {
        "options": {
            "from_formats": ["image"],
            "to_formats": ["md"],
            "image_export_mode": "placeholder",
            "do_ocr": True,
            "force_ocr": bool(force_ocr),
            "ocr_engine": engine,
            "ocr_lang": ocr_lang,
            "pdf_backend": "pypdfium2",
            "table_mode": "fast",
            "pipeline": "standard",
            "include_images": True,
            "images_scale": 1,
            "md_page_break_placeholder": "<!-- page-break -->"
        },
        "sources": [{"base64_string": img_png_b64, "filename": filename, "kind": "file"}],
        "target": {"kind":"inbody"}
    }

def build_payload_pdf(pdf_b64: str, engine: str,
                      langs_easy: List[str], langs_tess: str, force_ocr: bool) -> dict:
    return {
        "options": {
            "from_formats": ["pdf","image"],
            "to_formats": ["md"],
            "image_export_mode": "placeholder",
            "do_ocr": True,
            "force_ocr": bool(force_ocr),
            "ocr_engine": engine,
            "ocr_lang": (langs_easy if engine=="easyocr" else [langs_tess]),
            "pdf_backend": "pypdfium2",
            "table_mode": "fast",
            "pipeline": "standard",
            "include_images": True,
            "images_scale": 1,
            "md_page_break_placeholder": "<!-- page-break -->"
        },
        "sources": [{"base64_string": pdf_b64, "filename": "input.pdf", "kind":"file"}],
        "target": {"kind":"inbody"}
    }

def call_api(endpoints: List[str], payload: dict, timeout: int=120, debug: bool=False) -> str:
    if requests is None:
        return ""
    last_err = None
    for url in endpoints:
        try:
            if debug:
                print(f"[API] POST {url} ...")
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code == 422:
                if debug: print("[API] 422 Unprocessable Entity")
                continue
            r.raise_for_status()
            data = r.json()
            md = ""
            if isinstance(data, dict):
                if "document" in data:
                    md = (data["document"] or {}).get("md_content") or ""
                if not md and "documents" in data and data["documents"]:
                    md = (data["documents"][0] or {}).get("md_content") or ""
            if debug:
                print(f"[API] OK md_len={len(md or '')}")
            if md:
                return md
        except Exception as e:
            last_err = e
    if debug and last_err:
        print(f"[API] erro final: {last_err}")
    return ""

# --- variantes API ---
def upscale_15(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, (max(1,int(w*1.5)), max(1,int(h*1.5))), interpolation=cv2.INTER_CUBIC)

def binarize_light(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    thr = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 8)
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

def api_try_variants_for_image(bgr: np.ndarray, filename: str, args) -> str:
    variants = [
        ("raw", bgr),
        ("up", upscale_15(bgr)),
        ("bin", binarize_light(bgr)),
    ]
    for tag, img in variants:
        try:
            b64 = np_bgr_to_png_b64(img)
            md = call_api(
                args.api_endpoint,
                build_payload_image(b64, f"{filename}_{tag}.png",
                                    args.engine, args.langs_easy, args.langs_tess, args.force_ocr),
                debug=args.debug_api
            )
            if md.strip() and not md_parece_so_imagem(md):
                return md
            if args.try_secondary:
                sec = ("tesseract" if args.engine=="easyocr" else "easyocr")
                md2 = call_api(
                    args.api_endpoint,
                    build_payload_image(b64, f"{filename}_{tag}.png",
                                        sec, args.langs_easy, args.langs_tess, args.force_ocr),
                    debug=args.debug_api
                )
                if md2.strip() and not md_parece_so_imagem(md2):
                    return md2
        except Exception:
            pass
    return ""

# --- TTS ---
def speak(msg: str, enable: bool, rate: Optional[int], volume: Optional[float]):
    if not enable or not msg: return
    try:
        import pyttsx3
        e = pyttsx3.init()
        sel = None
        for v in e.getProperty("voices"):
            nm = (v.name or "").lower()
            if any(h in nm for h in ("portugu", "brazil", "pt")):
                sel = v.id; break
        if sel: e.setProperty("voice", sel)
        if rate: e.setProperty("rate", int(rate))
        if volume is not None: e.setProperty("volume", float(volume))
        e.say(msg); e.runAndWait()
    except Exception:
        pass

# ------- Processadores -------
def process_image(path: str, args) -> str:
    if not is_image_path(path) or not is_valid_image(path):
        print("(!) Arquivo de imagem inválido."); return ""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        print("(!) Falha ao carregar imagem."); return ""
    bgr_pre = preprocess(bgr, args.pre, args.boost_alpha, args.boost_beta)

    if args.api:
        md = api_try_variants_for_image(bgr_pre, os.path.splitext(os.path.basename(path))[0], args)
        if md.strip():
            return acessibilizar_md(md) if args.accessible else md

    if args.local:
        if args.engine == "easyocr":
            txt = ocr_local_easyocr(bgr_pre, args.langs_easy, args.gpu, args.fast)
            if (not txt.strip()) and args.try_secondary:
                txt = ocr_local_tesseract(bgr_pre, args.langs_tess, args.tess_config)
        else:
            txt = ocr_local_tesseract(bgr_pre, args.langs_tess, args.tess_config)
            if (not txt.strip()) and args.try_secondary:
                txt = ocr_local_easyocr(bgr_pre, args.langs_easy, args.gpu, args.fast)
        if args.accessible and txt.strip():
            txt = f"## Página única\n\n{txt}"
        return txt
    return ""

def process_pdf(path: str, args) -> str:
    if args.api:
        with open(path, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode("utf-8")
        md = call_api(args.api_endpoint,
                      build_payload_pdf(pdf_b64, args.engine, args.langs_easy, args.langs_tess, args.force_ocr),
                      debug=args.debug_api)
        if md.strip() and not md_parece_so_imagem(md):
            return acessibilizar_md(md) if args.accessible else md
        if args.try_secondary:
            sec = ("tesseract" if args.engine=="easyocr" else "easyocr")
            md2 = call_api(args.api_endpoint,
                           build_payload_pdf(pdf_b64, sec, args.langs_easy, args.langs_tess, args.force_ocr),
                           debug=args.debug_api)
            if md2.strip() and not md_parece_so_imagem(md2):
                return acessibilizar_md(md2) if args.accessible else md2

    try:
        from pdf2image import convert_from_path
    except Exception:
        print("(!) pdf2image não disponível para fallback por página.")
        return ""

    pages = convert_from_path(path, dpi=args.dpi_fallback)
    out_pages = []
    for i, pil in enumerate(pages, start=1):
        w0, h0 = pil.size
        max_pixels = int(args.max_mpix * 1_000_000)
        if (w0 * h0) > max_pixels:
            scale = (max_pixels / float(w0 * h0)) ** 0.5
            pil = pil.resize((max(1, int(w0*scale)), max(1, int(h0*scale))), resample=Image.LANCZOS)
        bgr = preprocess(pil_to_bgr(pil), args.pre, args.boost_alpha, args.boost_beta)

        md = api_try_variants_for_image(bgr, f"page_{i}", args)
        if md.strip():
            out_pages.append(f"<!-- page-break -->\n{md}")
            continue

    if out_pages:
        md_all = "\n\n".join(out_pages).strip()
        return acessibilizar_md(md_all) if args.accessible else md_all
    return ""

# ------- Main / CLI -------
def main(argv=None):
    p = argparse.ArgumentParser(prog="ocr-cli", description="OCR CLI — EasyOCR + Tesseract")

    p.add_argument("arquivo", help="PDF ou imagem")
    p.add_argument("--engine", choices=["easyocr","tesseract"], default="easyocr", help="motor principal")

    p.add_argument("--try-secondary", dest="try_secondary", action="store_true")
    p.add_argument("--no-try-secondary", dest="try_secondary", action="store_false")
    p.set_defaults(try_secondary=True)

    p.add_argument("--api", dest="api", action="store_true")
    p.add_argument("--no-api", dest="api", action="store_false")
    p.set_defaults(api=True)

    p.add_argument("--local", dest="local", action="store_true")
    p.add_argument("--no-local", dest="local", action="store_false")
    p.set_defaults(local=False)

    p.add_argument("--force-ocr", dest="force_ocr", action="store_true")
    p.add_argument("--no-force-ocr", dest="force_ocr", action="store_false")
    p.set_defaults(force_ocr=True)

    p.add_argument("--debug-api", action="store_true")

    p.add_argument("--pre", choices=["none","basic","medium"], default="none")
    p.add_argument("--boost-alpha", type=float, default=1.25)
    p.add_argument("--boost-beta", type=float, default=12)

    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--dpi-fallback", type=int, default=150)
    p.add_argument("--max-mpix", type=float, default=30.0)

    p.add_argument("--langs-easy", default="pt,en")
    p.add_argument("--langs-tess", default="por+eng")
    p.add_argument("--tess-cmd", default=None)
    p.add_argument("--tess-config", default="--oem 3 --psm 6")

    p.add_argument("--api-endpoint", action="append",
                   default=["http://200.137.132.64:5005/v1/convert/source"])

    p.add_argument("--out", choices=["md","json","txt"], default="md")
    p.add_argument("--accessible", dest="accessible", action="store_true")
    p.add_argument("--no-accessible", dest="accessible", action="store_false")
    p.set_defaults(accessible=True)

    p.add_argument("--gpu", action="store_true")
    p.add_argument("--fast", action="store_true")

    p.add_argument("--outdir", default=None)

    p.add_argument("--tts", action="store_true")
    p.add_argument("--tts-rate", type=int, default=170)
    p.add_argument("--tts-volume", type=float, default=1.0)

    args = p.parse_args(argv)

    args.langs_easy = [s.strip() for s in args.langs_easy.split(",") if s.strip()]
    if pytesseract and args.tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tess_cmd

    if not args.api and not args.local:
        print("(!) Nenhum modo de OCR habilitado.")
        sys.exit(2)

    in_abs = os.path.abspath(args.arquivo)
    in_dir = os.path.dirname(in_abs) or "."
    outdir = args.outdir or in_dir
    os.makedirs(outdir, exist_ok=True)

    speak("Iniciando leitura de documento.", args.tts, args.tts_rate, args.tts_volume)
    t0 = time.time()

    if not os.path.exists(args.arquivo):
        print("(!) Caminho inválido."); sys.exit(2)

    if is_pdf(args.arquivo):
        res = process_pdf(args.arquivo, args)
    elif is_image_path(args.arquivo) or is_valid_image(args.arquivo):
        res = process_image(args.arquivo, args)
    else:
        print("(!) Use PDF ou uma imagem válida."); sys.exit(2)

    dt = time.time() - t0
    if not res:
        print("(!) Não foi possível extrair texto."); sys.exit(1)

    if args.accessible and res and not re.search(r'\[Início do título nível', res):
        try:
            res = acessibilizar_md(res)
        except Exception:
            pass

    base = os.path.splitext(os.path.basename(args.arquivo))[0]
    if args.out == "md":
        outp = os.path.join(outdir, f"{base}_ocr{'_acc' if args.accessible else ''}.md")
        with open(outp, "w", encoding="utf-8") as w:
            w.write(res)
    elif args.out == "json":
        outp = os.path.join(outdir, f"{base}_ocr.json")
        with open(outp, "w", encoding="utf-8") as w:
            json.dump({"arquivo": args.arquivo, "conteudo": res}, w, ensure_ascii=False, indent=2)
    else:
        outp = os.path.join(outdir, f"{base}_ocr.txt")
        with open(outp, "w", encoding="utf-8") as w:
            w.write(res)

    print(f"[OK] Saída salva em: {outp} | tempo: {dt:.2f}s")
    speak(f"Processamento concluído. Salvo em {outp}", args.tts, args.tts_rate, args.tts_volume)

if __name__ == "__main__":
    main()
