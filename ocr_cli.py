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

import os, sys, base64, json, time, argparse, re
from typing import Optional, List, Tuple
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

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
    """
    1) Reforça contraste (alpha) e brilho (beta) no BGR
    2) Converte para GRAY
    Retorna BGR (3 canais) para compatibilidade com o pipeline.
    """
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
     # 1) brilho/contraste -> 2) cinza
         return boost_then_gray(bgr, alpha=alpha, beta=beta)
    # fallback
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
def build_payload_image(img_png_b64: str, filename: str, engine: str, langs_easy: List[str], langs_tess: str) -> dict:
    if engine == "easyocr":
        ocr_lang = langs_easy
    else:
        ocr_lang = [langs_tess]
    return {
        "options": {
            "from_formats": ["image"],
            "to_formats": ["md"],
            "image_export_mode": "placeholder",
            "do_ocr": True,
            "force_ocr": False,
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

def build_payload_pdf(pdf_b64: str, engine: str, langs_easy: List[str], langs_tess: str) -> dict:
    return {
        "options": {
            "from_formats": ["pdf","image"],
            "to_formats": ["md"],
            "image_export_mode": "placeholder",
            "do_ocr": True,
            "force_ocr": False,
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

def call_api(endpoints: List[str], payload: dict, timeout: int=120) -> str:
    if requests is None:
        return ""
    last_err = None
    for url in endpoints:
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code == 422: continue
            r.raise_for_status()
            data = r.json()
            md = ""
            if isinstance(data, dict):
                if "document" in data:
                    md = (data["document"] or {}).get("md_content") or ""
                if not md and "documents" in data and data["documents"]:
                    md = (data["documents"][0] or {}).get("md_content") or ""
            if md: return md
        except Exception as e:
            last_err = e
    return ""

# ------- TTS -------
def speak(msg: str, enable: bool, rate: Optional[int], volume: Optional[float]):
    if not enable or not msg: return
    try:
        import pyttsx3
        e = pyttsx3.init()
        # voz pt-BR se disponível
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
    if bgr is None: print("(!) Falha ao carregar imagem."); return ""
    bgr_pre = preprocess(bgr, args.pre, args.boost_alpha, args.boost_beta)

    # API primeiro?
    if args.api:
        img_b64 = np_bgr_to_png_b64(bgr_pre)
        md = call_api(args.api_endpoint, build_payload_image(img_b64, os.path.basename(path),
                                                             args.engine, args.langs_easy, args.langs_tess))
        if md.strip():
            md_acc = acessibilizar_md(md) if args.accessible else md
            return md_acc

        if args.try_secondary:
            sec = ("tesseract" if args.engine=="easyocr" else "easyocr")
            md2 = call_api(args.api_endpoint, build_payload_image(img_b64, os.path.basename(path),
                                                                  sec, args.langs_easy, args.langs_tess))
            if md2.strip():
                return acessibilizar_md(md2) if args.accessible else md2

    # Local
    if args.local:
        if args.engine == "easyocr":
            txt = ocr_local_easyocr(bgr_pre, args.langs_easy, args.gpu, args.fast)
            if (not txt.strip()) and args.try_secondary:
                txt = ocr_local_tesseract(bgr_pre, args.langs_tess, args.tess_config)
        else:
            txt = ocr_local_tesseract(bgr_pre, args.langs_tess, args.tess_config)
            if (not txt.strip()) and args.try_secondary:
                txt = ocr_local_easyocr(bgr_pre, args.langs_easy, args.gpu, args.fast)

        # Ajuda a acessibilização (há um título mínimo)
        if args.accessible and txt.strip():
            txt = f"## Página única\n\n{txt}"
        return txt
    return ""

def process_pdf(path: str, args) -> str:
    with open(path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

    # API direto no PDF (documento inteiro)
    if args.api:
        md = call_api(args.api_endpoint, build_payload_pdf(pdf_b64, args.engine, args.langs_easy, args.langs_tess))
        if md.strip():
            return acessibilizar_md(md) if args.accessible else md
        if args.try_secondary:
            sec = ("tesseract" if args.engine=="easyocr" else "easyocr")
            md2 = call_api(args.api_endpoint, build_payload_pdf(pdf_b64, sec, args.langs_easy, args.langs_tess))
            if md2.strip():
                return acessibilizar_md(md2) if args.accessible else md2

    # Converter páginas → imagem e repetir a lógica por página
    try:
        from pdf2image import convert_from_path
    except Exception:
        print("(!) pdf2image não disponível para fallback por página.")
        return ""

    pages = convert_from_path(path, dpi=args.dpi)
    out_pages = []
    for i, pil in enumerate(pages, start=1):
        bgr = preprocess(pil_to_bgr(pil), args.pre, args.boost_alpha, args.boost_beta)
        if args.api:
            b64 = np_bgr_to_png_b64(bgr)
            md = call_api(args.api_endpoint, build_payload_image(b64, f"page_{i}.png",
                                                                 args.engine, args.langs_easy, args.langs_tess))
            if not md.strip() and args.try_secondary:
                sec = ("tesseract" if args.engine=="easyocr" else "easyocr")
                md = call_api(args.api_endpoint, build_payload_image(b64, f"page_{i}.png",
                                                                     sec, args.langs_easy, args.langs_tess))
            if md.strip():
                out_pages.append(f"<!-- page-break -->\n{md}")
                continue

        if args.local:
            if args.engine == "easyocr":
                txt = ocr_local_easyocr(bgr, args.langs_easy, args.gpu, args.fast)
                if (not txt.strip()) and args.try_secondary:
                    txt = ocr_local_tesseract(bgr, args.langs_tess, args.tess_config)
            else:
                txt = ocr_local_tesseract(bgr, args.langs_tess, args.tess_config)
                if (not txt.strip()) and args.try_secondary:
                    txt = ocr_local_easyocr(bgr, args.langs_easy, args.gpu, args.fast)
            out_pages.append(f"## Página {i}\n\n{txt or '*(sem texto detectável)*'}")

    if out_pages:
        md_all = "\n\n".join(out_pages).strip()
        return acessibilizar_md(md_all) if args.accessible else md_all
    return ""

# ------- Main / CLI -------
def main(argv=None):
    p = argparse.ArgumentParser(prog="ocr-cli", description="OCR CLI — EasyOCR + Tesseract")
    p.add_argument("arquivo", help="PDF ou imagem")
    p.add_argument("--engine", choices=["easyocr","tesseract"], default="easyocr", help="motor principal")
    p.add_argument("--try-secondary", action="store_true", help="tentar motor secundário se vier vazio")
    p.add_argument("--api", dest="api", action="store_true", help="ativar uso da API")
    p.add_argument("--no-api", dest="api", action="store_false", help="desativar uso da API")
    p.set_defaults(api=True)
    p.add_argument("--local", dest="local", action="store_true", help="ativar OCR local")
    p.add_argument("--no-local", dest="local", action="store_false", help="desativar OCR local")
    p.set_defaults(local=True)
    p.add_argument("--pre", choices=["none","basic","medium"], default="none", help="pré-processamento padronizado: none, basic (CLAHE) ou medium (contraste/brilho + cinza)")
    p.add_argument("--boost-alpha", type=float, default=1.25, help="contraste (>=1) usado em boost-gray")
    p.add_argument("--boost-beta", type=float, default=12, help="brilho [-255..255] usado em boost-gray")
    p.add_argument("--dpi", type=int, default=300, help="DPI para PDF→imagem (fallback por página)")
    p.add_argument("--langs-easy", default="pt,en", help="idiomas EasyOCR (csv)")
    p.add_argument("--langs-tess", default="por+eng", help="idiomas Tesseract")
    p.add_argument("--tess-cmd", default=None, help="caminho do executável do Tesseract (Windows)")
    p.add_argument("--tess-config", default="--oem 3 --psm 6", help="config Tesseract")
    p.add_argument("--api-endpoint", action="append", default=["http://200.137.132.64:5005/v1/convert/source"], help="endpoint(s) da API")
    p.add_argument("--out", choices=["md","json","txt"], default="md", help="formato de saída")
    p.add_argument("--accessible", action="store_true", help="adaptar Markdown para leitura de tela")
    p.add_argument("--gpu", action="store_true", help="tentar GPU no EasyOCR (se disponível)")
    p.add_argument("--fast", action="store_true", help="EasyOCR rápido (detail=0/greedy)")
    # TTS
    p.add_argument("--tts", action="store_true", help="falar status por voz")
    p.add_argument("--tts-rate", type=int, default=170, help="velocidade TTS")
    p.add_argument("--tts-volume", type=float, default=1.0, help="volume TTS (0..1)")

    args = p.parse_args(argv)

    # Preparos
    args.langs_easy = [s.strip() for s in args.langs_easy.split(",") if s.strip()]
    if pytesseract and args.tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tess_cmd

    speak("Iniciando leitura de documento.", args.tts, args.tts_rate, args.tts_volume)
    t0 = time.time()

    if not os.path.exists(args.arquivo):
        print("(!) Caminho inválido."); speak("Caminho inválido.", args.tts, args.tts_rate, args.tts_volume); sys.exit(2)

    if is_pdf(args.arquivo):
        res = process_pdf(args.arquivo, args)
    elif is_image_path(args.arquivo) or is_valid_image(args.arquivo):
        res = process_image(args.arquivo, args)
    else:
        print("(!) Use PDF ou uma imagem válida."); speak("Formato não suportado.", args.tts, args.tts_rate, args.tts_volume); sys.exit(2)

    dt = time.time() - t0
    if not res:
        print("(!) Não foi possível extrair texto."); speak("Não foi possível extrair texto.", args.tts, args.tts_rate, args.tts_volume); sys.exit(1)

    # Acessibilidade aplicada no salvamento
    if args.accessible and res:
        try:
            res = acessibilizar_md(res)
            print("[OK] Acessibilidade aplicada (marcação de títulos, imagens e quebras).")
            speak("Acessibilidade aplicada com sucesso.", args.tts, args.tts_rate, args.tts_volume)
        except Exception as e:
            print(f"[Aviso] Erro ao aplicar acessibilidade: {e}")
            speak("Erro ao aplicar acessibilidade.", args.tts, args.tts_rate, args.tts_volume)

    # Saída (único bloco)
    base = os.path.splitext(os.path.basename(args.arquivo))[0]
    if args.out == "md":
        outp = f"{base}_ocr{'_acc' if args.accessible else ''}.md"
        with open(outp, "w", encoding="utf-8") as w:
            w.write(res)
    elif args.out == "json":
        outp = f"{base}_ocr.json"
        with open(outp, "w", encoding="utf-8") as w:
            json.dump({"arquivo": args.arquivo, "conteudo": res}, w, ensure_ascii=False, indent=2)
    else:
        outp = f"{base}_ocr.txt"
        with open(outp, "w", encoding="utf-8") as w:
            w.write(res)

    print(f"[OK] Saída salva em: {outp} | tempo: {dt:.2f}s")
    speak("Processamento concluído.", args.tts, args.tts_rate, args.tts_volume)

if __name__ == "__main__":
    main()
