"""
Relatório automático de imagens para análise de OCR
Autor: Thiago Matias da Silva (UFMA)
Descrição:
    Gera um relatório com informações técnicas (OSD, tamanho, resolução, confiança)
    para avaliar a qualidade de imagens destinadas ao OCR.
    Agora com 4 pré-processamentos: none, basic, medium e alt (deskew + binarização).
"""

import os
import csv
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ExifTags, ImageOps
import pytesseract

# ====== CONFIGURAÇÕES ======
# Caminho do Tesseract (ajuste se necessário)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Pastas
PASTA_IMAGENS = "input_imgs"
PASTA_SAIDA_PREPROC = "output_preproc"
ARQUIVO_RELATORIO = "relatorio_imagens.csv"

# ====== UTILS ======
VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

def garantir_pastas():
    if not os.path.exists(PASTA_IMAGENS):
        os.makedirs(PASTA_IMAGENS)
        print(f"Pasta '{PASTA_IMAGENS}' criada. Coloque as imagens lá e rode novamente.")
        raise SystemExit
    os.makedirs(PASTA_SAIDA_PREPROC, exist_ok=True)

def carregar_cv2_bgr(caminho: str) -> np.ndarray:
    img = cv2.imread(caminho, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Falha ao carregar imagem: {caminho}")
    return img

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def pil_to_bgr(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def exif_dpi(pil: Image.Image):
    """Tenta obter DPI via EXIF; retorna (xdpi, ydpi) ou (None, None)."""
    try:
        info = pil.info
        if "dpi" in info and isinstance(info["dpi"], tuple):
            return info["dpi"][0], info["dpi"][1]
        exif = pil.getexif()
        if exif:
            # Alguns arquivos usam 282/283 para X/YResolution
            tag_x = 282; tag_y = 283
            xdpi = exif.get(tag_x)
            ydpi = exif.get(tag_y)
            # Alguns retornam como (num, den)
            def _conv(v):
                if isinstance(v, tuple) and len(v) == 2 and v[1] != 0:
                    return float(v[0]) / float(v[1])
                if isinstance(v, (int, float)):
                    return float(v)
                return None
            return _conv(xdpi), _conv(ydpi)
    except Exception:
        pass
    return None, None

def salvar_bgr(path_out: str, bgr: np.ndarray):
    ok = cv2.imwrite(path_out, bgr)
    if not ok:
        raise RuntimeError(f"Falha ao salvar imagem em {path_out}")

# ====== MÉTRICAS ======
def laplacian_variance(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def grayscale_contrast_std(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))

def ocr_text_and_conf(pil_img: Image.Image, tess_lang="por+eng", tess_config="--oem 3 --psm 6"):
    """Retorna (texto, mean_conf, num_palavras_validas). Conf média ignora -1."""
    # Texto geral
    texto = pytesseract.image_to_string(pil_img, lang=tess_lang, config=tess_config)
    # Conf por palavra
    data = pytesseract.image_to_data(pil_img, lang=tess_lang, config=tess_config, output_type=pytesseract.Output.DICT)
    confs = []
    for c in data.get("conf", []):
        try:
            v = float(c)
            if v >= 0:
                confs.append(v)
        except Exception:
            pass
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return texto, mean_conf, len(confs)

def osd_info(pil_img: Image.Image):
    """OSD (Tesseract): orientation/script/conf. Retorna dict com strings."""
    try:
        raw = pytesseract.image_to_osd(pil_img)
        linhas = [ln for ln in raw.split("\n") if ": " in ln]
        d = dict(x.split(": ", 1) for x in linhas)
        return {
            "orient_deg": d.get("Orientation in degrees", "N/A"),
            "rotate": d.get("Rotate", "N/A"),
            "orient_conf": d.get("Orientation confidence", "0"),
            "script": d.get("Script", "N/A"),
            "script_conf": d.get("Script confidence", "0"),
        }
    except Exception:
        return {"orient_deg": "N/A", "rotate": "N/A", "orient_conf": "0",
                "script": "N/A", "script_conf": "0"}

# ====== PRÉ-PROCESSAMENTOS ======
def preproc_none(bgr: np.ndarray) -> np.ndarray:
    return bgr

def preproc_basic(bgr: np.ndarray) -> np.ndarray:
    """Gray -> CLAHE -> Sharpen leve."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.4, blur, -0.4, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

def preproc_medium(bgr: np.ndarray, alpha: float = 1.25, beta: float = 12) -> np.ndarray:
    """Boost contraste/brilho + Gray (mantém 3 canais no retorno)."""
    boosted = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)
    gray = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def _deskew_gray(gray: np.ndarray) -> np.ndarray:
    """Deskew simples usando momentos (para ângulos pequenos)."""
    # binariza para achar momentos
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # inverter se fundo é escuro (heurística)
    if np.mean(thr) < 127:
        thr = cv2.bitwise_not(thr)
    coords = np.column_stack(np.where(thr > 0))
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    # cv2 retorna ângulos em [-90,0); converter para algo próximo de 0
    if angle < -45:
        angle = 90 + angle
    # rotaciona
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preproc_alt(bgr: np.ndarray) -> np.ndarray:
    """
    Alternativo: deskew (simples) + adaptive threshold + abertura morfológica.
    Ótimo para textos fracos/escaneados.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = _deskew_gray(gray)
    # equalização leve
    gray = cv2.equalizeHist(gray)
    # binarização adaptativa
    bin_img = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 8)
    # abertura para limpar ruído
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k, iterations=1)
    return cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)

PREPROCS = {
    "none": preproc_none,
    "basic": preproc_basic,
    "medium": preproc_medium,
    "alt": preproc_alt,
}

# ====== CLASSIFICAÇÃO ======
def classificar(mean_conf: float, qtd_chars: int, nitidez: float) -> str:
    """
    Heurística:
      - Boa: mean_conf >= 70 e qtd_chars >= 400 e nitidez >= 120
      - Média: mean_conf >= 50 e qtd_chars >= 150 e nitidez >= 60
      - Caso contrário: Ruim
    Ajuste os limites conforme seu acervo.
    """
    if mean_conf >= 70 and qtd_chars >= 400 and nitidez >= 120:
        return "Boa"
    if mean_conf >= 50 and qtd_chars >= 150 and nitidez >= 60:
        return "Média"
    return "Ruim"

# ====== RELATÓRIO ======
CABECALHO = [
    "Arquivo", "Variante",
    "Largura (px)", "Altura (px)", "DPI X", "DPI Y",
    "Orientação (graus)", "Rotacionar", "Conf. orientação",
    "Script", "Conf. script",
    "Chars", "Mean Conf (Tesseract)", "Palavras válidas",
    "Nitidez (LaplacianVar)", "Contraste (stdGray)",
    "Classificação", "Data/Hora",
    "Imagem pré-processada"
]

def processar_imagem(caminho_img: str, tess_lang="por+eng", tess_config="--oem 3 --psm 6"):
    bgr = carregar_cv2_bgr(caminho_img)
    h, w = bgr.shape[:2]
    pil_orig = bgr_to_pil(bgr)
    xdpi, ydpi = exif_dpi(pil_orig)
    # fallback de DPI
    if xdpi is None or ydpi is None:
        xdpi = xdpi if xdpi is not None else 300
        ydpi = ydpi if ydpi is not None else 300

    nome_arq = os.path.basename(caminho_img)
    base, _ = os.path.splitext(nome_arq)

    linhas = []
    for variante, fn in PREPROCS.items():
        try:
            bgr_pp = fn(bgr.copy())
        except Exception as e:
            print(f"[Aviso] Falha no pré-processamento '{variante}' para {nome_arq}: {e}")
            bgr_pp = bgr.copy()

        # salvar imagem pré-processada
        saida_img = os.path.join(PASTA_SAIDA_PREPROC, f"{base}__{variante}.png")
        try:
            salvar_bgr(saida_img, bgr_pp)
        except Exception as e:
            print(f"[Aviso] Não foi possível salvar {saida_img}: {e}")

        # métricas
        pil_pp = bgr_to_pil(bgr_pp)
        texto, mean_conf, n_words = ocr_text_and_conf(pil_pp, tess_lang=tess_lang, tess_config=tess_config)
        qtd_chars = len((texto or "").strip())

        osd = osd_info(pil_pp)
        orient_deg = osd["orient_deg"]
        rotate = osd["rotate"]
        orient_conf = osd["orient_conf"]
        script = osd["script"]
        script_conf = osd["script_conf"]

        nitidez = laplacian_variance(bgr_pp)
        contraste = grayscale_contrast_std(bgr_pp)

        qualidade = classificar(mean_conf, qtd_chars, nitidez)

        linhas.append([
            nome_arq, variante,
            w, h, f"{xdpi}", f"{ydpi}",
            orient_deg, rotate, orient_conf,
            script, script_conf,
            qtd_chars, f"{mean_conf:.2f}", n_words,
            f"{nitidez:.2f}", f"{contraste:.2f}",
            qualidade, datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            os.path.relpath(saida_img)
        ])
    return linhas

def main():
    garantir_pastas()

    todas_linhas = []
    for nome in os.listdir(PASTA_IMAGENS):
        if not nome.lower().endswith(VALID_EXT):
            continue
        caminho = os.path.join(PASTA_IMAGENS, nome)
        try:
            linhas = processar_imagem(caminho)
            todas_linhas.extend(linhas)
        except Exception as e:
            print(f"[Erro] {nome}: {e}")

    # Escreve relatório
    with open(ARQUIVO_RELATORIO, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CABECALHO)
        w.writerows(todas_linhas)

    print(f"✅ Relatório gerado com sucesso: {ARQUIVO_RELATORIO}")
    print(f"🖼️ Imagens pré-processadas salvas em: {os.path.abspath(PASTA_SAIDA_PREPROC)}")

if __name__ == "__main__":
    main()
