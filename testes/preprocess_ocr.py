from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np
from pathlib import Path

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# CONFIGURAÇÃO — Caminho da imagem
# Coloque o nome completo ou só o nome se estiver na mesma pasta
img_path = Path(r"C:\Projetos\docling\cpf.png")  # <-- ajuste aqui se o nome for diferente

if not img_path.exists():
    raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# Funções utilitárias
def upscale_if_small(im: Image.Image, min_height=1000) -> Image.Image:
    """Aumenta o tamanho se a imagem for pequena (melhor para OCR)."""
    w, h = im.size
    if h < min_height:
        scale = min_height / h
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return im


def enhance_color_for_view(im: Image.Image) -> Image.Image:
    """Melhora contraste, nitidez e remove ruído."""
    im = upscale_if_small(im, 1100)
    im = im.filter(ImageFilter.MedianFilter(size=3))
    im = ImageOps.autocontrast(im, cutoff=1)
    im = ImageEnhance.Brightness(im).enhance(1.05)
    im = ImageEnhance.Contrast(im).enhance(1.2)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.4, percent=180, threshold=3))
    return im


def otsu_threshold(gray_np: np.ndarray) -> int:
    """Calcula automaticamente o melhor limiar de binarização."""
    hist, _ = np.histogram(gray_np.flatten(), bins=256, range=(0, 256))
    total = gray_np.size
    sum_total = np.dot(np.arange(256), hist)
    sumB = 0.0
    wB = 0.0
    var_max = -1.0
    threshold = 0
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t
    return threshold


def make_ocr_binarized(im_color: Image.Image) -> Image.Image:
    """Cria versão preto e branco ideal para OCR."""
    gray = im_color.convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.GaussianBlur(radius=0.6))
    g = np.array(gray)
    thr = otsu_threshold(g)
    bw = (g >= thr).astype(np.uint8) * 255
    if bw.mean() < 127:  # fundo escuro -> inverte
        bw = 255 - bw
    return Image.fromarray(bw, mode="L")


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# AQUI o processamento começa
img = Image.open(img_path).convert("RGB")

clean = enhance_color_for_view(img)
ocr = make_ocr_binarized(clean)

clean_path = img_path.with_name(img_path.stem + "_clean.png")
ocr_path = img_path.with_name(img_path.stem + "_ocr.png")

clean.save(clean_path)
ocr.save(ocr_path)

print(f"Imagens geradas com sucesso:\n - {clean_path}\n - {ocr_path}")
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
