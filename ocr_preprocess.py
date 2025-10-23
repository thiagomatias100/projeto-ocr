# ocr_preprocess.py
# Módulo de pré-processamento OCR (com opção MSER)
# Autor: Thiago Matias da Silva (UFMA)

import cv2
import numpy as np
import pytesseract
from typing import Tuple

# ---------------- CONFIGURAÇÃO PADRÃO ----------------

TESS_CONFIG_DOC = '--oem 3 --psm 6 -c tessedit_char_blacklist=§©®™'

# ---------------- FUNÇÕES AUXILIARES ----------------

def _resize_for_ocr(bgr: np.ndarray, target_short: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    short = min(h, w)
    if short >= target_short:
        return bgr
    scale = target_short / float(short)
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

def _deskew_by_lines(gray_like: np.ndarray) -> Tuple[np.ndarray, float]:
    """Corrige inclinação via linhas detectadas (Hough)."""
    _, bin_otsu = cv2.threshold(gray_like, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    H, W = bin_otsu.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, W // 80), 3))
    closed = cv2.morphologyEx(bin_otsu, cv2.MORPH_CLOSE, kernel, 1)
    lines = cv2.HoughLinesP(255 - closed, 1, np.pi / 180, threshold=120,
                            minLineLength=W // 8, maxLineGap=10)
    angle = 0.0
    if lines is not None and len(lines) > 0:
        angs = []
        for x1, y1, x2, y2 in lines[:, 0]:
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if ang < -45: ang += 90
            if ang > 45: ang -= 90
            angs.append(ang)
        if angs:
            angle = float(np.median(angs))

    if abs(angle) < 0.3:
        return gray_like, 0.0

    M = cv2.getRotationMatrix2D((gray_like.shape[1] / 2, gray_like.shape[0] / 2), angle, 1.0)
    rot = cv2.warpAffine(gray_like, M, (gray_like.shape[1], gray_like.shape[0]),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rot, angle

def _autocrop_white(binimg: np.ndarray, border: int = 6) -> np.ndarray:
    rows = np.where(binimg.min(axis=1) < 255)[0]
    cols = np.where(binimg.min(axis=0) < 255)[0]
    if rows.size and cols.size:
        r0, r1 = max(0, rows[0] - border), min(binimg.shape[0], rows[-1] + border)
        c0, c1 = max(0, cols[0] - border), min(binimg.shape[1], cols[-1] + border)
        return binimg[r0:r1, c0:c1]
    return binimg

# ---------------- MSER (Detector de Regiões) ----------------

def _apply_mser(gray: np.ndarray) -> np.ndarray:
    """Destaca regiões de texto com MSER e retorna máscara binária."""
    mser = cv2.MSER_create(_min_area=60, _max_area=8000, _delta=5)
    regions, _ = mser.detectRegions(gray)
    mask = np.zeros_like(gray)
    for p in regions:
        hull = cv2.convexHull(p.reshape(-1, 1, 2))
        cv2.drawContours(mask, [hull], -1, 255, -1)
    # limpeza leve (remove ruído e une textos próximos)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), 1)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)), 1)
    return mask

# ---------------- PIPELINE PRINCIPAL ----------------

def preprocess_for_ocr(
    bgr: np.ndarray,
    mode: str = "geral",
    target_short: int = 1400,
    use_sauvola: bool = True,
    use_mser: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna: (binario_para_tesseract, imagem_enhanced_color)
    """
    bgr = _resize_for_ocr(bgr, target_short=target_short)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 31)
    bg = np.clip(bg, 1, 255)
    norm = cv2.divide(gray, bg, scale=255)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(norm)
    if mode == "documento":
        den = cv2.bilateralFilter(clahe, d=7, sigmaColor=55, sigmaSpace=55)
    else:
        den = cv2.bilateralFilter(clahe, d=5, sigmaColor=40, sigmaSpace=40)

    blur = cv2.GaussianBlur(den, (0, 0), 1.2 if mode == "documento" else 1.0)
    sharp = cv2.addWeighted(den, 1.6, blur, -0.6, 0)
    deskewed, _ = _deskew_by_lines(sharp)

    # 5) Binarização
    if use_sauvola:
        g = deskewed.astype(np.float32)
        win = 31 if mode == "geral" else 25
        mean = cv2.boxFilter(g, -1, (win, win), normalize=True)
        sqm = cv2.boxFilter(g * g, -1, (win, win), normalize=True)
        var = np.clip(sqm - mean * mean, 0, None)
        std = np.sqrt(var)
        k, R = (0.34, 128.0)
        thr = mean * (1 + k * ((std / R) - 1))
        binimg = (g > thr).astype(np.uint8) * 255
    else:
        _, binimg = cv2.threshold(deskewed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if mode == "documento":
        binimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)
        binimg = cv2.morphologyEx(binimg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)

    if use_mser:
        mser_mask = _apply_mser(gray)
        binimg = cv2.bitwise_and(binimg, binimg, mask=mser_mask)

    binimg = _autocrop_white(binimg, border=6)
    enhanced_color = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)

    return np.ascontiguousarray(binimg.astype(np.uint8)), enhanced_color

# ---------------- OCR DOCUMENTO ----------------

def ocr_tesseract_image_documento(
    bgr: np.ndarray,
    lang_str: str = "por+eng+spa",
    config: str = None,
    use_mser: bool = False
) -> str:
    binimg, _ = preprocess_for_ocr(bgr, mode="documento", use_mser=use_mser)
    cfg = config if config is not None else TESS_CONFIG_DOC
    txt = pytesseract.image_to_string(binimg, lang=lang_str, config=cfg)
    return (txt or "").replace("\r", "").strip()

# ---------------- PREVIEW OPCIONAL ----------------

def preview_preprocess(bgr: np.ndarray, mode: str = "documento", use_mser: bool = False):
    import matplotlib.pyplot as plt
    binimg, enh = preprocess_for_ocr(bgr, mode=mode, use_mser=use_mser)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)); ax[1].set_title("Enhanced"); ax[1].axis("off")
    ax[2].imshow(binimg, cmap="gray"); ax[2].set_title("Binary (com MSER)" if use_mser else "Binary"); ax[2].axis("off")
    plt.tight_layout(); plt.show()
